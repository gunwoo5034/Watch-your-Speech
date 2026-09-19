# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE

import os
import wandb
from copy import deepcopy
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import warnings
warnings.filterwarnings("ignore")
from functools import partial
import multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import hydra
# import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataloaders import dataloader
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory, plot_melspec

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from inference_melgen import generate

from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel

def init_ema(model: torch.nn.Module) -> torch.nn.Module:
    ema_model = deepcopy(model).eval()        # EMA는 평가 모드로 유지
    for p in ema_model.parameters():
        p.requires_grad_(False)               # 학습 비활성화 (read-only)
    return ema_model

@torch.no_grad()
def ema_update(model, ema_model, beta: float):
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k in esd.keys():
        if k in msd:
            if torch.is_floating_point(esd[k]):
                esd[k].mul_(beta).add_(msd[k].detach().to(device=esd[k].device, dtype=esd[k].dtype),
                                        alpha=1 - beta)
            else:
                esd[k].copy_(msd[k])


def main_worker(rank , num_gpus, group_name, cfg):
    print(f"Rank: {rank}, num_gpus: {num_gpus}, group_name: {group_name}")

    if num_gpus >= 2:
        dist_cfg = cfg.pop("distributed")
        init_distributed(rank, num_gpus=num_gpus, group_name= group_name, **dist_cfg)
    print(f"2Rank: {rank}, num_gpus: {num_gpus}, group_name: {group_name}")
    train(
        rank=rank, num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.melgen,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        text_cfg = cfg.text,
        **cfg.train,
    )
def distributed_train(rank, num_gpus, group_name, cfg):

    # Distributed running initialization
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    train(
        rank=rank, num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.melgen,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        text_cfg = cfg.text,
        attention_cfg= cfg.attention,
        **cfg.train,
    )

def train(
    rank, num_gpus, save_dir,
    diffusion_cfg, model_cfg, dataset_cfg, generate_cfg,text_cfg,attention_cfg, # dist_cfg, wandb_cfg, # train_cfg,
    ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging,
    learning_rate, batch_size_per_gpu,
    name=None,
):

    """
    Parameters:
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automitically selects the maximum iteration if 'max' is selected
    n_iters (int):                  number of iterations to train, default is 1M
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    batch_size_per_gpu (int):       batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    name (str):                     prefix in front of experiment name
    """

    if rank == 0:
        writer = SummaryWriter(log_dir='logs')
        wandb.init(project="Watch-Your-Speech")
        wandb.config.update({
            "learning_rate": learning_rate,
            "batch_size": batch_size_per_gpu * num_gpus,
            "n_iters": n_iters,
            "iters_per_ckpt": iters_per_ckpt,
            "iters_per_logging": iters_per_logging
        })
    #check point 위치
    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, save_dir, 'checkpoint')

    # map diffusion hyperparameters to gpu
    #T ,beta, alpha, alpha_bar, sigma 정의
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_cfg, fast=False)  # dictionary of all diffusion hyperparameters

    # load training data
    trainloader= dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus)
    print('Data loaded')


    # predefine model
    builder = ModelBuilder()
    net_lipreading = builder.build_lipreadingnet()
    net_facial = builder.build_facial(fc_out=128, with_fc=True)
    net_diffwave = builder.build_diffwave_model(model_cfg)
    net_text = builder.build_text_model(text_cfg)
    net_attention = builder.build_attention_model(attention_cfg)
    net_fusion  = builder.build_fusion_model()
    net = AudioVisualModel((net_lipreading, net_facial, net_diffwave,net_text,net_attention,net_fusion)).cuda()
    print_size(net, verbose=False)

    ema_model = AudioVisualModel(
        (builder.build_lipreadingnet(),
        builder.build_facial(fc_out=128, with_fc=True),
        builder.build_diffwave_model(model_cfg),
        builder.build_text_model(text_cfg),
        builder.build_attention_model(attention_cfg),
        builder.build_fusion_model())
    ).cuda()

    ema_model.load_state_dict(net.state_dict())   # 가중치 복사
    ema_model.eval()
    # ema = deepcopy(net).to(torch.float32).cuda()
    # ema_model = init_ema(net)
    ema_decay = 0.9999
    ema_warmup = 5000
    global_step = 0
    criterion = nn.MSELoss()

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,weight_decay=1e-2)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(checkpoint_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # HACK to reset learning rate
                optimizer.param_groups[0]['lr'] = learning_rate

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            print(f"Model checkpoint found at iteration {ckpt_iter}, but was not successfully loaded - training from scratch.")
            ckpt_iter = -1
    else:
        print('No valid checkpoint model found - training from scratch.')
        ckpt_iter = -1

    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        epoch_loss = 0.
        for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}') if rank==0 else trainloader:
        # for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
            #train_sampler.set_epoch(epoch= n_iter//len(trainloader))
            melspec, mouthroi, face_image ,text = data
            melspec, mouthroi, face_image = melspec.cuda(rank), mouthroi.cuda(rank), face_image.cuda(rank)

            net.train()
            # back-propagation
            optimizer.zero_grad()
            loss = training_loss(net, criterion, melspec, mouthroi, face_image,text, diffusion_hyperparams , rank)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            if n_iter > ema_warmup:
                ema_update(net, ema_model, ema_decay)
            epoch_loss += reduced_loss

            # output to log
            if n_iter % iters_per_logging == 0 and rank == 0:
                # save training loss to tensorboard
                print("iteration: {} \tloss: {}".format(n_iter, reduced_loss))
                wandb.log({"loss": reduced_loss, "iteration": n_iter})

            # save checkpoint
            if n_iter % iters_per_ckpt == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                           'ema_state_dict': ema_model.state_dict()},
                           os.path.join(checkpoint_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

                # Generate samples
                generate_cfg["ckpt_iter"] = n_iter
                samples = generate(
                    rank, # n_iter,
                    diffusion_cfg, model_cfg, dataset_cfg,text_cfg,attention_cfg,
                    name=name,
                    save_dir=save_dir,
                    ckpt_iter="max",
                    n_samples=generate_cfg.n_samples,
                    w_video=generate_cfg.w_video,
                )

                # send images to log
                for i, (mel, mel_gt) in enumerate(zip(*samples)):
                    writer.add_figure(f'spec/{i+1}', plot_melspec(mel[0].cpu().numpy()), n_iter)
                    writer.add_figure(f'spec/{i+1}_gt', plot_melspec(mel_gt[0].cpu().numpy()), n_iter)
                    wandb.log({f'spec/{i+1}': [wandb.Image(plot_melspec(mel[0].cpu().numpy())),
                                                wandb.Image(plot_melspec(mel_gt[0].cpu().numpy()))]})


            n_iter += 1
        if rank == 0:
            epoch_loss /= len(trainloader)
            writer.add_scalar('train_loss', epoch_loss, n_iter)
            wandb.log({"epoch_loss": epoch_loss, "epoch": n_iter // len(trainloader)})

    # Close logger
    if rank == 0:
        writer.close()
        wandb.finish()

def training_loss(net, loss_fn, melspec, mouthroi, face_image,text , diffusion_hyperparams ,rank):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """
    # Predict melspectrogram from visual features using diffusion model
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    B, C, L = melspec.shape  # B is batchsize, C=80, L is number of melspec frames
    # diffusion_steps = torch.randint(T, size=(B,1,1)).cuda(rank)  # randomly sample diffusion steps from 1~T
    # z = torch.normal(0, 1, size=melspec.shape).cuda(rank)
    # transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    cond_drop_prob = 0.1
    velocity_pred , target = net(melspec, mouthroi, face_image,text, diffusion_steps = None, cond_drop_prob = cond_drop_prob)

    return loss_fn(velocity_pred, target)



@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    if not os.path.isdir("exp/"):
        os.makedirs("exp/")
        os.chmod("exp/", 0o775)

    num_gpus = torch.cuda.device_count()
    print(f'there are {num_gpus} gpus')
    group_name=time.strftime("%Y%m%d-%H%M%S")
    # if num_gpus > 1:
    #     torch.multiprocessing.spawn(main_worker, args=(num_gpus, group_name, cfg,), nprocs=num_gpus, join=True, daemon=False)

    # else:
    #     main_worker(num_gpus, group_name, cfg,)

    train_fn = partial(
        distributed_train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        cfg=cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
