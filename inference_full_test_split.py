# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE


import json
import os
import subprocess
import time
import warnings
warnings.filterwarnings("ignore")

from functools import partial
import multiprocessing as mp

import soundfile as sf
import matplotlib.image
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
import ASR.asr_models as asr_models
from dataloaders.dataset_lipvoicer import LipVoicerDataset
from dataloaders.stft import denormalise_mel
from hifi_gan.generator import Generator as Vocoder
from hifi_gan import utils as vocoder_utils
from hifi_gan.env import AttrDict

from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory


def sampling(net, seed, diffusion_hyperparams,
            w_video, condition=None,
            asr_guidance_net=None,
            w_asr=None,
            asr_start=None,
            guidance_text=None,
            tokenizer=None,
            decoder=None
            ):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the model
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated melspec(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T

    # tokenize textaaaaaaa
    if asr_guidance_net is not None:
        text_tokens = torch.LongTensor(tokenizer.encode(guidance_text))
        text_tokens = text_tokens.unsqueeze(0).cuda()

    mouthroi, face_image = condition
    sample_step = T
    timesteps = torch.linspace(1.0,0.0, sample_step+1)
    torch.manual_seed(seed)
    eps0 = torch.randn(mouthroi.shape[0], 80, mouthroi.shape[2]*4, device='cuda')
    x = eps0.clone().cuda()
    with torch.no_grad():
        for i in range(sample_step):
            t = timesteps[i]
            t_next = timesteps[i+1]
            dt = t - t_next
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            # if i == 0:
            #     noise = z.clone()
            # else:
            #     noise = (z - (1 - diffusion_steps) * (v_pred + noise))

            v_pred_con    = net.sample(x, eps0, mouthroi, face_image, guidance_text, diffusion_steps, cond_drop_prob=0)   # predict \epsilon according to \epsilon_\theta
            v_pred_uncond = net.sample(x, eps0, mouthroi, face_image, guidance_text, diffusion_steps, cond_drop_prob=1)

            # v_pred = v_pred_uncond + w_video * (v_pred_con - v_pred_uncond)
            v_pred = (1+ w_video) * v_pred_con - w_video * v_pred_uncond
            x = x - v_pred * dt

    return x


@torch.no_grad()
def generate(
        rank,
        seed,
        diffusion_cfg,
        model_cfg,
        dataset_cfg,
        text_cfg,
        attention_cfg,
        ckpt_path,
        w_video=0,
        w_asr=1.1,
        asr_start=250,
        save_dir=None,
        lipread_text_dir=None,
        **kwargs
    ):

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank)

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters

    # predefine MelGen model
    builder = ModelBuilder()
    net_lipreading = builder.build_lipreadingnet()
    net_facial = builder.build_facial(fc_out=128, with_fc=True)
    net_diffwave = builder.build_diffwave_model(model_cfg)
    net_text = builder.build_text_model(text_cfg)
    net_attention = builder.build_attention_model(attention_cfg)
    net_fusion  = builder.build_fusion_model()
    net = AudioVisualModel((net_lipreading, net_facial, net_diffwave,net_text,net_attention,net_fusion)).cuda()

    print_size(net)
    net.eval()

    # load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # net.load_state_dict(checkpoint['model_state_dict'])
        net.load_state_dict(checkpoint['ema_state_dict'])
        print('Successfully loaded MelGen checkpoint')
    except:
        raise Exception('No valid model found')

    if save_dir is None:
        save_dir = os.getcwd()
    output_directory = os.path.join(save_dir, 'generated_mels',str(seed))
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("saving to output directory", output_directory)

    if 'LRS2' in dataset_cfg.videos_dir or 'lrs2' in dataset_cfg.videos_dir:
        ds_name = 'LRS2'
    else:
        ds_name = 'LRS3'

    print('Loading ASR, tokenizer and decoder')
    asr_guidance_net, tokenizer, decoder = asr_models.get_models(ds_name)

    # HiFi-GAN
    print('Load HiFi-GAN')
    config_file = 'hifi_gan/config.json'
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Vocoder(h).cuda()
    checkpoint_file = 'hifi_gan/g_02400000'
    state_dict_g = vocoder_utils.load_checkpoint(checkpoint_file, 'cuda')
    vocoder.load_state_dict(state_dict_g['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()

    dataset = LipVoicerDataset('test', **dataset_cfg)

    guidance_dir_name = f'w1={w_video}'
    _output_directory = os.path.join(output_directory, ds_name,str(diffusion_hyperparams['T']) ,guidance_dir_name)
    os.makedirs(_output_directory, exist_ok=True)
    print("saving to output directory", _output_directory)

    for i in tqdm(range(len(dataset))):
        gt_melspec, gt_audio, mouthroi, face_image, gt_text, video_id = dataset[i]
        check_path =f"{os.path.join(_output_directory, video_id)}.wav"
        # if  os.path.exists(check_path):
        #     print(f"{check_path} pass")
        #     continue
        with open(os.path.join(lipread_text_dir, video_id+".txt"), 'r') as f:
            text = f.readline()
        gt_melspec = denormalise_mel(gt_melspec)
        gt_melspec = gt_melspec.unsqueeze(0)
        mouthroi = mouthroi.unsqueeze(0)        # add batch dimension
        face_image = face_image.unsqueeze(0)

        # inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        melspec = sampling(net, seed,
                        diffusion_hyperparams,
                        w_video,
                        condition=(mouthroi.cuda(), face_image.cuda()),
                        asr_guidance_net=asr_guidance_net,
                        w_asr=w_asr,
                        asr_start=asr_start,
                        guidance_text=text,
                        tokenizer=tokenizer,
                        decoder=decoder
                        )
        melspec = denormalise_mel(melspec)
        end.record()
        torch.cuda.synchronize()
        print('generated {} in {} seconds'.format(video_id, int(start.elapsed_time(end)/1000)))

        video_dir = video_id.split('/')[0]
        os.makedirs(os.path.join(_output_directory, video_dir), exist_ok=True)

        # generate audio from melspec
        audio = vocoder(melspec)
        audio = audio.squeeze()
        audio = audio / 1.1 / audio.abs().max()
        audio = audio.cpu().numpy()
        sf.write(os.path.join(_output_directory, video_id + '.wav'), audio, 16000)
        sf.write(os.path.join(_output_directory, video_id + '_gt.wav'), audio, 16000)
        # attach audio to video
        in_video_filename = os.path.join(dataset_cfg.dataset_root, 'test', video_id+".mp4")
        subprocess.call(f"ffmpeg -y -i {in_video_filename} \
                    -i {os.path.join(_output_directory, video_id + '.wav')} \
                    -c:v copy -map 0:v:0 -map 1:a:0 \
                    {os.path.join(_output_directory, video_id + '.mp4')}", shell=True)

        # save as file
        melspec = melspec.squeeze(0).cpu()
        torch.save(melspec, os.path.join(_output_directory, video_id + '.wav.spec'))

        # save as image
        melspec = melspec.numpy()
        gt_melspec = gt_melspec.squeeze(0).numpy()
        matplotlib.image.imsave(os.path.join(_output_directory, video_id+'.png'), melspec[::-1])
        matplotlib.image.imsave(os.path.join(_output_directory, video_id+'_gt.png'), gt_melspec[::-1])


        # save text
        text_filename = os.path.join(_output_directory, video_id+'.txt')
        with open(text_filename, 'w') as f:
            f.write("gt       :  " + gt_text+"\n")
            f.write("lipreader:  " + text)

    return


@hydra.main(version_base=None, config_path="configs/", config_name="config_test")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys
    print(cfg.generate.w_video)
    generate(cfg.rank,cfg.seed,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.melgen,
        dataset_cfg=cfg.dataset,
        text_cfg = cfg.text,
        attention_cfg= cfg.attention,
        **cfg.generate,
    )



if __name__ == "__main__":
    main()
