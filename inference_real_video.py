# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE


import json
import os
import random
import subprocess
import time
import warnings

from dataloaders.video_reader import VideoReader
warnings.filterwarnings("ignore")

from functools import partial
import multiprocessing as mp

from PIL import Image
import soundfile as sf
import numpy as np
import torch
import hydra
import gc
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from models.model_builder import ModelBuilder
from models.audiovisual_model import AudioVisualModel
import ASR.asr_models as asr_models
from dataloaders.dataset_lipvoicer import LipVoicerDataset
from dataloaders.stft import denormalise_mel
import torchvision.transforms as transforms
from dataloaders.lipreading_utils import *
from hifi_gan.generator import Generator as Vocoder
from hifi_gan import utils as vocoder_utils
from hifi_gan.env import AttrDict
from mouthroi_processing import crop_and_infer

from utils import print_size, calc_diffusion_hyperparams, local_directory, find_max_epoch
from pathlib import Path

def get_mouthroi_transform():
    # -- preprocess for the video stream
    # -- LRW config
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    preprocessing = Compose([
                            Normalize( 0.0,255.0 ),
                            CenterCrop(crop_size),
                            Normalize(mean, std) ])
    return preprocessing


def get_face_image_transform():
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    vision_transform_list = [transforms.Resize(224), transforms.ToTensor(), normalize]
    vision_transform = transforms.Compose(vision_transform_list)
    return vision_transform


def load_frame(clip_path):
        video_reader = VideoReader(clip_path, 1)
        start_pts, time_base, total_num_frames = video_reader._compute_video_stats()
        end_frame_index = total_num_frames - 1
        if end_frame_index < 0:
            clip = video_reader.read_video_only(start_pts, 1)
        else:
            clip = video_reader.read_video_only(random.randint(0, end_frame_index) * time_base, 1)
        frame = Image.fromarray(np.uint8(clip[0].to_rgb().to_ndarray())).convert('RGB')
        return frame


def sampling(net, diffusion_hyperparams,
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
    sample_step = 100

    timesteps = torch.linspace(1.0,0.0, sample_step+1)
    eps0 = torch.randn(mouthroi.shape[0], 80, mouthroi.shape[2]*4, device='cuda')
    x = eps0.clone().cuda()
    length_input = x.shape[2]
    with torch.no_grad():
        for i in tqdm(range(sample_step)):
            t = timesteps[i]
            t_next = timesteps[i+1]
            dt = t - t_next
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            v_pred_con    = net.sample(x, eps0, mouthroi, face_image, guidance_text, diffusion_steps, cond_drop_prob=0)   # predict \epsilon according to \epsilon_\theta
            v_pred_uncond = net.sample(x, eps0, mouthroi, face_image, guidance_text, diffusion_steps, cond_drop_prob=1)

            v_pred = v_pred_uncond + w_video * (v_pred_con - v_pred_uncond)

            x = x - v_pred * dt


            if t % 10 == 0:
                if asr_guidance_net is not None and t <= asr_start:
                    inputs = x, length_input
                    outputs_ao = asr_guidance_net(inputs, diffusion_steps)["outputs"]
                    preds_ao = decoder(outputs_ao)[0]
                    print("pred:"  ,preds_ao, '\n')
                    print("target:",guidance_text, '\n')


    return x


@torch.no_grad()
def generate(
        ckpt_num,
        generate_cfg,
        diffusion_cfg,
        model_cfg,
        text_cfg,
        attention_cfg,
        **kwargs
    ):

    torch.cuda.set_device(1)

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters

    # Build MelGen model
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

    # Load pretrained MelGen model
    try:
        ckpt_path = generate_cfg['ckpt_path']
        # ckpt_path = f"{ckpt_path}{ckpt_num}.pkl"
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(checkpoint['ema_state_dict'])
        print('Successfully loaded MelGen checkpoint')
    except:
        raise Exception('No valid model found')

    video_filename = generate_cfg['video_path']
    output_directory = f"{video_filename.split('/')[-2]}_{video_filename.split('/')[-1].replace('.mp4', '')}/{ckpt_num}"
    if generate_cfg['save_dir']:
        save_dir = generate_cfg['save_dir']
    else:
        save_dir = os.getcwd()
    output_directory = os.path.join(save_dir, output_directory)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("saving to output directory", output_directory)

    print('Loading ASR, tokenizer and decoder')
    asr_guidance_net, tokenizer, decoder = asr_models.get_models('LRS2')

    w_video = generate_cfg['w_video']
    w_asr = generate_cfg['w_asr']
    asr_start = generate_cfg['asr_start']
    guidance_dir_name = f"w1={w_video}"
    guidance_dir_name += f"_w2={w_asr}_asr_start={asr_start}"
    output_directory = os.path.join(output_directory, guidance_dir_name)
    os.makedirs(output_directory, exist_ok=True)
    print("saving to output directory", output_directory)

    print(f"Cropping lip region and predicting text")
    # get mouthcrop and text prediction for the video
    # video_files = list(sorted(Path(generate_cfg["video_path"]).glob("**/*.mp4")))
    # for video_path in video_files:
    mouthroi, text ,face_crop_160= crop_and_infer.main(generate_cfg["video_path"], output_directory)
    mouthroi_transform = get_mouthroi_transform()
    mouthroi = mouthroi_transform(mouthroi)
    mouthroi = mouthroi.unsqueeze(0).unsqueeze(0)

    face_image_transform = get_face_image_transform()
    index = random.randint(0, face_crop_160.shape[0])
    face_image = face_crop_160[index]
    # face_image = load_frame(video_filename)
    face_image = Image.fromarray(face_image)
    face_image = face_image_transform(face_image)
    face_image = face_image.unsqueeze(0)

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    print('Generating melspectrogram')
    melspec = sampling(net,
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
    print('generated melspec in {} seconds'.format(int(start.elapsed_time(end)/1000)))

    # save melspec
    video_name = video_filename.split('/')[-1].replace(".mp4", "")
    torch.save(melspec.squeeze(0).cpu(), os.path.join(output_directory, video_name + '.wav.spec'))

    # generate audio from melspec
    # HiFi-GAN
    print('Loading HiFi-GAN')
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

    print('Vocoding')
    audio = vocoder(melspec)
    audio = audio.squeeze()
    audio = audio / 1.1 / audio.abs().max()
    audio = audio.cpu().numpy()
    sf.write(os.path.join(output_directory, video_name + '.wav'), audio, 16000) #save generated wave

    # attach audio to video
    subprocess.call(f"ffmpeg -y -i {video_filename} \
                -i {os.path.join(output_directory, video_name + '.wav')} \
                -c:v copy -map 0:v:0 -map 1:a:0 \
                {os.path.join(output_directory, video_name + '.mp4')}", shell=True) #기존 비디오에 생성된 오디오 넣음

    return


@hydra.main(version_base=None, config_path="configs/", config_name="config_test")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    #ckpt_lst = os.listdir(f"/home/gunwoo/gunwoo/LipVoicer_Origin/exp/LRS2/wnet_h512_d12_T400_betaT0.02/checkpoint")
    # for ckpt_num in ckpt_lst:

    generate(
            "0",
            generate_cfg=cfg.generate,
            diffusion_cfg=cfg.diffusion,
            model_cfg=cfg.melgen,
            text_cfg = cfg.text,
            attention_cfg = cfg.attention
    )

if __name__ == "__main__":
    main()
