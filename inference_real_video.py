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


def sampling(net, seed, diffusion_hyperparams, w_video, condition, guidance_text):
    r"""
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the model
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated melspec(s) in torch.tensor, shape=size
    """

    mouthroi, face_image = condition
    sample_step = diffusion_hyperparams["T"]
    timesteps = torch.linspace(1.0, 0.0, sample_step + 1)
    torch.manual_seed(seed)
    eps0 = torch.randn(mouthroi.shape[0], 80, mouthroi.shape[2] * 4, device="cuda")
    x = eps0.clone()
    with torch.no_grad():
        for index in tqdm(range(sample_step)):
            timestep = timesteps[index]
            dt = timestep - timesteps[index + 1]
            steps = timestep * torch.ones((x.shape[0], 1), device=x.device)
            conditional = net.sample(
                x, eps0, mouthroi, face_image, guidance_text, steps, cond_drop_prob=0
            )
            unconditional = net.sample(
                x, eps0, mouthroi, face_image, guidance_text, steps, cond_drop_prob=1
            )
            x = x - ((1 + w_video) * conditional - w_video * unconditional) * dt

    return x


@torch.no_grad()
def generate(
        rank,
        seed,
        generate_cfg,
        diffusion_cfg,
        model_cfg,
        text_cfg,
        attention_cfg,
        **kwargs
    ):

    torch.cuda.set_device(rank)

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
    output_directory = Path(video_filename).stem
    if generate_cfg['save_dir']:
        save_dir = generate_cfg['save_dir']
    else:
        save_dir = os.getcwd()
    output_directory = os.path.join(save_dir, output_directory)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("saving to output directory", output_directory)

    w_video = generate_cfg['w_video']
    guidance_dir_name = f"w1={w_video}"
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
    melspec = sampling(
        net,
        seed,
        diffusion_hyperparams,
        w_video,
        condition=(mouthroi.cuda(), face_image.cuda()),
        guidance_text=text,
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


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    generate(
        cfg.rank,
        cfg.seed,
        generate_cfg=cfg.generate,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.melgen,
        text_cfg=cfg.text,
        attention_cfg=cfg.attention,
    )

if __name__ == "__main__":
    main()
