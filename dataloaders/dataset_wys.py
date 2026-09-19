# some parts of this code were borrowed from https://github.com/facebookresearch/VisualVoice/tree/main
# under the licence https://github.com/facebookresearch/VisualVoice/blob/main/LICENSE


import os
import random
import torch
import torch.nn as nn
import torch.utils.data
from scipy.io.wavfile import read
from glob import glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
from .video_reader import VideoReader
from .lipreading_utils import *
import cv2
import torchaudio
import torchvision.transforms as transforms
from .stft import normalise_mel
import pandas as pd

from .dataset_config import normalize_dataset_name

def files_to_list(data_path, suffix):
    """
    Load all .wav files in data_path
    """
    files = glob(os.path.join(data_path, f'**/*.{suffix}'), recursive=True)
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class WYSDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, split, name, root, videos_dir, mouthrois_dir, audios_dir,
                 text_dir, sampling_rate, videos_window_size, audio_stft_hop,
                 dataset_root, pred_text_main_dir=None, pred_text_pretrain_dir=None):
        self.dataset_name = normalize_dataset_name(name)
        self.ds_name = self.dataset_name.upper()
        self.root = root
        self.dataset_root = dataset_root
        self.mouthrois_dir = mouthrois_dir
        # When set, text for training samples is read from predicted lip-reading
        # transcripts instead of the ground-truth annotation files. Everything else
        # (mouthroi/melspec sampling, windowing) is unchanged.
        self.pred_text_main_dir = pred_text_main_dir
        self.pred_text_pretrain_dir = pred_text_pretrain_dir
        self.use_pred_text = pred_text_main_dir is not None
        if self.dataset_name == "lrs3":
            split_dir = ['pretrain','trainval'] if split in ['train', 'val'] else ['test']
            self.videos_dir = videos_dir
            self.audios_dir = audios_dir
            self.text_dir   = text_dir

            self.mouthroi_files = []
            for s in split_dir:
                _mouthrois_dir = os.path.join(mouthrois_dir, s)
                self.mouthroi_files += files_to_list(_mouthrois_dir, 'npz')
                if s == 'test':
                    self.test_mouthroi_files = sorted(self.mouthroi_files)
            self.mouthroi_files = sorted(self.mouthroi_files)



        elif self.dataset_name == "lrs2":
            split_dir = ['main']

            if split == 'train':
                split_dir.append('pretrain')
            self.mouthroi_files = []
            self.test_mouthroi_files = []
            test_video_lst = []

            for s in split_dir:
                if s == "main":
                    test_dir = os.path.join(videos_dir,"test")
                    main_dir = os.path.join(videos_dir,"main")
                    test_vid = set(sorted(os.listdir(test_dir)))
                    train_vid = list(set(sorted(os.listdir(main_dir))) - test_vid)

                    for vid in train_vid:
                        for file in os.listdir(os.path.join(main_dir,vid)):
                            mouthroi_file = os.path.join(mouthrois_dir, s, vid,file[:-4] + ".npz")
                            if os.path.isfile(mouthroi_file):
                                self.mouthroi_files.append(mouthroi_file) #두개주석풀기

                    for vid in list(test_vid):
                        for file in os.listdir(os.path.join(main_dir,vid)):
                            test_mouthroi_file = os.path.join(mouthrois_dir,"test",vid,file[:-4] + ".npz")
                            if os.path.isfile(test_mouthroi_file):
                                self.test_mouthroi_files.append(test_mouthroi_file)
                    self.test_mouthroi_files = set(self.test_mouthroi_files)

                    # if split == "pretrain":
                    #     videos_list_file = os.path.join(videos_dir, "train.txt")
                    # else:
                    #     videos_list_file = os.path.join(videos_dir, split+".txt")
                    # with open(videos_list_file, "r") as f:
                    #     _video_ids = f.readlines()
                    # for _vid in _video_ids:
                    #     mouthroi_file = os.path.join(mouthrois_dir, s, _vid.strip("\n")+".npz")
                    #     mouthroi_file = mouthroi_file.replace(" NF", "").replace(" MV", "")
                    #     if os.path.isfile(mouthroi_file):
                    #         self.mouthroi_files.append(mouthroi_file)
                elif s == "pretrain":
                    self.mouthroi_files += glob(os.path.join(mouthrois_dir, "pretrain", "**/*.npz"), recursive=True)


            self.videos_dir = videos_dir
            self.audios_dir = audios_dir
            self.text_dir   = text_dir
            self.mouthroi_files = sorted(self.mouthroi_files)
            self.test_mouthroi_files = sorted(self.test_mouthroi_files)
        self.test = True if split=='test' else False
        self.videos_window_size = videos_window_size
        self.audio_stft_hop = audio_stft_hop
        random.seed(1234)
        random.shuffle(self.mouthroi_files)
        self.sampling_rate = sampling_rate

        self.mouthroi_transform = self.get_mouthroi_transform()[split]
        self.face_image_transform = self.get_face_image_transform()

    def __getitem__(self, index):
        while True:
            # Get paths
            if self.test:
                mouthroi_filename = self.test_mouthroi_files[index]
            else:
                mouthroi_filename = self.mouthroi_files[index]
            pfilename = Path(mouthroi_filename)
            if self.ds_name in ["LRS3", "LRS2"]:
                video_id = '/'.join([pfilename.parts[-2], pfilename.stem])
                video_filename   = mouthroi_filename.replace(self.mouthrois_dir, self.videos_dir).replace('.npz','.mp4')
                melspec_filename = mouthroi_filename.replace(self.mouthrois_dir, self.audios_dir).replace('.npz','.wav.spec')
                text_filename    = video_filename.replace(self.videos_dir, self.text_dir).replace('.mp4', '.txt')
                if self.use_pred_text:
                    split_name = pfilename.parts[-3]
                    pred_root = self.pred_text_main_dir if split_name == "main" else self.pred_text_pretrain_dir
                    text_filename = os.path.join(pred_root, video_id + ".txt")
            # Get mouthroi
            mouthroi = np.load(mouthroi_filename)['data']
            if mouthroi.shape[0] >= self.videos_window_size or self.test:
                break
            else:
                index = random.randrange(len(self.mouthroi_files))
        melspec = torch.load(melspec_filename)


        face_image = self.load_frame(video_filename)

        video = cv2.VideoCapture(video_filename)
        info = {'audio_fps': self.sampling_rate, 'video_fps': video.get(cv2.CAP_PROP_FPS)}

        if self.test:
            audio, fs = torchaudio.load(melspec_filename.replace('.spec', ''))
            text_filename = video_filename.replace(".mp4", ".txt")
            # text_filename = text_filename.replace("videos/test","LRS2_test_lipreading")
            with open(text_filename, "r") as f:
                text = f.readlines()[0][7:-1]
            #text = self.preprocess_text(text_filename, test = self.test)

            # Normalisations & transforms
            audio = audio / 1.1 / audio.abs().max()
            face_image = self.face_image_transform(face_image)
            mouthroi = torch.FloatTensor(self.mouthroi_transform(mouthroi)).unsqueeze(0)
            melspec = normalise_mel(melspec)
            return (melspec, audio, mouthroi, face_image, text, video_id)
        else:

            # Get corresponding crops
            mouthroi, melspec ,start_frame= self.extract_window(mouthroi, melspec, info)
            if mouthroi.shape[0] < self.videos_window_size:
                return self.__getitem__(random.randrange(len(self)))

            # Augmentations
            face_image = self.augment_image(face_image)

            # Noramlisations & Transforms
            face_image = self.face_image_transform(face_image)
            mouthroi   = torch.FloatTensor(self.mouthroi_transform(mouthroi)).unsqueeze(0)   # add channel dim
            melspec    = normalise_mel(melspec)
            if self.use_pred_text:
                text = self.preprocess_text_pred(text_filename)
            else:
                text = self.preprocess_text(text_filename,start_frame)
            return (melspec, mouthroi, face_image, text)

    def __len__(self):
        return len(self.test_mouthroi_files) if self.test else len(self.mouthroi_files)

    def extract_window(self, mouthroi, mel, info ):
        hop = self.audio_stft_hop

        # vid : T,C,H,W
        vid_2_aud = info['audio_fps'] / info['video_fps'] / hop

        st_fr = random.randint(0, mouthroi.shape[0] - self.videos_window_size)
        mouthroi = mouthroi[st_fr:st_fr + self.videos_window_size]

        st_mel_fr = int(st_fr * vid_2_aud)
        mel_window_size = int(self.videos_window_size * vid_2_aud)

        mel = mel[:, st_mel_fr:st_mel_fr + mel_window_size]

        return mouthroi, mel ,st_fr

    @staticmethod
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

    @staticmethod
    def augment_image(image):
        if(random.random() < 0.5):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        return image

    @staticmethod
    def get_mouthroi_transform():
        # -- preprocess for the video stream
        preprocessing = {}
        # -- LRW config
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])
        preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])
        preprocessing['test'] = preprocessing['val']
        return preprocessing

    @staticmethod
    def get_face_image_transform():
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.Resize(224), transforms.ToTensor(), normalize]
        vision_transform = transforms.Compose(vision_transform_list)
        return vision_transform


    @staticmethod
    def preprocess_text(txt_filename ,start_frame = 0, test = False):


        if txt_filename.split('/')[-3] == "pretrain":
            #print(f"Processing file: {txt_filename}")  # 파일명 출력

            try:
                df = pd.read_csv(txt_filename, sep='\s+', header=1, names=["WORD", "START", "END", "ASDSCORE"], skiprows=3)
                ##수정정
                start_time = start_frame * 0.04

                # start_time = max(0, start_time -0.5)
                end_time   = start_time + 1

                word = []

                filtered_words = df[(df['END'].astype(float) >= start_time) & (df['START'].astype(float) <= end_time)].astype(str)

                # print(f"Filtered words from {txt_filename}:")
                # print(filtered_words)  # 필터링된 데이터 출력

                # 에러가 발생하는 부분
                txt = filtered_words['WORD'].str.cat(sep=' ')

                txt = txt.replace("{LG}", "").replace("{NS}", "").replace("\n", "").replace("  ", " ")
                txt = txt.lower().strip()

                if txt == "":
                    txt = " "

                return txt

            except Exception as e:
                print(f" Error in file: {txt_filename}")
                print(f" Exception: {e}")
                print(f" Data Type of 'WORD' Column: {filtered_words['WORD'].dtype if 'WORD' in filtered_words else 'Column Missing'}")
                print(f" Head of 'WORD' Column:\n{filtered_words['WORD'].head() if 'WORD' in filtered_words else 'Column Missing'}")
                raise  # 에러 다시 발생시켜서 전체 실행 중지
        else:
            with open(txt_filename, "r") as f:
                if test:
                    txt = f.readline()[:]
                else:
                    txt = f.readline()[7:]  # discard 'Text:  ' prefix #Original [7:] ==gunwoo modifiy 20240926
            txt = txt.replace("{LG}", "")  # remove laughter
            txt = txt.replace("{NS}", "")  # remove noise
            txt = txt.replace("\n", "")
            txt = txt.replace("  ", " ")
            txt = txt.lower().strip()
            return txt

    @staticmethod
    def preprocess_text_pred(txt_filename):
        # Predicted lip-reading transcripts are plain, unprefixed, already lowercased
        # single-line text covering the whole utterance/clip (no per-window alignment).
        with open(txt_filename, "r") as f:
            txt = f.readline()
        txt = txt.replace("{LG}", "")  # remove laughter
        txt = txt.replace("{NS}", "")  # remove noise
        txt = txt.replace("\n", "")
        txt = txt.replace("  ", " ")
        txt = txt.lower().strip()
        if txt == "":
            txt = " "
        return txt

    # @staticmethod
    # def preprocess_text(txt_filename ,start_frame = 0, test = False):


    #     if txt_filename.split('/')[-3] == "pretrain":
    #         #print(f"Processing file: {txt_filename}")  # 파일명 출력

    #         try:
    #             df = pd.read_csv(txt_filename, sep='\s+', header=1, names=["WORD", "START", "END", "ASDSCORE"], skiprows=3)
    #             start_time = start_frame * 0.04
    #             word = []

    #             filtered_words = df[(df['END'].astype(float) >= start_time) & (df['START'].astype(float) <= start_time + 1.0)].astype(str)


    #             if len(df['WORD']) <= 25:
    #                 with open(txt_filename, 'r') as f:
    #                     txt = f.readline()[7:]
    #                     txt = txt.replace("{LG}", "").replace("{NS}", "").replace("\n", "").replace("  ", " ")
    #                     txt = txt.lower().strip()
    #                     return txt
    #             else:
    #                 #start_point = len(df['WORD']) - 25
    #                 true_indices = np.where((df['END'].astype(float) >= start_time) & (df['START'].astype(float) <= start_time + 1.0) == True)[0]
    #                 min_idx , max_idx = min(true_indices) , max(true_indices)
    #                 start_index = max(0, min_idx - (25 - (max_idx - min_idx + 1)) // 2)
    #                 end_index = min(len(df['WORD']), start_index + 25)
    #                 start_index = max(0, end_index - 25)
    #                 selected_segment = df[start_index:end_index]
    #             # print(f"Filtered words from {txt_filename}:")
    #             # print(filtered_words)  # 필터링된 데이터 출력

    #             # 에러가 발생하는 부분
    #             txt = filtered_words['WORD'].str.cat(sep=' ')

    #             txt = txt.replace("{LG}", "").replace("{NS}", "").replace("\n", "").replace("  ", " ")
    #             txt = txt.lower().strip()

    #             if txt == "":
    #                 txt = " "

    #             return txt

    #         except Exception as e:
    #             print(f" Error in file: {txt_filename}")
    #             print(f" Exception: {e}")
    #             print(f" Data Type of 'WORD' Column: {filtered_words['WORD'].dtype if 'WORD' in filtered_words else 'Column Missing'}")
    #             print(f" Head of 'WORD' Column:\n{filtered_words['WORD'].head() if 'WORD' in filtered_words else 'Column Missing'}")
    #             raise  # 에러 다시 발생시켜서 전체 실행 중지
    #     else:
    #         with open(txt_filename, "r") as f:
    #             if test:
    #                 txt = f.readline()[:]
    #             else:
    #                 txt = f.readline()[7:]  # discard 'Text:  ' prefix #Original [7:] ==gunwoo modifiy 20240926
    #         txt = txt.replace("{LG}", "")  # remove laughter
    #         txt = txt.replace("{NS}", "")  # remove noise
    #         txt = txt.replace("\n", "")
    #         txt = txt.replace("  ", " ")
    #         txt = txt.lower().strip()
    #         return txt
    @staticmethod
    def preprocess_train_text(txt_filename, start_frame):
        df = pd.read_csv(txt_filename, sep='\s+', header=1, names=["WORD", "START", "END", "ASDSCORE"], skiprows=3)
        start_time = start_frame * 0.04
        word = []

        filtered_words = df[(df['START'].astype(float) >= start_time) & (df['END'].astype(float) <= start_time + 1.0)]
        # txt = ' '.join(str(filtered_words['WORD']))
        txt = filtered_words['WORD'].str.cat(sep=' ')

        txt = txt.replace("{LG}", "")  # remove laughter
        txt = txt.replace("{NS}", "")  # remove noise
        txt = txt.replace("\n", "")
        txt = txt.replace("  ", " ")
        txt = txt.lower().strip()
        if txt == "":
            txt = " "

        return txt
