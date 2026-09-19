#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import torchvision
from transformers import BertTokenizer
import torch.nn.functional as F
from torch import nn


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    #tokenizer.silence_token_id = tokenizer.additional_special_tokens_ids[1]
    return tokenizer

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return "AudioVisualModel"


    def __init__(self, nets):
        super(AudioVisualModel, self).__init__()
        self.num_timesteps = 400
        # initialize model
        self.net_lipreading, self.net_facial, self.net_diffwave, self.net_text,self.net_attention,self.net_fusion = nets
        self.tokenizer = init_tokenizer()
        self.text_proj = nn.Linear(768, 512) #original
        self.fusion_proj = nn.Linear(768,512)
        # classifier guidance null conditioners
        torch.manual_seed(0)        # so we have the same null tokens on all nodes
        self.register_buffer("mouthroi_null", torch.randn(1, 1, 1, 88, 88))  # lips regions frames are 88x88 each
        self.register_buffer("face_null", torch.randn(1, 3, 224, 224))  # face image size is 224x224

        # self.freeze_net_text()
    def forward(self, melspec, mouthroi, face_image, text, diffusion_steps, cond_drop_prob):
        # classifier guidance
        if diffusion_steps is None:
            t = torch.rand((melspec.shape[0],), device=melspec.device)

        noise = torch.randn_like(melspec)
        x0 = melspec
        xt = self.add_noise(x0, noise, t)
        batch = melspec.shape[0]

        if cond_drop_prob == 0 :
            _mouthroi = mouthroi
            _face_image = face_image
            _text = text
        elif cond_drop_prob == 1:
            prob_keep_mask = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - (cond_drop_prob), melspec.device)
            _face_image = torch.where(prob_keep_mask.squeeze(1), face_image, self.face_null)
            _mouthroi , _text = self.mouth_text_mask(mouthroi, text, prob = [0, 0, 0, 1], device = melspec.device)
            # _mouthroi = torch.where(prob_keep_mask, mouthroi, self.mouthroi_null)
            # _text = self.prob_mask_text(text, cond_drop_prob)
        else:
            prob_keep_mask = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - (cond_drop_prob*2), melspec.device)
            _face_image = torch.where(prob_keep_mask.squeeze(1), face_image, self.face_null)
            _mouthroi , _text = self.mouth_text_mask(mouthroi, text, prob = [1-(cond_drop_prob*3), cond_drop_prob, cond_drop_prob,  cond_drop_prob], device = melspec.device)

        # pass through visual stream and extract lipreading features
        lipreading_feature = self.net_lipreading(_mouthroi)

        # pass through visual stream and extract identity features
        identity_feature   = self.net_facial(_face_image)

        max_length = lipreading_feature.shape[-1]
        #text = self.preprocess_text(text, self.tokenizer)
        text = self.tokenizer(_text, padding='max_length', truncation=True, max_length=max_length,
                              return_tensors="pt").to(melspec.device)
        text_output = self.net_text(text.input_ids, attention_mask = text.attention_mask,  return_dict = True)
        st_point = random.randint(0,5)
        text_feat = self.text_proj(text_output.last_hidden_state).permute(0,2,1)   # [128,256]
        # text_feat = text_feat[:,:,None]
        # text_feat = text_feat.repeat(1,1, lipreading_feature.shape[-1]) # [128, 256, 25]

        video_att, text_att, vt_att , tv_att= self.net_attention(lipreading_feature, text_feat, text_mask = None)
        # fusion_feature = self.net_fusion(video_att, text_att, vt_att).squeeze().permute(0,2,1)
        # fusion_feature = self.fusion_proj(fusion_feature).permute(0,2,1).unsqueeze(2)

        video_att = video_att.permute(0,2,1).unsqueeze(2)
        text_att  =  text_att.permute(0,2,1).unsqueeze(2)
        vt_att    =    vt_att.permute(0,2,1).unsqueeze(2)
        tv_att    =    tv_att.permute(0,2,1).unsqueeze(2)
        # what type of visual feature to use
        identity_feature = identity_feature.repeat(1, 1, 1, lipreading_feature.shape[-1])
        visual_feature   = torch.cat((identity_feature,video_att,text_att, vt_att ,tv_att), dim=1)
        visual_feature   = visual_feature.squeeze(2)  # so dimensions are B, C, num_frames

        #layer norm == 3줄 주석 생각
        # Batch, Channel, Frame = visual_feature.size()
        # layer_norm = nn.LayerNorm(Channel).cuda()
        # visual_feature = layer_norm(visual_feature.transpose(1,2)).transpose(1,2)

        velocity_pred = self.net_diffwave((xt, t.unsqueeze(1)), cond=visual_feature)
        target = noise - x0
        return velocity_pred , target
    def preprocess_text(self, text, tokenizer):
        if isinstance(text, tuple):  # tuple을 list로 변환
            text = list(text)

        if isinstance(text, list):  # 배치 입력 지원
            return [t if isinstance(t, str) and t.strip() else "[SILENCE]" for t in text]

        return text if isinstance(text, str) and text.strip() else "[SILENCE]"
    def add_noise(self,
                  original_samples: torch.FloatTensor,
                  noise : torch.FloatTensor,
                  timesteps: torch.FloatTensor,
                  ) -> torch.FloatTensor:

        # timepoints = timesteps.float() / self.num_timesteps  # t in [0,1]
        timepoints = timesteps.unsqueeze(1).unsqueeze(1)   # broadcasting
        return (1 - timepoints) * original_samples + timepoints * noise

    def mouth_text_mask(self, mouthroi, texts, prob = [0.7, 0.1, 0.1, 0.1] ,device = 'cuda'):
        def case_ls(batch_size, prob = [0.7, 0.1, 0.1, 0.1]):
            n0 = int(batch_size * prob[0])
            n1 = int(batch_size * prob[1])
            n2 = int(batch_size * prob[2])
            n3 = batch_size - n0 - n1 - n2
            case_list = [0] * n0 + [1] * n1 + [2] * n2 + [3] * n3
            random.shuffle(case_list)
            return case_list
        if prob[0] == 1:
            _mouthroi = torch.zeros(mouthroi, device=device, dtype=torch.bool)
            if isinstance(texts, tuple) or isinstance(texts, list):
                _text = tuple([""] * len(texts))
            else:
                _text = tuple([""])
        # elif prob[0] == 0:
        #     _mouthroi = mouthroi
        #     _text = tuple(texts)
        else:
            case_list = torch.tensor(case_ls(mouthroi.shape[0], prob = prob)).to(device)
            keep_video = (case_list == 0) | (case_list == 2)   # case 0,2: video keep
            keep_text  = (case_list == 0) | (case_list == 1)

            _mouthroi = torch.where(keep_video.view(mouthroi.shape[0],1,1,1,1), mouthroi, self.mouthroi_null)
            _text = tuple((txt if kv.item() else "") for txt, kv in zip(texts, keep_text))
        return _mouthroi, _text
    @staticmethod
    def prob_mask_like(shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        else:
            return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
    def freeze_net_text(self):
        # for param in self.net_text.parameters():
        #     param.requires_grad = False
        for name, param in  self.net_text.named_parameters():
            print(name, param.requires_grad)


    def sample(self, z, noise, mouthroi, face_image, text, diffusion_steps, cond_drop_prob):



        batch = z.shape[0]
        if cond_drop_prob == 0 :
            _mouthroi = mouthroi
            _face_image = face_image
            _text = text
        elif cond_drop_prob == 1:
            prob_keep_mask = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - (cond_drop_prob), mouthroi.device)
            _face_image = torch.where(prob_keep_mask.squeeze(1), face_image, self.face_null)
            _mouthroi , _text = self.mouth_text_mask(mouthroi, text, prob = [0, 0, 0, 1], device = mouthroi.device)
            # _mouthroi = torch.where(prob_keep_mask, mouthroi, self.mouthroi_null)
            # _text = self.prob_mask_text(text, cond_drop_prob)
        else:
            prob_keep_mask = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - (cond_drop_prob*2), mouthroi.device)
            _face_image = torch.where(prob_keep_mask.squeeze(1), face_image, self.face_null)
            _mouthroi , _text = self.mouth_text_mask(mouthroi, text, prob = [1-(cond_drop_prob*3), cond_drop_prob, cond_drop_prob,  cond_drop_prob], device = mouthroi.device)

        # pass through visual stream and extract lipreading features
        lipreading_feature = self.net_lipreading(_mouthroi)
        _face_image = torch.zeros_like(_face_image).to(mouthroi.device)
        # pass through visual stream and extract identity features
        identity_feature   = self.net_facial(_face_image)

        #text Encoder
        max_length = lipreading_feature.shape[-1]
        text = self.tokenizer(_text, padding='max_length', truncation=True, max_length=max_length,
                              return_tensors="pt").to(z.device)
        text_output = self.net_text(text.input_ids, attention_mask = text.attention_mask,  return_dict = True)
        st_point = random.randint(0,5)
        text_feat = self.text_proj(text_output.last_hidden_state).permute(0,2,1)   # [128,256]


        #Fusion Module
        video_att, text_att, vt_att , tv_att= self.net_attention(lipreading_feature, text_feat, text_mask = None)


        video_att = video_att.permute(0,2,1).unsqueeze(2)
        text_att  =  text_att.permute(0,2,1).unsqueeze(2)
        vt_att    =    vt_att.permute(0,2,1).unsqueeze(2)
        tv_att    =    tv_att.permute(0,2,1).unsqueeze(2)
        # what type of visual feature to use
        identity_feature = identity_feature.repeat(1, 1, 1, lipreading_feature.shape[-1])
        visual_feature   = torch.cat((identity_feature,video_att,text_att, vt_att ,tv_att), dim=1)
        visual_feature   = visual_feature.squeeze(2)  # so dimensions are B, C, num_frames


        velocity_pred = self.net_diffwave((z, diffusion_steps), cond=visual_feature)
        return velocity_pred