import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import random

from omegaconf import DictConfig, OmegaConf
import hydra
class InputEmbeddings(nn.Module):
    def __init__(self,d_model : int, vocab_size : int):
        super(InputEmbeddings,self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):

        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len:int, dropout:float = 0.1)->None:
        super(PositionalEncoding,self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)


        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len, dtype = torch.float).unsqueeze(1)
        _2i = torch.exp(torch.arange(0,d_model,2,dtype=torch.float))
        pe[:,0::2] = torch.sin(position/10000**(_2i/d_model))
        pe[:,1::2] = torch.cos(position/10000**(_2i/d_model))
        pe = pe.unsqueeze(0) #(1,Seq_len,d_model)

        self.register_buffer('pe',pe)
    def forward(self,x):
        x = x.squeeze()
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6) -> None:
        super(LayerNormalization,self).__init__()
        self.eps   = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias  = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim =-1, keepdim = True)
        std  = x.std( dim =-1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int ,d_ff : int, dropout : float) -> None:
        super(FeedForwardBlock,self).__init__()
        self.linear_1 = nn.Linear(d_model , d_ff)
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    #h : head 개수
    def __init__(self,d_model : int, h: int, dropout: float) -> None:
        super(MultiHeadAttentionBlock,self).__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h ==0, 'd_model is not divisible by h'

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model,d_model) #Wk
        self.w_v = nn.Linear(d_model,d_model) #Wv

        #concat 하는 부분에서의 wo값
        self.w_o = nn.Linear(d_model,d_model) #Wo
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):
        d_k = query.shape[-1] #(Batch, seq_len, d_model)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = attention_scores.softmax(dim=-1) #(Batch,h,seq_len, seq_len)
        if dropout is not None :
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask = None):
        # Q,K,V를  d_k, d_k, d_v 차원으로 projection
        query = self.w_q(q) #(Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        #Q,K,V를 head 수 만큼 분리해주기
        #(Batch, seq_len, d_model) -> (Batch, Seq_len, h, d_k) -> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key   = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        #(Batch, h, Seq_len, d_k) -> (Batch, Seq_len, h,  d_k) -> (Batch,Seq_len, d_k)
        #contiguous(인접한) : Tensor의 각 값들이 메모리에도 순차적으로 저장되어 있는지 여부를 의미
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h*self.d_k) # -1은 나머지 차원을 자동으로 조정하라는 의미

        #(Batch,Seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout : float) -> None:
        super(ResidualConnection,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    #sublayer ?
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float) -> None:
        super(EncoderBlock,self).__init__()
        self.self_atttention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,y,src_mask):
        #x = query, y = key,value
        x = self.residual_connections[0](x, lambda x: self.self_atttention_block(x,y,y,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers : nn.ModuleList, mode: str = "self") -> None :
        super(Encoder,self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        self.mode = mode
    def forward(self,x,y,mask):
        for layer in self.layers:
            x = layer(x,y,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self,self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock,dropout:float) -> None:
        super(DecoderBlock,self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        #encoder output 값을 받는다(cross_attention)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self,layers: nn.ModuleList) -> None:
        super(Decoder,self).__init__()
        self.layers = layers
        self.norm  = LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

#ProjectionLayer
class ProjectionLayer(nn.Module):

    def __init__(self,d_model : int, vocab_size : int) -> None:
        super(ProjectionLayer,self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self,x):
        #(Batch, seq_len,d_model) -> (Batch,seq_len,vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(self,encoder :Encoder, decoder : Decoder, src_embed : InputEmbeddings, tgt_embed : InputEmbeddings, src_pos : PositionalEncoding, tgt_pos :PositionalEncoding, projection_layer : ProjectionLayer) -> None:
        super(Transformer,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


    def encode(self, src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self,x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size : int, tgt_vocab_size : int, src_seq_len : int, tgt_seq_len : int, d_model : int=512, N:int = 6, h : int = 8, dropout : float=0.1, d_ff : int=2048 ) -> Transformer:
    #Create Embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_vocab_size,dropout)

    #Create encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    #Create decoder blocks
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block  = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    #Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #Create the transformer
    transformer = Transformer(encoder, decoder, src_embed,tgt_embed, src_pos, tgt_pos, projection_layer)

    #initial parameters
    for p in transformer.parameters():
        if p.dim() >1 :
            nn.init.xavier_uniform_(p)

    return transformer

class MultiAttentionEncoder(nn.Module):
    def __init__(self, video_length : int =25, text_length : int = 25 , d_model : int = 512, N : int = 6, h : int = 8, dropout : float = 0.1, d_ff : int = 2048):
        super(MultiAttentionEncoder,self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # self.video_linear = nn.Linear(d_model, d_model)
        # self.text_linear  = nn.Linear(d_model, d_model)
        self.video_pos = PositionalEncoding(d_model, video_length,dropout)
        self.text_pos  = PositionalEncoding(d_model, text_length ,dropout)

        video_attention_block = []
        for _ in range(N):
            self_attention_block = MultiHeadAttentionBlock(d_model , h, dropout)
            feed_forward_block   = FeedForwardBlock(d_model , d_ff ,dropout)
            encoder_block_video  = EncoderBlock(self_attention_block , feed_forward_block ,dropout)
            video_attention_block.append(encoder_block_video)

        vt_attention_block = []
        for _ in range(N):
            cross_attention_block = MultiHeadAttentionBlock(d_model , h, dropout)
            feed_forward_block    = FeedForwardBlock(d_model , d_ff ,dropout)
            encoder_block_vt      = EncoderBlock(cross_attention_block , feed_forward_block ,dropout)
            vt_attention_block.append(encoder_block_vt)

        text_attention_block = []
        for _ in range(N):
            self_attention_block = MultiHeadAttentionBlock(d_model , h, dropout)
            feed_forward_block   = FeedForwardBlock(d_model , d_ff ,dropout)
            encoder_block_text   = EncoderBlock(self_attention_block , feed_forward_block ,dropout)
            text_attention_block.append(encoder_block_text)

        tv_attention_block = []
        for _ in range(N):
            cross_attention_block = MultiHeadAttentionBlock(d_model , h, dropout)
            feed_forward_block    = FeedForwardBlock(d_model , d_ff ,dropout)
            encoder_block_tv      = EncoderBlock(cross_attention_block , feed_forward_block ,dropout)
            tv_attention_block.append(encoder_block_tv)

        self.video_encoder = Encoder(nn.ModuleList(video_attention_block))
        self.text_encoder  = Encoder(nn.ModuleList(text_attention_block))
        self.vt_encoder    = Encoder(nn.ModuleList(vt_attention_block))
        self.tv_encoder    = Encoder(nn.ModuleList(tv_attention_block))

    def positional_encoding(self,x: torch.Tensor, d_model: int, dropout: float = 0.1) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서, shape은 [batch, seq_len, d_model]
            d_model (int): 모델의 차원
            dropout (float): dropout 비율
        """
        seq_len = x.size(1)

        # positional encoding을 계산
        pe = torch.zeros(seq_len, d_model, device=x.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape 맞추기: [1, seq_len, d_model] 후 더하기
        pe = pe.unsqueeze(0)
        x = x + pe

        return x
    def forward(self, video_emb, text_emb , text_mask = None):
        video_emb = self.dropout(self.positional_encoding(video_emb.squeeze(2).permute(0,2,1),self.d_model))

        # video_emb = video_emb.squeeze(2).permute(0,2,1)
        #video_emb = self.video_pos(video_emb)

        text_emb  = self.dropout(self.positional_encoding(text_emb.permute(0,2,1),d_model = self.d_model))
        # text_emb  = text_emb.permute(0,2,1)
        # text_emb  = self.text_pos(text_emb)

        #video self attention
        video_att = self.video_encoder(video_emb, video_emb ,mask = None) # x = [batch, seq_len, d_model]

        #text self attention
        text_att  = self.text_encoder(text_emb, text_emb, mask = None) # x = [batch, seq_len, d_model]

        #video text cross attention
        vt_att    = self.vt_encoder(video_emb, text_emb, mask = None) #query = video, key&value = text

        tv_att    = self.tv_encoder(text_emb, video_emb, mask= None) #query = text , key&value = video
        return video_att, text_att, vt_att , tv_att

class FusionModule(nn.Module):
    def __init__(self, linear1:int = 100, linear2 : int = 50, linear3:int = 100, dropout: float = 0.1):
        super(FusionModule,self).__init__()

        self.block = nn.Sequential(
            nn.Linear(linear1, linear2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear2,linear3),
            nn.Softmax(dim=-1)
        )

    def forward(self, emb):
        return self.block(emb)



class MultiConditionFusion(nn.Module):
    def __init__(self, input_ch :int = 256):
        super(MultiConditionFusion,self).__init__()

        self.emb_v  = FusionModule(linear1= input_ch, linear2 = 128, linear3= input_ch)#linear 1,2,3채우기
        self.emb_t  = FusionModule(linear1= input_ch, linear2 = 128, linear3= input_ch)
        self.emb_vt = FusionModule(linear1= input_ch, linear2 = 128, linear3= input_ch)

    def forward(self, video, text, video_text):
        w_video = self.emb_v(video)
        w_text  = self.emb_t(text)
        w_vt    = self.emb_vt(video_text)


        w_video = w_video * video
        w_text  = w_text  * text
        w_vt    = w_vt    * video_text

        combined = torch.cat((w_video,w_text,w_vt), dim = -1) # dim은 다시보기
        combined = combined.permute(0,2,1).unsqueeze(2)
        return combined

@hydra.main(version_base=None, config_path="/home/gunwoo/gunwoo/LipVoicer_attention_test/configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    model = MultiAttentionEncoder(**cfg.attention)
    fusionmodel = MultiConditionFusion(input_ch = 256)
    #model = MouthLandmark_Model(mouth_cfg= cfg.mouthlandmark)
    video_emb = torch.Tensor(np.random.rand(128, 256, 1, 25))
    text_emb  = torch.Tensor(np.random.rand(128, 256, 25))
    video_att, text_att, vt_att = model(video_emb, text_emb)
    y = fusionmodel(video_att , text_att, vt_att)
    print(y.shape)
    print(y)

if __name__ == "__main__":
    main()