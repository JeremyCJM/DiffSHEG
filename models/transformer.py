"""
Copyright 2023 Junming Chen. Built on DiffMotion
"""

from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
# import clip

import math
import time
import os
import pickle
from torch.nn.utils import weight_norm

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1 # 1 if max_seq_len % period == 0 else 0
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x, dropout=False):

        x = x + self.pe[:, :x.size(1), :]
        
        if dropout:
            x = self.dropout(x)
        return x



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class LinearTemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, aud_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(aud_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(aud_latent_dim, latent_dim)
        self.value = nn.Linear(aud_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb, mask=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y

class LinearTemporalDiffusionTransformerLayer_2(nn.Module):

    def __init__(self,
                 opt,
                 seq_len=60,
                 latent_dim=32,
                 aud_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.opt = opt
        if opt.model_base == "transformer_encoder":
            pre_proj_dim = latent_dim
            if opt.expCondition_gesture_only != None:
                pre_proj_dim = latent_dim + aud_latent_dim + opt.expression_dim
            elif self.opt.gesCondition_expression_only:
                pre_proj_dim = latent_dim + aud_latent_dim + opt.dim_pose
            else:
                pre_proj_dim = latent_dim + aud_latent_dim
            if self.opt.addTextCond:
                pre_proj_dim += self.opt.word_f
            if self.opt.addEmoCond:
                pre_proj_dim += self.opt.emotion_f
            if (self.opt.expAddHubert and self.opt.expCondition_gesture_only is None) or self.opt.addHubert or self.opt.addWav2Vec2:
                if self.opt.encode_hubert:
                    pre_proj_dim += 128
                elif self.opt.encode_wav2vec2:
                    pre_proj_dim += 256
                else:
                    pre_proj_dim += 1024
            
            self.feat_proj = nn.Linear(pre_proj_dim, latent_dim)
                


        self.sa_block = LinearTemporalSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)

        if not self.opt.unidiffuser:
            self.ca_block = LinearTemporalCrossAttention(
                seq_len, latent_dim, aud_latent_dim, num_head, dropout, time_embed_dim)

        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask, add_cond={}):
        if self.opt.model_base == 'transformer_encoder':
            if add_cond in [{}, None] and xf != None:
                x = self.feat_proj(torch.cat((x, xf), -1))
            elif add_cond not in [{}, None] and xf != None:
                try:
                    x = self.feat_proj(torch.cat((x, xf, add_cond), -1))
                except:
                    import pdb; pdb.set_trace()

        x = self.sa_block(x, emb, src_mask) 

        if self.opt.model_base == 'transformer_decoder':
            x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)

        return x

class LinearTemporalDiffusionTransformerLayer(nn.Module):
    def __init__(self,
                 opt,
                 seq_len=60,
                 latent_dim=32,
                 aud_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1,
                 cond_proj=True):
        super().__init__()
        self.opt = opt
        if opt.model_base == "transformer_encoder":
            pre_proj_dim = latent_dim
            if self.opt.cond_projection in ["linear_excludeX", "mlp_excludeX"]:
                pre_proj_dim = 0

            if opt.expCondition_gesture_only != None:
                pre_proj_dim = pre_proj_dim + aud_latent_dim + opt.expression_dim
            elif self.opt.gesCondition_expression_only:
                pre_proj_dim = pre_proj_dim + aud_latent_dim + opt.dim_pose
            else:
                pre_proj_dim = pre_proj_dim + aud_latent_dim

            if (self.opt.expAddHubert and self.opt.expCondition_gesture_only is None) or self.opt.addHubert or self.opt.addWav2Vec2:
                if self.opt.encode_hubert:
                    pre_proj_dim += 128
                elif self.opt.encode_wav2vec2:
                    pre_proj_dim += 256
                else:
                    pre_proj_dim += 1024            

            if cond_proj:
                if "linear" in self.opt.cond_projection:
                    self.feat_proj = nn.Linear(pre_proj_dim, latent_dim)
                elif "mlp" in self.opt.cond_projection:
                    self.feat_proj = nn.Sequential(
                        nn.LayerNorm(pre_proj_dim),
                        nn.Linear(pre_proj_dim, latent_dim*2),
                        nn.SiLU(),
                        nn.Linear(latent_dim*2, latent_dim),
                    )

        self.sa_block = LinearTemporalSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)

        if not self.opt.unidiffuser and self.opt.model_base == 'transformer_decoder':
            self.ca_block = LinearTemporalCrossAttention(
                seq_len, latent_dim, aud_latent_dim, num_head, dropout, time_embed_dim)

        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask, add_cond={}, null_cond_emb=None):
        # t1 = time.time()
        if self.opt.cond_residual or self.opt.cond_projection in ["linear_excludeX", "mlp_excludeX"]:
            x_ori = x
        if self.opt.cond_projection in ["linear_includeX", "mlp_includeX"]:
            if self.opt.model_base == 'transformer_encoder':
                if add_cond in [{}, None] and xf != None:
                    x = torch.cat((x, xf), -1)
                elif add_cond not in [{}, None] and xf != None:
                    try:
                        x = torch.cat((x, xf, add_cond), -1)
                    except:
                        import pdb; pdb.set_trace()
        elif self.opt.cond_projection in ["linear_excludeX", "mlp_excludeX"]:
            if self.opt.model_base == 'transformer_encoder':
                if add_cond in [{}, None] and xf != None:
                    x = xf
                elif add_cond not in [{}, None] and xf != None:
                    try:
                        x = torch.cat((xf, add_cond), -1)
                    except:
                        import pdb; pdb.set_trace()

        else:
            raise NotImplementedError
        
        if self.opt.classifier_free and xf != None:
            if self.training:
                mask = (torch.linspace(0, 1, x.shape[0]) < self.opt.null_cond_prob).to(x.device)
                x = torch.where(mask.unsqueeze(1).unsqueeze(2), null_cond_emb.repeat(x.shape[1],1).unsqueeze(0), x)
            elif self.opt.cond_scale != 1:
                mask = (torch.linspace(0, 1, x.shape[0]) < 0.5).to(x.device)
                x = torch.where(mask.unsqueeze(1).unsqueeze(2), null_cond_emb.repeat(x.shape[1],1).unsqueeze(0), x)

        if xf != None:
            x = self.feat_proj(x)

        if self.opt.cond_residual or self.opt.cond_projection in ["linear_excludeX", "mlp_excludeX"]:
            x = x + x_ori

        x = self.sa_block(x, emb, src_mask) 

        if self.opt.model_base == 'transformer_decoder':
            x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)

        return x


class MotionTransformer(nn.Module):
    def __init__(self,
                 opt,
                 input_feats,
                 audio_dim = 128,
                 style_dim = 4,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu", 
                 num_text_layers=4,
                 aud_latent_dim=256,
                 text_ff_size=2048,
                 text_num_heads=4,
                 no_clip=False,
                 pe_type='learnable',
                 block = None,
                 **kargs):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation  
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4

        self.audio_dim = audio_dim

        self.opt = opt

        if pe_type == 'learnable':
            self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
        elif pe_type == 'ppe_sinu':
            self.PPE = PeriodicPositionalEncoding(latent_dim, period = 25)
        elif pe_type == 'pe_sinu':
            self.PE = PeriodicPositionalEncoding(latent_dim, period = 600)
        elif pe_type == 'ppe_sinu_dropout':
            self.PPE_drop = PeriodicPositionalEncoding(latent_dim, period = 25)
        elif pe_type == 'pe_sinu_repeat':
            self.PE = PeriodicPositionalEncoding(latent_dim, period = 200)
        

        # compute dimension of the condition
        self.pre_proj_dim = latent_dim
        if self.opt.cond_projection in ["linear_excludeX", "mlp_excludeX"]:
            self.pre_proj_dim = 0

        if opt.expCondition_gesture_only != None:
            self.pre_proj_dim = self.pre_proj_dim + aud_latent_dim + opt.expression_dim
        elif self.opt.gesCondition_expression_only:
            self.pre_proj_dim = self.pre_proj_dim + aud_latent_dim + opt.dim_pose
        else:
            self.pre_proj_dim = self.pre_proj_dim + aud_latent_dim
        if self.opt.addTextCond:
            self.pre_proj_dim += self.opt.word_f
        if self.opt.addEmoCond:
            self.pre_proj_dim += self.opt.emotion_f
        if (self.opt.expAddHubert and self.opt.expCondition_gesture_only is None) or self.opt.addHubert or self.opt.addWav2Vec2:
            if self.opt.encode_hubert:
                self.pre_proj_dim += 128
            elif self.opt.encode_wav2vec2:
                self.pre_proj_dim += 256
            else:
                self.pre_proj_dim += 1024
            

        # classifier-free
        self.null_cond_emb = None
        if self.opt.classifier_free:
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.pre_proj_dim))
                
        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, latent_dim)

        if self.opt.separate != None:
            self.up_proj = nn.Linear(self.opt.lower_dim, self.opt.higher_dim)
            self.down_proj = nn.Linear(self.opt.higher_dim, self.opt.lower_dim)
        
        self.audio_proj = nn.Linear(audio_dim, aud_latent_dim)

        if self.opt.encode_hubert:
            self.hubert_encoder = nn.Sequential(*[
                nn.Conv1d(1024, 128, 3, 1, 1, bias=False),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, 128, 3, 1, 1, bias=False)
            ])
        elif self.opt.encode_wav2vec2:
            self.hubert_encoder = nn.Linear(768, 256)

        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        if not (self.opt.ExprID_off and block=="expression"):
            self.pid_embed = nn.Sequential(
                nn.Linear(style_dim, self.time_embed_dim),
                nn.SiLU(),
                nn.Linear(self.time_embed_dim, self.time_embed_dim),
            )

        self.temporal_decoder_blocks = nn.ModuleList()


        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                LinearTemporalDiffusionTransformerLayer(
                    opt=opt,
                    seq_len=num_frames,
                    latent_dim=latent_dim,
                    aud_latent_dim=aud_latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout
                )
            )
        
        self.out = nn.Linear(latent_dim, self.input_feats)
        

    def generate_src_mask(self, T, length, causal=False):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        
        if causal:
            for i in range(B):
                for j in range(i+1, length[i]):
                    src_mask[i, j] = 0

        return src_mask
    

    

    def forward(self, x, timesteps, audio_emb, length, person_id, add_cond={}, pe_type="learnable", y=None, block=None):
        """
        x: B, T, D
        """

        
        if len(person_id.shape) == 1:
            person_id = person_id.unsqueeze(0)
        
        exp_cond = None
        if self.opt.expCondition_gesture_only != None:
            audio_emb, exp_cond = torch.split(audio_emb, [self.audio_dim, self.opt.expression_dim], dim=-1)
        elif self.opt.gesCondition_expression_only:
            audio_emb, exp_cond = torch.split(audio_emb, [self.audio_dim, self.opt.dim_pose], dim=-1)
        
        # encode hubert
        add_cond_new = add_cond.copy()
        if 'pretrain_aud_feat' in add_cond.keys():
            if self.opt.encode_hubert:
                add_cond_new["pretrain_aud_feat"] = self.hubert_encoder(add_cond["pretrain_aud_feat"].transpose(-1,-2)).transpose(-1,-2)
            elif self.opt.encode_wav2vec2:
                add_cond_new["pretrain_aud_feat"] = self.hubert_encoder(add_cond["pretrain_aud_feat"])
        
        if self.opt.addTextCond:
            text_feat_seq = self.text_encoder(add_cond['text'])
            assert(audio_emb.shape[1] == text_feat_seq.shape[1])
            add_cond_new['text'] = text_feat_seq
        if self.opt.addEmoCond:
            emo_feat_seq = self.emotion_embedding(add_cond['emo'])
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
            emo_feat_seq = self.emotion_embedding_tail(emo_feat_seq) 
            emo_feat_seq = emo_feat_seq.permute([0,2,1]) 
            add_cond_new['emo'] = emo_feat_seq
        
        if add_cond_new in [{}, None]:
            add_cond_new = exp_cond
        else:
            add_cond_new = torch.cat([v for v in add_cond_new.values()], dim=-1)
            if exp_cond != None:
                add_cond_new = torch.cat((add_cond_new, exp_cond), dim=-1)

        if self.opt.classifier_free and not self.training and self.opt.cond_scale != 1:
            x = torch.cat([x] * 2)
            timesteps = torch.cat([timesteps] * 2)
            audio_emb = torch.cat([audio_emb] * 2)
            length = torch.cat([length] * 2)
            person_id = torch.cat([person_id] * 2)
            if add_cond_new != None:
                add_cond_new = torch.cat([add_cond_new] * 2)
        
        if self.opt.classifier_free and self.opt.ExprID_off_uncond and block=="expression":
            if self.training:
                mask = (torch.linspace(0, 1, person_id.shape[0]) < self.opt.null_cond_prob).to(person_id.device)
                person_id = torch.where(mask.unsqueeze(1).unsqueeze(2), torch.zeros(person_id.shape), x)
            else:
                mask = (torch.linspace(0, 1, person_id.shape[0]) < 0.5).to(person_id.device)
                person_id = torch.where(mask.unsqueeze(1).unsqueeze(2), torch.zeros(person_id.shape), x)


        
        if self.opt.no_style or (self.opt.ExprID_off and block=="expression"):
            emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        else:
            emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + self.pid_embed(person_id)
            

        B, T = x.shape[0], x.shape[1]
        length = torch.LongTensor([T for ii in range(B)]).to(x.device)

        # B, T, latent_dim
        h = self.joint_embed(x)
        
        # add positional embedding to gesture seq
        if pe_type == 'learnable':
            h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        elif pe_type == 'ppe_sinu':
            h = self.PPE(h)
        elif pe_type == 'pe_sinu' or pe_type == 'pe_sinu_repeat':
            h = self.PE(h)
        elif pe_type == 'ppe_sinu_dropout':
            h = self.PPE_drop(h, dropout=True)
        
        audio_emb = self.audio_proj(audio_emb)
        src_mask = self.generate_src_mask(T, length).to(x.device).unsqueeze(-1)

        for module in self.temporal_decoder_blocks:
            h = module(h, audio_emb, emb, src_mask, add_cond=add_cond_new, null_cond_emb=self.null_cond_emb) # h:[128, 196, 512], audio_emb:[128, 77, 256], emb:[128, 2048], mask:[128, 196, 1]
        output = self.out(h).view(B, T, -1).contiguous()

        if self.opt.classifier_free and not self.training and self.opt.cond_scale != 1:
            output = output[:output.shape[0]//2] + self.opt.cond_scale * (output[output.shape[0]//2:] - output[:output.shape[0]//2])
        return output


class UniDiffuser(nn.Module):
    def __init__(self,
                 opt,
                 input_feats,
                 audio_dim = 128,
                 style_dim = 4,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu", 
                 num_text_layers=4,
                 aud_latent_dim=256,
                 text_ff_size=2048,
                 text_num_heads=4,
                 no_clip=False,
                 pe_type='learnable',
                 **kargs):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation  
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.opt = opt

        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        if self.opt.cond_projection != 'none':
            self.encoder_aud = LinearTemporalDiffusionTransformerLayer(
                                opt=opt,
                                seq_len=num_frames,
                                latent_dim=audio_dim,
                                aud_latent_dim=0,
                                time_embed_dim=self.time_embed_dim,
                                ffn_dim=ff_size,
                                num_head=num_heads,
                                dropout=dropout,
                                cond_proj=False
                            )
        else:
            self.encoder_aud = LinearTemporalDiffusionTransformerLayer_2(
                                opt=opt,
                                seq_len=num_frames,
                                latent_dim=audio_dim,
                                aud_latent_dim=0,
                                time_embed_dim=self.time_embed_dim,
                                ffn_dim=ff_size,
                                num_head=num_heads,
                                dropout=dropout
                            )
        

        self.opt.expression_only = True
        self.encoder_exp = MotionTransformer(
                    opt=opt,
                    input_feats=opt.expression_dim,
                    audio_dim=audio_dim*2,
                    style_dim=style_dim,
                    num_frames=num_frames,
                    num_layers=num_layers,
                    ff_size=ff_size,
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation, 
                    num_text_layers=num_text_layers,
                    aud_latent_dim=aud_latent_dim,
                    text_ff_size=text_ff_size,
                    text_num_heads=text_num_heads,
                    no_clip=no_clip,
                    pe_type=pe_type,
                    block="expression"
                    )
        self.opt.expression_only = False
        
        self.opt.expCondition_gesture_only = 'pred' 
        self.encoder_ges = MotionTransformer(
                        opt=opt,
                        input_feats=opt.dim_pose,
                        audio_dim=audio_dim*2,
                        style_dim=style_dim,
                        num_frames=num_frames,
                        num_layers=num_layers,
                        ff_size=ff_size,
                        latent_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        activation=activation, 
                        num_text_layers=num_text_layers,
                        aud_latent_dim=aud_latent_dim,
                        text_ff_size=text_ff_size,
                        text_num_heads=text_num_heads,
                        no_clip=no_clip,
                        pe_type=pe_type,
                        block="gesture"
                        )
        self.opt.expCondition_gesture_only = None
        self.opt.gesture_only = False
        
        
        
    def generate_src_mask(self, T, length, causal=False):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        
        if causal:
            for i in range(B):
                for j in range(i+1, length[i]):
                    src_mask[i, j] = 0

        return src_mask

    def _predict_xstart_from_eps(self, x_t, t, eps, sqrt_alphas):
        assert x_t.shape == eps.shape
        sqrt_recip_alphas_cumprod_t, sqrt_recipm1_alphas_cumprod_t = sqrt_alphas
            
        return (
            sqrt_recip_alphas_cumprod_t * x_t
            - sqrt_recipm1_alphas_cumprod_t * eps
        )
    
    
    
    def forward(self, x, timesteps, sqrt_alphas, audio_emb, length, person_id, add_cond={}, pe_type="learnable", y=None):
        
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        B, T = x.shape[0], x.shape[1]
        src_mask = self.generate_src_mask(T, length).to(x.device).unsqueeze(-1)

        # encode aud and concat with itself
        if self.opt.expAddHubert or self.opt.addHubert or self.opt.addWav2Vec2:
            audio_feat = self.encoder_aud(audio_emb, None, emb, src_mask, {})
        else:
            audio_feat = self.encoder_aud(audio_emb, None, emb, src_mask, add_cond)
        audio_emb = torch.cat((audio_emb, audio_feat), dim=-1)

        gesture, expression = torch.split(x, self.opt.split_pos, dim=-1)
        
        self.opt.expression_only = True
        exp_noise_t = self.encoder_exp(expression, timesteps, audio_emb, length, person_id, add_cond, pe_type, y, block="expression")
        self.opt.expression_only = False

        
        self.opt.expCondition_gesture_only = 'pred' 
        expr_cond = self._predict_xstart_from_eps(expression, timesteps, exp_noise_t.detach(), sqrt_alphas)
        audio_emb = torch.cat((audio_emb, expr_cond), dim=-1)
        
        
        if self.opt.visualize_unify_x0_step and timesteps[0] % self.opt.visualize_unify_x0_step == 0:
            for idx, expr in enumerate(expr_cond):
                exp_x0_path = os.path.join(self.opt.unify_x0_step_path, "%05d" % idx, f"{timesteps[idx]}.npy") 
                os.makedirs(os.path.dirname(exp_x0_path), exist_ok=True)
                np.save(exp_x0_path, expr.cpu().numpy())
            
        if self.opt.expAddHubert:
            add_cond = {} # TODO: consider other conditions

        
        ges_noise_t = self.encoder_ges(gesture, timesteps, audio_emb, length, person_id, add_cond, pe_type, y, block="gesture")
        self.opt.expCondition_gesture_only = None
        self.opt.gesture_only = False


        x = torch.cat((ges_noise_t, exp_noise_t), dim=-1)
        
        return x
    
