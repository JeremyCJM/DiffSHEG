import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import pickle
import lmdb
import pyarrow
import torch.nn.functional as F

class ShowDataset(data.Dataset):
    """
    TED dataset.
    Prepares conditioning information (previous poses + control signal) and the corresponding next poses"""

    def __init__(self, opt, data_path):
        """
        Args:
        control_data: The control input
        joint_data: body pose input
        Both with shape (samples, time-slices, features)s
        """
        self.opt = opt

        if self.opt.mode != "test_custom_audio":
            print("Loading data ...")
            self.lmdb_env = lmdb.open(data_path, readonly=True, lock=False)
            if opt.use_aud_feat or self.opt.expAddHubert or self.opt.addHubert:
                self.aud_feat_path = os.path.join(os.path.dirname(os.path.dirname(data_path)), f"cached_aud_hubert/{os.path.basename(data_path).split('_')[-2]}/hubert_large_ls960_ft")
                self.aud_lmdb_env = lmdb.open(self.aud_feat_path, readonly=True, lock=False)
            
            if opt.use_aud_feat or self.opt.addWav2Vec2:
                self.aud_feat_path = os.path.join(os.path.dirname(os.path.dirname(data_path)), f"cached_aud_wav2vec2/{os.path.basename(data_path).split('_')[-2]}/wav2vec2_base_960h")
                self.aud_lmdb_env = lmdb.open(self.aud_feat_path, readonly=True, lock=False)

            with self.lmdb_env.begin() as txn:
                self.n_samples = txn.stat()["entries"] 
            # dict_keys(['poses', 'expression', 'aud_feat', 'speaker', 'aud_file', 'betas'])

        mean_std_dict = np.load("data/SHOW/talkshow_mean_std.npy", allow_pickle=True)[()]

        self.pose_mean = self.extract_pose(torch.from_numpy(mean_std_dict["pose_mean"]).float())
        self.pose_std = self.extract_pose(torch.from_numpy(mean_std_dict["pose_std"]).float())
        self.expression_mean = torch.cat([torch.from_numpy(mean_std_dict["pose_mean"][:3]).float(), torch.from_numpy(mean_std_dict["expression_mean"]).float()], dim=-1)
        self.expression_std = torch.cat([torch.from_numpy(mean_std_dict["pose_mean"][:3]).float(), torch.from_numpy(mean_std_dict["expression_std"]).float()], dim=-1)

        self.motion_mean = torch.cat([self.pose_mean, self.expression_mean], dim=-1)
        self.motion_std = torch.cat([self.pose_std, self.expression_std], dim=-1)

        if self.opt.usePredExpr:
            face_list = os.listdir(self.opt.usePredExpr)
            face_list = [ff for ff in face_list if ff.endswith(".npy")]
            face_list.sort()
            self.face_list = [os.path.join(self.opt.usePredExpr, pp) for pp in face_list]
            


            
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        """
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            pose, expression, aud_raw, mfcc, mel, speaker, aud_file, betas = sample
            # pose, expression, mfcc = pose.T, expression.T, mfcc.T
            pose = torch.from_numpy(pose.copy()).float()
            expression = torch.from_numpy(expression.copy()).float()
            aud_raw = torch.from_numpy(aud_raw.copy()).float()
            mfcc = torch.from_numpy(mfcc.copy()).float() 
            mel = torch.from_numpy(mel.copy()).float()
            speaker = torch.from_numpy(speaker.copy()).float() 
            betas = torch.from_numpy(betas.copy()).float() 

            jaw_pose, leye_pose, reye_pose, global_orient, body_pose, hand_pose = torch.split(pose, [3,3,3,3, 63, 90], dim=-1)
            low1, up1, low2, up2, low3, up3, low4, up4 = torch.split(body_pose, [6, 3, 6, 3, 6, 3, 6, 30], dim=-1)
            pose = torch.cat([up1, up2, up3, up4, hand_pose], dim=-1)
            expression = torch.cat([jaw_pose, expression], dim=-1)

            pose = self.standardize(pose, self.pose_mean, self.pose_std)
            expression = self.standardize(expression, self.expression_mean, self.expression_std)

            hubert = None
            if self.opt.audio_feat == "hubert" or self.opt.expAddHubert or self.opt.addHubert:
                with self.aud_lmdb_env.begin(write=False) as txn_aud:
                    key = "{:005}".format(idx).encode("ascii")
                    hubert = txn_aud.get(key)
                    hubert = pyarrow.deserialize(hubert)
                    hubert = torch.from_numpy(hubert.copy()).float() 
                    hubert = F.interpolate(hubert.swapaxes(-1,-2).unsqueeze(0), size=pose.shape[0], mode='linear', align_corners=True).swapaxes(-1,-2).squeeze()
            
            wav2vec2 = None
            if self.opt.audio_feat == "wav2vec2" or self.opt.addWav2Vec2:
                with self.aud_lmdb_env.begin(write=False) as txn_aud:
                    key = "{:005}".format(idx).encode("ascii")
                    wav2vec2 = txn_aud.get(key)
                    wav2vec2 = pyarrow.deserialize(wav2vec2)
                    wav2vec2 = torch.from_numpy(wav2vec2.copy()).float() 

            if self.opt.audio_feat == "mfcc":
                aud_feat = mfcc
            elif self.opt.audio_feat == "mel":
                aud_feat = mel
            elif self.opt.audio_feat == "raw":
                aud_feat = aud_raw
            elif self.opt.audio_feat == "hubert":
                aud_feat = hubert
            elif self.opt.audio_feat == "wav2vec2":
                aud_feat = wav2vec2

            if self.opt.expAddHubert or self.opt.addHubert:
                return {'poses': pose,
                        'expression': expression,
                        'aud_feat': aud_feat,
                        'pretrain_aud_feat': hubert,
                        'speaker': speaker,
                        'aud_file': aud_file,
                        'betas': betas
                        }
            elif self.opt.addWav2Vec2:
                return {'poses': pose,
                        'expression': expression,
                        'aud_feat': aud_feat,
                        'pretrain_aud_feat': wav2vec2,
                        'speaker': speaker,
                        'aud_file': aud_file,
                        'betas': betas
                        }
            else:
                return {'poses': pose,
                        'expression': expression,
                        'aud_feat': aud_feat,
                        'speaker': speaker,
                        'aud_file': aud_file,
                        'betas': betas
                        }
        
    def extract_pose(self, pose):
        jaw_pose, leye_pose, reye_pose, global_orient, body_pose, hand_pose = torch.split(pose, [3,3,3,3, 63, 90], dim=-1)
        low1, up1, low2, up2, low3, up3, low4, up4 = torch.split(body_pose, [6, 3, 6, 3, 6, 3, 6, 30], dim=-1)
        # pose = torch.cat([jaw_pose, up1, up2, up3, up4, hand_pose], dim=-1)
        pose = torch.cat([up1, up2, up3, up4, hand_pose], dim=-1)
        return pose
    
    def standardize(self, data, mean, std):
        scaled = (data - mean) / std
        return scaled

    def inv_standardize(self, data, mean, std):
        try:
            inv_scaled = data * std + mean
        except:
            inv_scaled = data * std.numpy() + mean.numpy()
        return inv_scaled