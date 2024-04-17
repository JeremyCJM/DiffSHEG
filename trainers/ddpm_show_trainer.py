import numpy as np
import torch
import torch.nn.functional as F
import random
import time
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist
from tqdm import tqdm
import os
from enum import Enum
from utils import other_tools
from models.respace import SpacedDiffusion, space_timesteps

from datasets import data_tools


import wandb

from mmcv.runner import get_dist_info
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

import datasets.rotation_converter as rot_cvt
from .loss_factory import get_loss_func


class DDPMTrainer_show(object):

    def __init__(self, args, encoder, eval_model=None):
        self.opt = args
        self.device = args.device
        self.encoder = encoder
        self.epoch = 0
        self.eval_model = eval_model
        if eval_model is not None and 'test' not in self.opt.mode:
            self.load_fid_net(args.e_path)
            self.eval_model.eval()
            
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        model_mean_type = {
            "epsilon": ModelMeanType.EPSILON,
            "start_x": ModelMeanType.START_X,
            "previous_x": ModelMeanType.PREVIOUS_X
        }


        self.diffusion = GaussianDiffusion(
            opt=args,
            betas=betas,
            model_mean_type=model_mean_type[args.model_mean_type],
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        
        if self.opt.ddim:
            self.diffusion_ddim_val = SpacedDiffusion(
                use_timesteps=space_timesteps(self.diffusion_steps, 'ddim25'),
                opt=args,
                betas=betas,
                model_mean_type=model_mean_type[args.model_mean_type],
                model_var_type=ModelVarType.FIXED_SMALL,
                loss_type=LossType.MSE,
                rescale_timesteps=False,
            )

        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler

        self.huber_loss = get_loss_func("huber_loss")

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

        if self.opt.mode == 'train' and not self.opt.debug:
            if self.opt.multiprocessing_distributed:
                wandb.init(project=f"Diffusion_{self.opt.dataset_name}", group=f"DDP_{self.opt.name}") 
            else:
                wandb.init(project=f"Diffusion_{self.opt.dataset_name}")
            wandb.run.name = f"{self.opt.name}"

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False, add_cond={}, inpaint_dict=None):
        audio_emb, motions, p_id = batch_data
        if self.opt.use_single_style:
            p_id = torch.zeros_like(p_id)
            p_id[:, :1] = 1
        

        audio_emb = audio_emb.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()
        p_id = p_id.detach().to(self.device).float()
        
        self.audio_emb = audio_emb
        self.motions = motions
        x_start = motions
        B, T = x_start.shape[:2]
        
        cur_len = torch.LongTensor([T for ii in range(B)]).to(self.device)
        t, _ = self.sampler.sample(B, x_start.device)

        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={
                "audio_emb": audio_emb, 
                "length": cur_len, 
                "person_id": p_id,
                "add_cond": add_cond,
                "y": inpaint_dict,
                "pe_type": self.opt.PE
            }
        )

        self.real_noise = output['target']
        self.fake_noise = output['pred']

        
        self.real_vel = output['target_vel']
        self.fake_vel = output['pred_vel']
        if self.opt.model_mean_type == 'epsilon':
            self.real_x0 = output['target_x0']
            self.fake_x0 = output['pred_x0']

        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)
        
        

    def generate_batch(self, audio_emb, p_id, dim_pose, add_cond={}, inpaint_dict=None):
        audio_emb = audio_emb.to(self.device)
        B = len(audio_emb)
        T = audio_emb.shape[1]
        cur_len = torch.LongTensor([T for ii in range(B)]).to(self.device)

        if self.opt.ddim:
            output = self.diffusion_ddim_val.ddim_sample_loop(
                self.encoder,
                (B, T, dim_pose),
                clip_denoised=False,
                progress=True,
                model_kwargs={
                    "audio_emb": audio_emb, 
                    "length": cur_len, 
                    "person_id": p_id,
                    "add_cond": add_cond,
                    "y": inpaint_dict,
                    "pe_type": self.opt.PE
                })
        else:
            output = self.diffusion.p_sample_loop(
                self.encoder,
                (B, T, dim_pose),
                clip_denoised=False,
                progress=True,
                model_kwargs={
                    "audio_emb": audio_emb, 
                    "length": cur_len, 
                    "person_id": p_id,
                    "add_cond": add_cond,
                    "y": inpaint_dict,
                    "pe_type": self.opt.PE
                })

        return output

    def backward_G(self):
        if self.opt.expr_weight == 1:
            loss_model_pred = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        else:
            loss_model_pred = self.mse_criterion(self.fake_noise[..., :self.opt.lower_dim], 
                                              self.real_noise[..., :self.opt.lower_dim]).mean(dim=-1) + \
                            self.mse_criterion(self.fake_noise[..., self.opt.lower_dim:], 
                                              self.real_noise[..., self.opt.lower_dim:]).mean(dim=-1) * \
                            self.opt.expr_weight
        loss_model_pred = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        loss_model_pred = (loss_model_pred * self.src_mask).sum() / self.src_mask.sum()
        self.loss_model_pred = 1000 * loss_model_pred
        self.final_loss = self.loss_model_pred
        loss_logs = OrderedDict({})
        loss_logs['loss_model_pred'] = self.loss_model_pred.item()
        
        

        if self.opt.add_vel_loss and self.epoch > self.opt.vel_loss_start:
            loss_vel_rec = self.mse_criterion(self.fake_vel, self.real_vel).mean(dim=-1)
            loss_vel_rec = (loss_vel_rec * self.src_mask[:, :-1]).sum() / self.src_mask[:, :-1].sum()
            # self.loss_vel_rec = loss_vel_rec
            self.loss_vel_rec = 100 * loss_vel_rec
            loss_logs['loss_vel_rec'] = self.loss_vel_rec.item()
            self.final_loss += loss_vel_rec

            if self.opt.model_mean_type == 'epsilon':
                if self.opt.dataset_name == 'beat' and self.opt.sem_rep is not None:
                    loss_x0_rec = self.huber_loss(self.real_x0*(self.in_sem.unsqueeze(2)+1), self.fake_x0*(self.in_sem.unsqueeze(2)+1))
                else: 
                    loss_x0_rec = self.huber_loss(self.real_x0, self.fake_x0)
                # self.loss_x0_rec = 200 * loss_x0_rec
                self.loss_x0_rec = 100 * loss_x0_rec
                loss_logs['loss_x0_rec'] = self.loss_x0_rec.item()
                self.final_loss += self.loss_x0_rec
                
        loss_logs['final_loss'] = self.final_loss.item()
        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.final_loss.backward()
        self.clip_norm([self.encoder])
        self.step([self.opt_encoder])

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it, fgd, mse, pck, best_fgd, best_mse, best_pck):
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'FGD': fgd,
            'best_fgd': best_fgd,
            'MSE': mse,
            'best_mse': best_mse,
            'PCK': pck,
            'best_pck': best_pck
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.PE == "pe_sinu_repeat":
            mm = checkpoint['encoder']['PE.pe'][:, :self.n_poses, :]
            checkpoint['encoder']['PE.pe'] = torch.cat((mm, mm, mm, mm), -2)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        try:
            self.encoder.module.load_state_dict(checkpoint['encoder'], strict=False)
        except:
            self.encoder.load_state_dict(checkpoint['encoder'], strict=False)

        return checkpoint['ep'], checkpoint.get('total_it', 0), \
                checkpoint.get('best_fgd', 99999), checkpoint.get('best_mse', 99999), \
                checkpoint.get('best_pck', 0)
    
    def load_fid_net(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        try:
            state_dict = checkpoint['model_state']
        except:
            state_dict = checkpoint['state_dict']
        try:
            self.eval_model.load_state_dict(state_dict, strict=True)
        except:
            try:
                self.eval_model.load_state_dict({k.replace('module.', ''):v for k,v in state_dict.items()}, strict=True)
            except:
                self.eval_model.module.load_state_dict(state_dict, strict=True)
    

    def one_hot(self, ids, dim):
        ones_eye = torch.eye(dim)
        return (ones_eye[ids.long()].squeeze())
    

    def train(self, train_dataset, val_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        it = 0
        cur_epoch = 0
        best_fgd = 99999
        best_mse = 99999
        best_pck = 0
        if self.opt.resume:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it, best_fgd, best_mse, best_pck = self.load(model_dir)
            if self.opt.reset_lr:
                for i, param_group in enumerate(self.opt_encoder.param_groups):
                    param_group['lr'] = self.opt.lr

        start_time = time.time()

        if self.opt.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.opt.batch_size, shuffle=(train_sampler is None),
            num_workers=self.opt.workers, pin_memory=True, sampler=train_sampler)

        if not self.opt.classifier_free:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.opt.batch_size, shuffle=False,
                num_workers=self.opt.workers, pin_memory=True, sampler=val_sampler)
        else:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.opt.batch_size//2, shuffle=False,
                num_workers=self.opt.workers, pin_memory=True, sampler=val_sampler)
        
        ##################################### Training #####################################
        print("Start training ...")
        logs = OrderedDict()
        
        
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            self.epoch = epoch
            if self.opt.distributed:
                train_sampler.set_epoch(epoch)
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                # tt1 = time.time()
                if not self.opt.expression_only:
                    tar_pose = batch_data["poses"].to(self.device) 
                    if self.opt.remove_hand:
                        tar_pose = tar_pose[..., :63]
                    tar_pose = tar_pose.to(self.device)
                if not self.opt.gesture_only:
                    in_facial = batch_data["expression"].to(self.device)  # torch.Size([256, 34, 51])                

                if self.opt.expression_only:
                    motions = in_facial
                elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None:
                    motions = tar_pose
                else:
                    motions = torch.cat((tar_pose, in_facial), dim=-1)

                audio_emb = batch_data["aud_feat"].to(self.device) 

                if self.opt.expCondition_gesture_only:
                    audio_emb = torch.cat((audio_emb, in_facial), dim=-1)

                p_id = batch_data["speaker"] - 20
                p_id = self.one_hot(p_id, self.opt.speaker_dim).to(self.device)

                add_cond = {}
                if self.opt.expAddHubert or self.opt.addHubert or self.opt.addWav2Vec2:
                    add_cond["pretrain_aud_feat"] = batch_data["pretrain_aud_feat"].to(self.device)

                if self.opt.remove_audio:
                    audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
                if self.opt.remove_style:
                    p_id = torch.zeros_like(p_id).to(p_id.device)
                
                batch_data = [audio_emb, motions, p_id]

                inpaint_dict = {}
                if self.opt.overlap_len > 0:
                    inpaint_dict['gt'] = motions
                    inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                    device=motions.device)  # Do inpainting/generation in those frames
                    inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                self.forward(batch_data, add_cond=add_cond, inpaint_dict=inpaint_dict)
                log_dict = self.update()
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    
                    if rank == 0:
                        print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                    if not self.opt.debug:
                        wandb.log(mean_loss, step=epoch)

                if self.opt.debug:
                    break

            if rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, None, None, None, best_fgd, best_mse, best_pck)

            if (epoch+1) % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, it, None, None, None, best_fgd, best_mse, best_pck)
            
            ################################## Evaluation ####################################
            
            
            # embed_space_evaluator = EmbeddingSpaceEvaluator(self.opt.embed_net_path, self.device)
            if (epoch+1) % self.opt.eval_every_e == 0:
                self.eval_mode()
                count = 0.0
                diversity = AverageMeter('diversity')
                pck = AverageMeter('pck')
                mse = AverageMeter('mse')
                fgd = AverageMeter('fgd')
                progress = ProgressMeter(
                    len(val_loader) + (self.opt.distributed and (len(val_loader.sampler) * self.opt.world_size < len(val_loader.dataset))),
                    [diversity, pck, mse],
                    prefix='Test: ')
                with torch.no_grad():
                    for i, batch_data in tqdm(enumerate(val_loader)):
                        if not self.opt.expression_only:
                            tar_pose = batch_data["poses"].to(self.device) 
                            if self.opt.remove_hand:
                                tar_pose = tar_pose[..., :63]
                            tar_pose = tar_pose.to(self.device)
                        if not self.opt.gesture_only:
                            in_facial = batch_data["expression"].to(self.device)  # torch.Size([256, 34, 51])                

                        if self.opt.expression_only or self.opt.gesCondition_expression_only:
                            motions = in_facial
                        elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None:
                            motions = tar_pose
                        else:
                            motions = torch.cat((tar_pose, in_facial), dim=-1)

                        audio_emb = batch_data["aud_feat"].to(self.device) 

                        if self.opt.expCondition_gesture_only:
                            audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
                        
                        add_cond = {}
                        if self.opt.expAddHubert or self.opt.addHubert or self.opt.addWav2Vec2:
                            add_cond["pretrain_aud_feat"] = batch_data["pretrain_aud_feat"].to(self.device)

                        p_id = batch_data["speaker"] - 20
                        p_id = self.one_hot(p_id, self.opt.speaker_dim).to(self.device)

                        if self.opt.remove_audio:
                            audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
                        if self.opt.remove_style:
                            p_id = torch.zeros_like(p_id).to(p_id.device)
                        
                        batch_data = [audio_emb, motions, p_id]

                        if self.opt.use_single_style:
                            p_id = torch.zeros_like(p_id)
                            p_id[:, :1] = 1

                        audio_emb = audio_emb.detach().float()
                        motions = motions.detach().float()
                        p_id = p_id.detach().float()
                        count += len(motions)

                        inpaint_dict = {}
                        if self.opt.overlap_len > 0:
                            inpaint_dict['gt'] = motions
                            inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                            device=motions.device)  # Do inpainting/generation in those frames
                            inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 

                        outputs = self.generate_batch(audio_emb, p_id, self.opt.net_dim_pose, add_cond, inpaint_dict)
                        B, seq, C = outputs.shape

                        if not self.opt.no_fgd:
                            latent_out = self.eval_model(outputs[:, :88, :].float())
                            latent_ori = self.eval_model(motions[:, :88, :].float())
                            if i == 0:
                                latent_out_all = latent_out.cpu().numpy()
                                latent_ori_all = latent_ori.cpu().numpy()
                            else:
                                latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()], axis=0)
                                latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()], axis=0)

                        ## to cpu
                        outputs, motions = outputs.cpu(), motions.cpu()
                        
                        motions = motions.unsqueeze(-1).numpy()
                        outputs = outputs.unsqueeze(-1).numpy()


                        ### MSE & PCK
                        
                        diff = outputs - motions
                        diff_square = diff ** 2
                        correct = np.sum(diff_square, axis=3)
                        correct = np.sqrt(correct) < 0.5
                        pck_val = np.mean(correct)
                        mse_val = np.mean(diff_square)
                        
                        ### diversity
                        B_div = 50 ## In Ye et al. (ECCV'22), Batch size is 50 when evaluating diversity
                        if B < B_div:
                            B_div = B
                        # out_split = outputs.split(B_div, dim=0)
                        out_split = np.split(outputs, np.arange(B_div, B, B_div), axis=0)
                        for idx in range(B // B_div):
                            div_val = 0.0
                            for ii in range(0, B_div):
                                for jj in range(ii+1, B_div):
                                    dif = out_split[idx][ii,:,:,:] - out_split[idx][jj,:,:,:]
                                    div_val += np.mean(np.absolute(dif)) ### TODO: check: np.mean() or np.sum()
                            div_val = div_val * 2 / (B_div * (B_div - 1)) 
                            # update in averagemeter
                            diversity.update(div_val, B_div)

                        
                        # update in averagemeter
                        mse.update(mse_val, B)
                        pck.update(pck_val, B)

                        if self.opt.debug or (self.opt.max_eval_samples != -1 and pck.count >= self.opt.max_eval_samples):
                            break

                    if self.opt.distributed:  #  all_reduce() is for aggreagating results on each rank
                        diversity.all_reduce()
                        mse.all_reduce()
                        pck.all_reduce()
                    
                    if rank == 0:
                        if not self.opt.debug:
                            wandb.log({"MSE": mse.avg, "PCK": pck.avg, "Diversity": diversity.avg}, step = epoch)
                        print(f"[Validation]: Epoch: {epoch}, MSE: {mse.avg}, PCK: {pck.avg}, Diversity: {diversity.avg}")
                    
                    if not self.opt.no_fgd:
                        fgd_val = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
                        fgd.update(fgd_val)
                        if self.opt.distributed:
                            fgd.all_reduce()
                        if rank == 0:
                            if not self.opt.debug:
                                wandb.log({"FGD": fgd.avg}, step = epoch)
                            print(f"[Validation]: Epoch: {epoch}, FGD: {fgd.avg}")

                    if best_fgd > fgd.avg and rank == 0 and self.eval_model != None:
                        best_fgd = fgd.avg
                        self.save(pjoin(self.opt.model_dir, 'fgd_best.tar'), epoch, it, fgd.avg, mse.avg, pck.avg, best_fgd, best_mse, best_pck)
                    if best_mse > mse.avg and rank == 0:
                        best_mse = mse.avg
                        self.save(pjoin(self.opt.model_dir, 'mse_best.tar'), epoch, it, fgd.avg, mse.avg, pck.avg, best_fgd, best_mse, best_pck)
                    if best_pck < pck.avg and rank == 0:
                        best_pck = pck.avg
                        self.save(pjoin(self.opt.model_dir, 'pck_best.tar'), epoch, it, fgd.avg, mse.avg, pck.avg, best_fgd, best_mse, best_pck)
                    
            self.train_mode()


    def test(self, test_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt.is_train = False
        cur_epoch = 0

        model_dir = pjoin(self.opt.model_dir, self.opt.ckpt)
        cur_epoch, it, _, _, _ = self.load(model_dir)

        if self.opt.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
        else:
            test_sampler = None
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.opt.batch_size, shuffle=False,
            num_workers=self.opt.workers, pin_memory=True, sampler=test_sampler)

        logs = OrderedDict()

        self.encoder.eval()
        ckpt_epoch = f"ckpt_e{cur_epoch}"
        if 'fgd_best' in model_dir:
            ckpt_epoch = f"BestFGD_e{cur_epoch}"
        elif 'mse_best' in model_dir:
            ckpt_epoch = f"BestMSE_e{cur_epoch}"
        elif 'pck_best' in model_dir:
            ckpt_epoch = f"BestPCK_e{cur_epoch}"
        if self.opt.ddim:
            ckpt_epoch = ckpt_epoch + f"_{self.opt.timestep_respacing}" 

        results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.name, ckpt_epoch)
        if self.opt.fix_very_first:
            results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first")
        else:
            results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}")

        if self.opt.classifier_free:
            results_dir = results_dir + f"_ClsFreeCondScale{self.opt.cond_scale}"
            
            
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        middle_name = self.opt.mode
        if self.opt.test_on_trainset:
            results_dir = results_dir.replace(self.opt.mode, "test_on_trainset")
            middle_name = "test_on_trainset"
        elif self.opt.test_on_val:
            results_dir = results_dir.replace(self.opt.mode, "test_on_val")
            middle_name = "test_on_val"
        
        if self.opt.usePredExpr:
            results_dir = results_dir.replace(middle_name, middle_name + "_usePredExpr")

        if self.opt.output_gt:
            results_dir = results_dir.replace(middle_name, middle_name + "_GT")

        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        if self.opt.expression_only:
            json_dir = os.path.join(results_dir, "face_json")
            os.makedirs(json_dir, exist_ok=True)



        count = 0
        for i, batch_data in enumerate(test_loader):
            if not self.opt.expression_only:
                tar_pose = batch_data["poses"].to(self.device) 
                if self.opt.remove_hand:
                    tar_pose = tar_pose[..., :63]
                tar_pose = tar_pose.to(self.device)
            if not self.opt.gesture_only:
                in_facial = batch_data["expression"].to(self.device)  # torch.Size([256, 34, 51])                

            if self.opt.expression_only:
                motions = in_facial
            elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None:
                motions = tar_pose
            else:
                motions = torch.cat((tar_pose, in_facial), dim=-1)

            audio_emb = batch_data["aud_feat"].to(self.device) 

            if self.opt.expCondition_gesture_only:
                audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
            elif self.opt.gesCondition_expression_only:
                audio_emb = torch.cat((audio_emb, tar_pose), dim=-1)

            add_cond = {}
            if self.opt.expAddHubert or self.opt.addHubert or self.opt.addWav2Vec2:
                add_cond["pretrain_aud_feat"] = batch_data["pretrain_aud_feat"].to(self.device)


            p_id = batch_data["speaker"] - 20
            p_id = self.one_hot(p_id, self.opt.speaker_dim).to(self.device)

            if self.opt.remove_audio:
                audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
            if self.opt.remove_style:
                p_id = torch.zeros_like(p_id).to(p_id.device)
            
            batch_data = [audio_emb, motions, p_id]

            if not self.opt.output_gt:
                ###### debug
                if self.opt.use_single_style:
                    
                    p_id = torch.zeros_like(p_id)
                    p_id[:, :1] = 1
                ###### debug end
                

                inpaint_dict = {}
                if self.opt.overlap_len > 0:
                    inpaint_dict['gt'] = motions
                    inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                    device=motions.device)  # Do outpainting/generation in those frames
                    inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 

                
                outputs = self.generate_batch(audio_emb, p_id, self.opt.net_dim_pose, add_cond, inpaint_dict)
                
                outputs = outputs.cpu().numpy()

            else:
                outputs = motions.cpu().numpy()
            

            if self.opt.expression_only or self.opt.gesCondition_expression_only:
                outputs = test_dataset.inv_standardize(outputs, test_dataset.expression_mean, test_dataset.expression_std)
            elif self.opt.gesture_only or self.opt.expCondition_gesture_only:
                outputs = test_dataset.inv_standardize(outputs, test_dataset.pose_mean, test_dataset.pose_std)
            else:
                outputs = test_dataset.inv_standardize(outputs, test_dataset.motion_mean, test_dataset.motion_std)

            for out in outputs:
                np.save(pjoin(results_dir, "%05d.npy" % count), out)
                count += 1

            if self.opt.debug:
                break
        return results_dir

    def test_arbitrary_len(self, test_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt.is_train = False
        cur_epoch = 0

        model_dir = pjoin(self.opt.model_dir, self.opt.ckpt)
        cur_epoch, it, _, _, _ = self.load(model_dir)

        if self.opt.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
        else:
            test_sampler = None
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.opt.batch_size, shuffle=False,
            num_workers=self.opt.workers, pin_memory=True, sampler=test_sampler)

        logs = OrderedDict()

        self.encoder.eval()
        ckpt_epoch = f"ckpt_e{cur_epoch}"
        if 'fgd_best' in model_dir:
            ckpt_epoch = f"BestFGD_e{cur_epoch}"
        elif 'mse_best' in model_dir:
            ckpt_epoch = f"BestMSE_e{cur_epoch}"
        elif 'pck_best' in model_dir:
            ckpt_epoch = f"BestPCK_e{cur_epoch}"
        if self.opt.ddim:
            ckpt_epoch = ckpt_epoch + f"_{self.opt.timestep_respacing}"
            if self.opt.addBlend:
                ckpt_epoch = ckpt_epoch + f"_lastStepInterp"

        results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.name, ckpt_epoch)

        if self.opt.rename:
            results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.rename, ckpt_epoch)

        middle_name = self.opt.mode
        # if self.opt.overlap_len > 0:
        if self.opt.fix_very_first:
            results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first")
            middle_name = f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first"
        else:
            results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}")
            middle_name = f"{self.opt.name}/fixStart{self.opt.overlap_len}"
            
        if self.opt.usePredExpr:
            results_dir = results_dir.replace(middle_name, middle_name + "_usePredExpr")
        
        if self.opt.classifier_free:
            results_dir = results_dir.replace(middle_name, middle_name + f"_ClsFreeCondScale{self.opt.cond_scale}")
        
        if self.opt.output_gt:
            results_dir = results_dir.replace(middle_name, middle_name + "_GT")
            
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        if self.opt.unidiffuser or self.opt.net_dim_pose == self.opt.dim_pose + self.opt.expression_dim:
            ori_results_dir = results_dir
            results_dir = os.path.join(ori_results_dir, "gesture")
            results_dir_expr = os.path.join(ori_results_dir, "expression")

            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(results_dir_expr, exist_ok=True)
        
        def get_windows(x, size, step):
            if isinstance(x, dict):
                out = {}
                for key in x.keys():
                    out[key] = get_windows(x[key], size, step)
                out_dict_list = []
                for i in range(len(out[list(out.keys())[0]])):
                    out_dict_list.append({key: out[key][i] for key in out.keys()})
                return out_dict_list
            else:
                seq_len = x.shape[1]
                if seq_len <= size:
                    return [x]
                else:
                    win_num = (seq_len - (size-step)) / float(step)
                    out = [x[:, mm*step : mm*step + size, ...] for mm in range(int(win_num))]
                    if win_num - int(win_num) != 0:
                        out.append(x[:, int(win_num)*step:, ...])  
                    return out
                
        count = 0
        for i, batch_data in enumerate(test_loader):
            if not self.opt.expression_only:
                
                tar_pose = batch_data["poses"].to(self.device) 
                if self.opt.remove_hand:
                    tar_pose = tar_pose[..., :63]
                tar_pose = tar_pose.to(self.device)
            if not self.opt.gesture_only:
                in_facial = batch_data["expression"].to(self.device)  # torch.Size([256, 34, 51])                

            if self.opt.expression_only or self.opt.gesCondition_expression_only:
                motions = in_facial
            elif self.opt.gesture_only or self.opt.expCondition_gesture_only != None:
                motions = tar_pose
            else:
                motions = torch.cat((tar_pose, in_facial), dim=-1)

            audio_emb = batch_data["aud_feat"].to(self.device) 

            if self.opt.expCondition_gesture_only:
                audio_emb = torch.cat((audio_emb, in_facial), dim=-1)
            elif self.opt.gesCondition_expression_only:
                audio_emb = torch.cat((audio_emb, tar_pose), dim=-1)

            add_cond = {}
            if self.opt.expAddHubert or self.opt.addHubert or self.opt.addWav2Vec2:
                add_cond["pretrain_aud_feat"] = batch_data["pretrain_aud_feat"].to(self.device)


            p_id = batch_data["speaker"] - 20
            p_id = self.one_hot(p_id, self.opt.speaker_dim).to(self.device)

            # print(f'epoch {epoch} iter {} load data done')
            if self.opt.remove_audio:
                audio_emb = torch.zeros_like(audio_emb).to(audio_emb.device)
            if self.opt.remove_style:
                p_id = torch.zeros_like(p_id).to(p_id.device)
            
            batch_data = [audio_emb, motions, p_id]
            

            if not self.opt.output_gt:
                window_step = self.opt.n_poses - self.opt.overlap_len
                audio_emb_list = get_windows(audio_emb, self.opt.n_poses, window_step)
                motions_list = get_windows(motions, self.opt.n_poses, window_step)
                if add_cond not in [None, {}]:
                    add_cond_list = get_windows(add_cond, self.opt.n_poses, window_step)
                    
                    
                
                out_motions = []
                time_start = time.time()
                for ii, [audio_emb, motions] in enumerate(zip(audio_emb_list, motions_list)):
                    print(f"Rank {rank}: Video {i+1} / {len(test_loader)}, Clip {ii+1} / {len(audio_emb_list)} ")
                    if add_cond not in [None, {}]:
                        add_cond = add_cond_list[ii]
                    inpaint_dict = {}
                    if self.opt.overlap_len > 0:
                        inpaint_dict['gt'] = torch.zeros_like(motions)
                        inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                        device=motions.device)  # Do inpainting/generation in those frames
                        # m_lens[0] = motions.shape[1]
                        
                        if ii == 0:
                            if self.opt.fix_very_first:
                                inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                                inpaint_dict['gt'][:, :self.opt.overlap_len, ...] = motions[:, -self.opt.overlap_len:, ...]
                            else:
                                pass
                        elif ii > 0:
                            inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                            inpaint_dict['gt'][:, :self.opt.overlap_len, ...] = outputs[:, -self.opt.overlap_len:, ...]

                    
                    outputs = self.generate_batch(audio_emb, p_id, self.opt.net_dim_pose, add_cond, inpaint_dict)
                    outputs_np = outputs.cpu().numpy()
                    if ii == len(motions_list) - 1:
                        out_motions.append(outputs_np)
                    else:
                        out_motions.append(outputs_np[:, :window_step])
                    
                    if self.opt.debug:
                        break

                out_motions = np.concatenate(out_motions, 1)
                time_end = time.time()
                # print(f"Time cost: {time_end - time_start}; Frames: {out_motions.shape[1]}; FPS: {out_motions.shape[1] / (time_end - time_start)}")
            else:
                out_motions = motions.cpu().numpy()
            

            if self.opt.expression_only or self.opt.gesCondition_expression_only:
                out_motions = test_dataset.inv_standardize(out_motions, test_dataset.expression_mean, test_dataset.expression_std)
            elif self.opt.gesture_only or self.opt.expCondition_gesture_only:
                out_motions = test_dataset.inv_standardize(out_motions, test_dataset.pose_mean, test_dataset.pose_std)
            else:
                out_motions = test_dataset.inv_standardize(out_motions, test_dataset.motion_mean, test_dataset.motion_std)
            
            if self.opt.unidiffuser or self.opt.net_dim_pose == self.opt.dim_pose + self.opt.expression_dim:
                out_motions, out_expression = np.split(out_motions, [self.opt.split_pos], axis=-1)

            
            if self.opt.distributed:
                np.save(pjoin(results_dir,  "%05d" % count + f"_rank{rank}.npy"), out_motions)
                if self.opt.unidiffuser or self.opt.net_dim_pose == self.opt.dim_pose + self.opt.expression_dim:
                    np.save(pjoin(results_dir_expr,  "%05d" % count + f"_rank{rank}.npy"), out_expression)
            else:
                np.save(pjoin(results_dir, "%05d.npy" % count), out_motions)
                if self.opt.unidiffuser or self.opt.net_dim_pose == self.opt.dim_pose + self.opt.expression_dim:
                    np.save(pjoin(results_dir_expr, "%05d.npy" % count), out_expression)
            count += 1

        torch.distributed.barrier()
        if rank == 0:
            print("Running distribution test")
            infer_path = os.path.join(os.getcwd(), os.path.dirname(results_dir))
            os.chdir("0_BEAT_ori/codes/audio2pose/")
            os.system(f"python test_talkshow_ddpm.py --config configs/talkshow_88_GesExpr.yaml --ddpm_result_path {infer_path} --port 34253 ")
            os.chdir("../../../")
        return results_dir
    
    def test_custom_aud(self, test_audio_path, test_dataset):
        import librosa 
        
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt.is_train = False
        cur_epoch = 0

        model_dir = pjoin(self.opt.model_dir, self.opt.ckpt)
        cur_epoch, it, _, _, _ = self.load(model_dir)

        if self.opt.expAddHubert or self.opt.addHubert:
            from transformers import Wav2Vec2Processor, HubertModel
            print("Loading the Wav2Vec2 Processor...")
            wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
            print("Loading the HuBERT Model...")
            hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

        if os.path.isdir(test_audio_path):
            aud_list = os.listdir(test_audio_path)
            aud_list = [aud for aud in aud_list if aud.endswith(".wav")]
            aud_list.sort()
        else:
            aud_list = [os.path.basename(test_audio_path)]
            test_audio_path = os.path.dirname(test_audio_path)
        
        def get_windows(x, size, step):
            if isinstance(x, dict):
                out = {}
                for key in x.keys():
                    out[key] = get_windows(x[key], size, step)
                out_dict_list = []
                for i in range(len(out[list(out.keys())[0]])):
                    out_dict_list.append({key: out[key][i] for key in out.keys()})
                return out_dict_list
            else:
                seq_len = x.shape[1]
                if seq_len <= size:
                    return [x]
                else:
                    win_num = (seq_len - (size-step)) / float(step)
                    out = [x[:, mm*step : mm*step + size, ...] for mm in range(int(win_num))]
                    if win_num - int(win_num) != 0:
                        out.append(x[:, int(win_num)*step:, ...])  
                    return out
                
        

        logs = OrderedDict()
        self.encoder.eval()


        ckpt_epoch = f"ckpt_e{cur_epoch}"
        if 'fgd_best' in model_dir:
            ckpt_epoch = f"BestFGD_e{cur_epoch}"
        elif 'mse_best' in model_dir:
            ckpt_epoch = f"BestMSE_e{cur_epoch}"
        elif 'pck_best' in model_dir:
            ckpt_epoch = f"BestPCK_e{cur_epoch}"
        if self.opt.ddim:
            ckpt_epoch = ckpt_epoch + f"_{self.opt.timestep_respacing}"
            if self.opt.addBlend:
                ckpt_epoch = ckpt_epoch + f"_lastStepInterp"
        


        for p_id_ori in [1,2,3,4]:
            results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.name, ckpt_epoch, f"pid_{p_id_ori}")

            if self.opt.rename:
                results_dir = pjoin("results", f"{self.opt.dataset_name}_{self.opt.n_poses}", self.opt.mode, self.opt.rename, ckpt_epoch, f"pid_{p_id_ori}")

            middle_name = self.opt.mode
            if self.opt.overlap_len > 0:
                if self.opt.fix_very_first:
                    results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first")
                    middle_name = f"{self.opt.name}/fixStart{self.opt.overlap_len}_fix_very_first"
                else:
                    results_dir = results_dir.replace(self.opt.name, f"{self.opt.name}/fixStart{self.opt.overlap_len}")
                    middle_name = f"{self.opt.name}/fixStart{self.opt.overlap_len}"
            

            if self.opt.usePredExpr:
                results_dir = results_dir.replace(middle_name, middle_name + "_usePredExpr")

            if self.opt.output_gt:
                results_dir = results_dir.replace(middle_name, middle_name + "_GT")
            
                
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)

            if self.opt.unidiffuser or self.opt.net_dim_pose == 192:
                ori_results_dir = results_dir
                results_dir = os.path.join(ori_results_dir, "gesture")
                results_dir_expr = os.path.join(ori_results_dir, "expression")

                os.makedirs(pjoin(results_dir_expr, 'face_json'), exist_ok=True)
                os.makedirs(pjoin(results_dir, 'bvh'), exist_ok=True)
                os.makedirs(results_dir, exist_ok=True)
                os.makedirs(results_dir_expr, exist_ok=True)
        

            p_id = p_id_ori - 1
            p_id = torch.ones((len(aud_list), 1)) * p_id
            p_id = self.one_hot(p_id, self.opt.speaker_dim).detach().to(self.device)
            
            
            sr = 16000
            for i, name in enumerate(aud_list):
                time_sum = 0
                aud_path = os.path.join(test_audio_path, name)
                if name.endswith(".wav"):
                    aud_ori, sr = librosa.load(aud_path)
                elif name.endswith(".npy"):
                    aud_ori = np.load(aud_path)
                
                aud = librosa.resample(aud_ori, orig_sr=sr, target_sr=18000)

                time_start = time.time()
                mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)
                time_1 = time.time()
                time_sum += time_1 - time_start
                mel = mel[..., :-1]
                audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))
                audio_emb = audio_emb.unsqueeze(0).to(self.device)
                B, N, _ = audio_emb.shape
                C = self.opt.net_dim_pose
                motions = torch.zeros((B, audio_emb.shape[-2], C)).to(self.device)

                window_step = self.opt.n_poses - self.opt.overlap_len
                audio_emb_list = get_windows(audio_emb, self.opt.n_poses, window_step)
                motions_list = get_windows(motions, self.opt.n_poses, window_step)

                add_cond = {}

                if self.opt.expAddHubert or self.opt.addHubert:
                    time_2 = time.time()
                    add_cond["pretrain_aud_feat"] = get_hubert_from_16k_speech_long(hubert_model, wav2vec2_processor, torch.from_numpy(aud_ori).unsqueeze(0).to(self.device), device=self.device)
                    add_cond["pretrain_aud_feat"] = F.interpolate(add_cond["pretrain_aud_feat"].swapaxes(-1,-2).unsqueeze(0), size=audio_emb.shape[-2], mode='linear', align_corners=True).swapaxes(-1,-2)
                    time_3 = time.time()
                    time_sum += time_3 - time_2
                    
                # Put dict values into self.deivce
                if isinstance(add_cond, dict):
                    for key in add_cond.keys():
                        add_cond[key] = add_cond[key].to(self.device)


                if add_cond not in [None, {}]:
                    add_cond_list = get_windows(add_cond, self.opt.n_poses, window_step)
                
                
                out_motions = []
                for ii, [audio_emb, motions] in enumerate(zip(audio_emb_list, motions_list)):
                    print(f"Rank {rank}, Style: pid_{p_id_ori}: Name: {os.path.basename(aud_path)}, Video {i+1} / {len(aud_list)}, Clip {ii+1} / {len(audio_emb_list)} ")
                    if add_cond not in [None, {}]:
                        add_cond = add_cond_list[ii]
                    inpaint_dict = {}
                    if self.opt.overlap_len > 0:
                        inpaint_dict['gt'] = torch.zeros_like(motions)
                        inpaint_dict['outpainting_mask'] = torch.zeros_like(motions, dtype=torch.bool,
                                                        device=motions.device)  # Do outpainting/generation in those frames
                        
                        if ii == 0:
                            if self.opt.fix_very_first:
                                inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                                inpaint_dict['gt'][:, :self.opt.overlap_len, ...] = motions[:, -self.opt.overlap_len:, ...]
                            else:
                                pass
                        elif ii > 0:
                            inpaint_dict['outpainting_mask'][..., :self.opt.overlap_len, :] = True  # True means use gt motion 
                            inpaint_dict['gt'][:, :self.opt.overlap_len, ...] = outputs[:, -self.opt.overlap_len:, ...]

                    time_4 = time.time()
                    outputs = self.generate_batch(audio_emb, p_id, self.opt.net_dim_pose, add_cond, inpaint_dict)
                    time_end = time.time()
                    time_sum += time_end - time_4
                    outputs_np = outputs.cpu().numpy()
                    if ii == len(motions_list) - 1:
                        out_motions.append(outputs_np)
                    else:
                        out_motions.append(outputs_np[:, :window_step])
                    
                    if self.opt.debug:
                        break

                out_motions = np.concatenate(out_motions, 1)
                print(f"Time cost: {time_sum}; Frames: {out_motions.shape[1]}; FPS: {out_motions.shape[1] / time_sum}")
                

                out_motions = test_dataset.inv_standardize(out_motions, test_dataset.motion_mean, test_dataset.motion_std)
                
                np.save(pjoin(results_dir, f"{name.split('.')[0]}.npy"), out_motions)


        print("Finished")
        return results_dir


@torch.no_grad()
def get_hubert_from_16k_speech_long(hubert_model, wav2vec2_processor, speech, device="cuda:0"):
    hubert_model = hubert_model.to(device)
    # if speech.ndim ==2:
    #     speech = speech[:, 0] # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0) # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'