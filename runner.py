import numpy
import os
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
# from utils.plot_script import *

from models import MotionTransformer, UniDiffuser
from trainers import DDPMTrainer_beat, DDPMTrainer_talkshow
from datasets import ShowDataset

from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import sys
sys.path.append(os.path.join(sys.path[2], "A_TalkSHOW_ori"))



def build_models(opt, dim_pose, audio_dim=128, audio_latent_dim=256, style_dim=4):
    if opt.unidiffuser:
        encoder = UniDiffuser(
            opt=opt,
            input_feats=dim_pose,
            audio_dim=audio_dim,
            aud_latent_dim=audio_latent_dim,
            style_dim=style_dim,
            num_frames=opt.n_poses,
            num_layers=opt.num_layers,
            latent_dim=opt.latent_dim,
            no_clip=opt.no_clip,
            no_eff=opt.no_eff,
            pe_type=opt.PE)
    else:
        encoder = MotionTransformer(
            opt=opt,
            input_feats=dim_pose,
            audio_dim=audio_dim,
            style_dim=style_dim,
            num_frames=opt.n_poses,
            num_layers=opt.num_layers,
            latent_dim=opt.latent_dim,
            no_clip=opt.no_clip,
            no_eff=opt.no_eff,
            pe_type=opt.PE)
    return encoder

def build_fgd_val_model(opt):
    eval_model_module = __import__(f"models.motion_autoencoder", fromlist=["something"])
    eval_model = getattr(eval_model_module, 'HalfEmbeddingNet')(opt)

    print(f"init 'HalfEmbeddingNet' success")
    return eval_model

def main():
    parser = TrainCompOptions()
    opt = parser.parse()

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu_id, ngpus_per_node, opt)



def main_worker(gpu_id, ngpus_per_node, opt):
    # rank, world_size = get_dist_info()
    opt.gpu_id = gpu_id

    if opt.gpu_id is not None:
        print("Use GPU: {}".format(opt.gpu_id))

    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu_id
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)


    opt.device = torch.device("cuda")
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    # if opt.rank == 0:
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    if opt.world_size > 1:
        dist.barrier()
            
    elif opt.dataset_name.lower() == 'beat':
        opt.data_root = 'data/BEAT'
        opt.fps = 15
        opt.net_dim_pose = 192 # body: [16, 34, 141], expression: [16, 34, 51], in_audio: [16, 36266]
        opt.split_pos = 141
        opt.dim_pose = 141
        if opt.remove_hand:
            opt.dim_pose = 33
        opt.expression_dim = 51 

        if opt.expression_only or opt.gesCondition_expression_only:
            opt.net_dim_pose = opt.expression_dim # expression
            opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/face_300.bin'
        elif opt.gesture_only or opt.expCondition_gesture_only != None or \
                opt.textExpEmoCondition_gesture_only:
            opt.net_dim_pose = opt.dim_pose # gesture
            if opt.axis_angle:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ges_axis_angle_300.bin'
            else:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/ae_300.bin'
        else:
            opt.net_dim_pose = opt.dim_pose + opt.expression_dim # gesture + expression
            if opt.axis_angle:
                opt.e_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/weights/GesAxisAngle_Face_300.bin'
            else:
                raise NotImplementedError
        
        opt.audio_dim = 128
        if opt.use_aud_feat:
            opt.audio_dim = 1024
        opt.style_dim = 30 # totally 30 subjects
        opt.speaker_dim = 30 
        opt.word_index_num = 5793
        opt.word_dims = 300
        opt.word_f = 128
        opt.emotion_f = 8
        opt.emotion_dims = 8
        opt.freeze_wordembed = False
        opt.hidden_size = 256
        opt.n_layer = 4

        if opt.n_poses == 150:
            opt.stride = 50
        elif opt.n_poses == 34: 
            opt.stride = 10
        opt.pose_fps = 15
        opt.vae_length = 300
        opt.new_cache = False
        opt.audio_norm = False
        opt.facial_norm = True
        opt.pose_norm = True
        opt.train_data_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/train/'
        opt.val_data_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/val/'
        opt.test_data_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/test/'
        opt.mean_pose_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/train/'
        opt.std_pose_path = f'data/BEAT/beat_cache/{opt.beat_cache_name}/train/'
        opt.multi_length_training = [1.0]
        opt.audio_rep = 'wave16k'
        opt.facial_rep = 'facial52'
        opt.speaker_id = 'id'
        opt.pose_rep = 'bvh_rot'
        opt.word_rep = 'text'
        opt.sem_rep = 'sem'
        opt.emo_rep = 'emo'

    elif opt.dataset_name.lower() == 'talkshow':
        opt.talkshow_config = 'options/talkshow_configs/body_pixel.json'
        opt.speaker_dim = 4
        opt.fps = 30
        opt.dim_pose = 129
        opt.split_pos = 129
        if opt.remove_hand:
            opt.dim_pose = 39
        opt.expression_dim = 103
        if opt.ablation == "reverse_ges2exp":
            opt.expression_dim, opt.dim_pose = opt.dim_pose, opt.expression_dim
        if opt.expression_only or opt.gesCondition_expression_only:
            opt.net_dim_pose = opt.expression_dim # expression
            opt.e_path = f'data/SHOW/ae_weights/expression.pth.tar'
        elif opt.gesture_only or opt.expCondition_gesture_only != None:
            opt.net_dim_pose = opt.dim_pose # gesture
            opt.e_path = f'data/SHOW/ae_weights/gesture.pth.tar'
        else:
            opt.net_dim_pose = opt.dim_pose + opt.expression_dim # gesture + expression
            opt.e_path = f'data/SHOW/ae_weights/gesture_expression.pth.tar'
        
        if opt.audio_feat == 'mfcc':
            opt.audio_dim = 64
        elif opt.audio_feat == 'mel':
            opt.audio_dim = 128
        elif opt.audio_feat == 'raw':
            opt.audio_dim = 1
        elif opt.audio_feat == 'hubert':
            opt.audio_dim = 1024
        opt.style_dim = 4
        opt.speaker_dim = 4 
        opt.n_poses = 88
        opt.pose_fps = 30
        opt.vae_length = 300
    
    else:
        raise KeyError('Dataset Does Not Exist')



    print("=> creating model '{}'".format(opt.model_base))
    model = build_models(opt, opt.net_dim_pose, opt.audio_dim, opt.audio_latent_dim, opt.style_dim)

    if opt.no_fgd == False:
        eval_model = build_fgd_val_model(opt)
    else:
        eval_model = None

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if opt.gpu_id is not None:
                torch.cuda.set_device(opt.gpu_id)
                model.cuda(opt.gpu_id)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                opt.batch_size = int(opt.batch_size / ngpus_per_node)
                opt.workers = int((opt.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu_id], find_unused_parameters=True)

                if not opt.no_fgd:
                    eval_model.cuda(opt.gpu_id)
                    eval_model = torch.nn.parallel.DistributedDataParallel(eval_model, device_ids=[opt.gpu_id], find_unused_parameters=False)
            else:
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
                model = torch.nn.parallel.DistributedDataParallel(model)
                if not opt.no_fgd:
                    # eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(eval_model)  
                    eval_model = torch.nn.parallel.DistributedDataParallel(eval_model, device_ids=[opt.rank], broadcast_buffers=True, find_unused_parameters=False).to(opt.rank)
    elif opt.gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_id)
        model = model.cuda(opt.gpu_id)
        if not opt.no_fgd:
            eval_model = eval_model.cuda(opt.gpu_id)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        if not opt.no_fgd:
            eval_model = eval_model.to(device)
    else:
        # Use single gpu
        model = model.cuda()
        if not opt.no_fgd:
            eval_model = eval_model.cuda()

    if torch.cuda.is_available():
        if opt.gpu_id:
            device = torch.device('cuda:{}'.format(opt.gpu_id))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if opt.dataset_name == 'beat':
        runner = DDPMTrainer_beat(opt, model, eval_model=eval_model)
    elif opt.dataset_name == 'talkshow':
        runner = DDPMTrainer_talkshow(opt, model, eval_model=eval_model)
    else:
        runner = DDPMTrainer(opt, model)

    if opt.mode == "train":
        if opt.dataset_name.lower() == 'beat':
            train_dataset = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"]).BeatDataset(opt, "train")  
            val_dataset = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"]).BeatDataset(opt, "val")
        
        elif opt.dataset_name.lower() == 'talkshow':
            train_dataset = ShowDataset(opt, 'data/SHOW/cached_data/talkshow_train_cache')
            val_dataset = ShowDataset(opt, 'data/SHOW/cached_data/talkshow_val_cache')


        runner.train(train_dataset, val_dataset)

    elif "test" in opt.mode:
        if opt.dataset_name.lower() == 'beat':
            if opt.test_on_trainset:
                test_dataset = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"]).BeatDataset(opt, "train")
            elif opt.test_on_val:
                test_dataset = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"]).BeatDataset(opt, "val")
            else:
                test_dataset = __import__(f"datasets.{opt.dataset_name}", fromlist=["something"]).BeatDataset(opt, "test")

        elif opt.dataset_name.lower() == 'talkshow':
            if opt.test_on_trainset:
                test_dataset = ShowDataset(opt, 'data/SHOW/cached_data/talkshow_train_cache')
            elif opt.test_on_val:
                test_dataset = ShowDataset(opt, 'data/SHOW/cached_data/talkshow_val_cache')
            else:
                test_dataset = ShowDataset(opt, 'data/SHOW/cached_data/talkshow_test_cache')
                
        if opt.mode == "test":
            results_dir = runner.test(test_dataset)
        elif opt.mode == "test_arbitrary_len":
            opt.batch_size = 1
            results_dir = runner.test_arbitrary_len(test_dataset)
        elif opt.mode == "test_custom_audio":
            results_dir = runner.test_custom_aud(opt.test_audio_path, test_dataset)
        print(results_dir)




if __name__ == '__main__':
    main()
    
