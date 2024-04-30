import argparse
import os
import torch
from mmcv.runner.dist_utils  import get_dist_info
import torch.distributed as dist


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument('--decomp_name', type=str, default="Decomp_SP001_SM001_H512", help='Name of autoencoder model')
        self.parser.add_argument('--model_base', type=str, default="transformer_encoder", choices=["transformer_decoder", "transformer_encoder", "st_unet"], help='Model architecture')
        self.parser.add_argument('--model_mean_type', type=str, default="epsilon", choices=["epsilon", "start_x", "previous_x"], help='Choose which type of data the model ouputs')
        self.parser.add_argument('--PE', type=str, default='pe_sinu', choices=['learnable', 'ppe_sinu', 'pe_sinu', 'pe_sinu_repeat', 'ppe_sinu_dropout'], help='Choose the type of positional emb')
        self.parser.add_argument("--ddim", action="store_true", help='Use ddim sampling')
        self.parser.add_argument("--timestep_respacing", type=str, default='ddim1000', help="Set ddim steps 'ddim{STEP}'")
        self.parser.add_argument("--cond_projection", type=str, default='mlp_includeX', choices=["linear_includeX", "mlp_includeX", "none", "linear_excludeX", "mlp_excludeX"], help="condition projection choices")
        self.parser.add_argument("--cond_residual", type=bool, default=True, help='Weather to use residual during condition projection')


        self.parser.add_argument("--gpu_id", type=int, default=None, help='GPU id')
        self.parser.add_argument("--distributed", action="store_true", help='Weather to use DDP training')
        self.parser.add_argument("--data_parallel", action="store_true", help='Weather to use DP training')
        self.parser.add_argument("--max_eval_samples", type=int, default=-1, help='max_eval_samples')
        self.parser.add_argument("--n_poses", type=int, help='number of poses for a training sequence')
        self.parser.add_argument("--axis_angle", type=bool, default=True, help='whether use the axis_angle rot representaiton')
        self.parser.add_argument("--rename", default=None, help='rename the experiment name during test')

        self.parser.add_argument("--debug", action="store_true", help='debug mode, only run one iteration')
        self.parser.add_argument('--mode', type=str, default='train', choices=["train", "val", "test", "test_arbitrary_len", "test_custom_audio"], help='train, val or test')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--data_mode', type=str, default='original', choices=['original', 'add_init_state'], help='Data modes')
        self.parser.add_argument('--data_type', type=str, default='pos', choices=['pos', 'vel', 'pos_vel'], help='Data types')
        self.parser.add_argument('--data_sel', type=str, default='upperbody', choices=['upperbody', 'all', 'upperbody_head', 'upperbody_hands'], help='Data selection')
        self.parser.add_argument('--data_root', type=str, default='./Freeform/processed_data_200', help='Dataset path')
        self.parser.add_argument('--beat_cache_name', default='beat_4english_15_141', help='Beat cache name')
        self.parser.add_argument('--use_aud_feat', type=str, default=None, choices=["interpolate", "conv"], help='Audio feature path')
        self.parser.add_argument('--audio_feat', type=str, default='mel', choices=["mel", "mfcc", "raw", "hubert", 'wav2vec2'], help='Audio feature type')
        self.parser.add_argument('--test_audio_path', type=str, default=None, help='test audio file or directory path')
        self.parser.add_argument('--same_overlap_noisy', action="store_true", help='During the outpainting process, use the same overlapping noisyGT')
        self.parser.add_argument('--no_repaint', action="store_true", help='Do not perform repaint during long-form generation')
        

        self.parser.add_argument('--vel_interval', type=int, default=10, help='Interval to compute the velocity')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--overlap_len', type=int, default=0, help='Fix the initial N frames for this clip')
        self.parser.add_argument('--addBlend', type=bool, default=True, help='Blend in the overlapping region at the last two denoise steps')
        self.parser.add_argument('--fix_very_first', action='store_true', help='Fix the very first {overlap_len} frames for this video to be the same as GT')
        self.parser.add_argument('--remove_audio', action='store_true', help='set audio to 0')
        self.parser.add_argument('--remove_style', action='store_true', help='set style to 0')
        self.parser.add_argument('--remove_hand', action='store_true', help='remove hand rotations from motion data')
        self.parser.add_argument('--no_fgd', action='store_true', help='do not compute fgd')
        self.parser.add_argument('--ablation',  type=str, default=None, choices=["no_x0", "no_detach", "reverse_ges2exp"],  help='ablation options')
        self.parser.add_argument('--rebuttal', type=str, default=None, choices=["noMelSpec", "noHuBert", "noMidAud"],  help='rebuttal ablation options')
        self.parser.add_argument('--visualize_unify_x0_step',  type=int, default=None, help='visualize expression x0 in unified mode every N step')


        self.parser.add_argument('--audio_dim', type=int, default=128, help='Input Audio feature dimension.')
        self.parser.add_argument('--audio_latent_dim', type=int, default=256, help='Audio latent dimension.')
        self.parser.add_argument('--style_dim', type=int, default=4, help='Input Style vector dimension. Can be one hot.')
        self.parser.add_argument('--dim_text_hidden', type=int, default=512, help='Dimension of hidden unit in text encoder')
        self.parser.add_argument('--dim_att_vec', type=int, default=512, help='Dimension of attention vector')
        self.parser.add_argument('--dim_z', type=int, default=128, help='Dimension of latent Gaussian vector')

        self.parser.add_argument('--n_layers_pri', type=int, default=1, help='Number of layers in prior network')
        self.parser.add_argument('--n_layers_pos', type=int, default=1, help='Number of layers in posterior network')
        self.parser.add_argument('--n_layers_dec', type=int, default=1, help='Number of layers in generator')

        self.parser.add_argument('--dim_pri_hidden', type=int, default=1024, help='Dimension of hidden unit in prior network')
        self.parser.add_argument('--dim_pos_hidden', type=int, default=1024, help='Dimension of hidden unit in posterior network')
        self.parser.add_argument('--dim_dec_hidden', type=int, default=1024, help='Dimension of hidden unit in generator')

        self.parser.add_argument('--dim_movement_enc_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(encoder)')
        self.parser.add_argument('--dim_movement_dec_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(decoder)')
        self.parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of motion snippet')

        self.parser.add_argument('--embed_net_path', type=str, default="feature_extractor/gesture_autoencoder_checkpoint_best.bin", help='embed_net_path')

        self.parser.add_argument('--fix_head_var', action="store_true", help='Make expression prediction derterministic')
        self.parser.add_argument('--expression_only', action="store_true", help='train epxression only')
        self.parser.add_argument('--gesture_only', action="store_true", help='train gesture only')
        self.parser.add_argument('--expCondition_gesture_only', type=str, choices=['gt', 'pred'], default=None, help='train gesture only, with expressions as condition')
        self.parser.add_argument('--gesCondition_expression_only', action="store_true", help='train expression only, with gesture as condition')
        self.parser.add_argument('--textExpEmoCondition_gesture_only', action="store_true", help='use all conditions: audio, text, emo, pid, facial')
        self.parser.add_argument('--addTextCond', action="store_true", help='add Text feature to audio feature')
        self.parser.add_argument('--addEmoCond', action="store_true", help='add Emo feature to audio feature')
        self.parser.add_argument('--expAddHubert', action="store_true", help='concat Hubert feature to encoded audio feature only for expression generation')
        self.parser.add_argument('--addHubert', type=bool, default=True, help='concat Hubert feature to encoded audio feature for both expression and gesture generation')
        self.parser.add_argument('--addWav2Vec2', action="store_true", help='concat Wav2Vec2 feature to encoded audio feature for both expression and gesture generation')
        self.parser.add_argument('--encode_wav2vec2', action="store_true", help='encode the wav2vec2 feature')
        self.parser.add_argument('--encode_hubert', type=bool, default=True, help='encode the hubert feature')
        self.parser.add_argument('--separate', type=str, choices=['v1', 'v2'], default=None, help='limit information exchange between expression and gestures, v1 share encoder, v2 two independent encoders')
        self.parser.add_argument('--usePredExpr', type=str, default=None, help='Path to the predicted expressions.')
        self.parser.add_argument('--unidiffuser', type=bool, default=True, help='Use the unified framework for joint expression and gesture generation')

        self.parser.add_argument('--separate_pure', action="store_true", help='pure two encoders')

        # classifier-free guidance
        self.parser.add_argument('--classifier_free', action="store_true", help='Use classifier-free guidance')
        self.parser.add_argument('--null_cond_prob', type=float, default=0.2, help='Probability of null condition during classifier-free training')
        self.parser.add_argument('--cond_scale', type=float, default=1.0, help='Scale of the condition in classifier-free guidance sampling')

        # Try Expression ID off
        self.parser.add_argument('--ExprID_off', action="store_true", help='Turn off the expression ID condition')
        self.parser.add_argument('--ExprID_off_uncond', action="store_true", help='Turn off the expression ID condition under the classifier-free uncondition part of training')


        self.parser.add_argument('--use_joints', action="store_true", help='Whether convert to joints if using TED 3D dataset')
        self.parser.add_argument('--use_single_style', action="store_true", help='Whether to use single style')
        self.parser.add_argument('--test_on_trainset', action="store_true", help='Whether to test on training set')
        self.parser.add_argument('--test_on_val', action="store_true", help='Whether to test on validation set')
        self.parser.add_argument('--output_gt', action="store_true", help='Directly output GT during test')
        self.parser.add_argument('--no_style', action="store_true", help='Do not use style vectors')
        self.parser.add_argument('--no_resample', action="store_true", help='Do not use resample during inpainting based sampling')
        self.parser.add_argument('--add_vel_loss', type=bool, default=True, help='Add velocity loss')
        self.parser.add_argument('--vel_loss_start', type=int, default=-1, help='velocity loss and huber loss start epoch')
        self.parser.add_argument('--expr_weight', type=int, default=1, help='expression weight')
        
        # inference
        self.parser.add_argument('--jump_n_sample', type=int, default=5, help='hyperparameter for resampling')
        self.parser.add_argument('--jump_length', type=int, default=3, help='hyperparameter for resampling')
        

        ## Distributed
        self.parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
        self.parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        self.parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
        self.parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        self.parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
        self.parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train

        args = vars(self.opt)

        if self.opt.rank == 0:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
            if self.is_train:
                # save to the disk
                expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
                if not os.path.exists(expr_dir):
                    os.makedirs(expr_dir)
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        if self.opt.world_size > 1:
            dist.barrier()
        return self.opt
