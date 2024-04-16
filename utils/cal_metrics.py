import os, sys
module_path = os.path.abspath(".")
if module_path not in sys.path:
    sys.path.append(module_path)

from math import nan
from glow.config import JsonConfig
from glow.builder import build
from torch.utils.data import DataLoader
from motion import *
import datetime, time
import numpy as np
import torch
import torch.nn.functional as F
import umap
from scipy import linalg
from glow.embedding_net import EmbeddingNet
import matplotlib.pyplot as plt
import subprocess
from textwrap import wrap
import matplotlib.animation as animation

import torch

mean_data = np.array(
        [0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039,
         -0.9236511, 0.3061306, -0.0012415, -0.5155854, 0.8129665, 0.0871897, 0.2348464, 0.1846561, 0.8091402,
         0.9271948, 0.2960011, -0.013189, 0.5233978, 0.8092403, 0.0725451, -0.2037076, 0.1924306, 0.8196916])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def create_video_and_save(save_path, target, output, mean_data, title,
                          audio_path=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True):
    dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                     (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length
    print('rendering a video...')
    start = time.time()

    fig = plt.figure(figsize=(8, 4))
    axes = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]
    axes[0].view_init(elev=20, azim=-60)
    axes[1].view_init(elev=20, azim=-60)
    fig_title = title

    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

    # un-normalization and convert to poses
    mean_data = mean_data.flatten()
    output = output + mean_data
    output_poses = convert_dir_vec_to_pose(output)
    target_poses = None
    if target is not None:
        target = target + mean_data
        target_poses = convert_dir_vec_to_pose(target)

    def animate(i):
        for k, name in enumerate(['human', 'generated']):
            if name == 'human' and target is not None and i < len(target):
                pose = target_poses[i]
            elif name == 'generated' and i < len(output):
                pose = output_poses[i]
            else:
                pose = None

            if pose is not None:
                axes[k].clear()
                for j, pair in enumerate(dir_vec_pairs):
                    axes[k].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                 [pose[pair[0], 2], pose[pair[1], 2]],
                                 [pose[pair[0], 1], pose[pair[1], 1]],
                                 zdir='z', linewidth=5)
                axes[k].set_xlim3d(-0.5, 0.5)
                axes[k].set_ylim3d(0.5, -0.5)
                axes[k].set_zlim3d(0.5, -0.5)
                axes[k].set_xlabel('x')
                axes[k].set_ylabel('z')
                axes[k].set_zlabel('y')
                axes[k].set_title('{} ({}/{})'.format(name, i + 1, len(output)))

    if target is not None:
        num_frames = max(len(target), len(output))
    else:
        num_frames = len(output)
    ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

    try:
        video_path = '{}/temp.mp4'.format(save_path)
        ani.save(video_path, fps=15, dpi=150)  # dpi 150 for a higher resolution
        del ani
        plt.close(fig)
    except RuntimeError:
        assert False, 'RuntimeError'

    # merge audio and video
    if audio_path is not None:
        merged_video_path = '{}/temp_with_audio.mp4'.format(save_path)
        cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2',
               merged_video_path]
        if clipping_to_shortest_stream:
            cmd.insert(len(cmd) - 1, '-shortest')
        subprocess.call(cmd)
        if delete_audio_file:
            os.remove(audio_path)
        os.remove(video_path)

    print('done, took {:.1f} seconds'.format(time.time() - start))
    return output_poses, target_poses


def prepare_cond(jt_data, ctrl_data, device):
    nn, seqlen, n_feats = jt_data.shape

    jt_data = jt_data.reshape((nn, seqlen * n_feats))
    nn, seqlen, n_feats = ctrl_data.shape
    ctrl_data = ctrl_data.reshape((nn, seqlen * n_feats))
    cond = torch.from_numpy(np.expand_dims(np.concatenate((jt_data, ctrl_data), axis=1), axis=-1))
    return cond.to(device)

def convert_dir_vec_to_pose(vec):
    dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                     (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def generate_sample(graph, batch, hparams, device, eps_std=1.0):
        for k in batch:
            if k == "style":
                continue
            batch[k] = batch[k].to(device)

        graph = graph.eval()

        autoreg_all = batch["autoreg"].cpu().numpy()
        control_all = batch["control"].cpu().numpy()

        seqlen = hparams.Data.seqlen
        n_lookahead = hparams.Data.n_lookahead

        clip_length = 1
        # Initialize the lstm hidden state
        if hasattr(graph, "module"):
            graph.module.init_lstm_hidden()
        else:
            graph.init_lstm_hidden()

        nn, n_timesteps, n_feats = autoreg_all.shape
        sampled_all = np.zeros((nn, n_timesteps - n_lookahead, n_feats))
        autoreg = np.zeros((nn, seqlen, n_feats), dtype=np.float32)  # initialize from a mean pose
        autoreg[:, :seqlen, :] = autoreg_all[:, :seqlen, :] # seed pose
        sampled_all[:, :seqlen, :] = autoreg


        # Loop through control sequence and generate new data
        with torch.no_grad():
            for i in range(0, control_all.shape[1] - seqlen - n_lookahead, clip_length):
                control = control_all[:, i:(i + seqlen + n_lookahead + clip_length), :]

                # prepare conditioning for moglow (control + previous poses)
                cond = prepare_cond(autoreg.copy(), control.copy(), device)

                # sample from Moglow
                sampled = graph(z=None, cond=cond, eps_std=eps_std, reverse=True, z_shape=[nn, 27, 1])
                sampled = sampled.cpu().numpy()[:, :, 0].reshape(sampled.shape[0], clip_length, -1)

                # store the sampled frame
                sampled_all[:, (i + seqlen): (i + seqlen + clip_length), :] = sampled

                # update saved pose sequence
                autoreg = np.concatenate((autoreg[:, clip_length:, :].copy(), sampled[:, :, :]), axis=1)

        return sampled_all[:, :, :], autoreg_all # 42 frames

def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))

    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled

def evaluate_testset(data_motion, test_data_loader, generator, embed_space_evaluator, hparams, device):
    # to evaluation mode
    print('*' * 30)
    print('evaluating metrics ...')
    generator.train(False)

    if embed_space_evaluator:
        embed_space_evaluator.reset()
    joint_mae = AverageMeter('mae_on_joint')
    accel = AverageMeter('accel')
    pck = AverageMeter('pck')
    diversity = AverageMeter('diversity')

    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            tot_len = len(test_data_loader)
            seqlen = hparams.Data.seqlen
            out_dir_vec, target = generate_sample(generator, data, hparams, device)
            out_dir_vec, target = out_dir_vec[:,-34:,:], target[:,-34:,:] # 34 frames
            out_dir_vec = inv_standardize(out_dir_vec, data_motion.scaler)
            target = inv_standardize(target, data_motion.scaler)
            out_dir_vec = torch.from_numpy(out_dir_vec).float().to(device)
            target = torch.from_numpy(target).float().to(device)

            out_origin, target_origin = out_dir_vec, target

            # calculate MAE of joint coordinates
            out_dir_vec = out_dir_vec.cpu().numpy()
            out_dir_vec += mean_data
            out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
            target = target.cpu().numpy()
            target += mean_data
            target_poses = convert_dir_vec_to_pose(target)

            B, seq, _, _ = out_joint_poses.shape
            batch_size = B
            div_val = 0.0
            for i in range(0, B):
                for j in range(i+1, B):
                    dif = out_joint_poses[i,:,:,:] - out_joint_poses[j,:,:,:]
                    div_val += np.mean(np.absolute(dif))
            div_val = div_val * 2 / (B * (B - 1))

            diff = out_joint_poses - target_poses
            correct = np.sum(diff ** 2, axis=3)
            correct = np.sqrt(correct) < 0.1
            pck_val = np.mean(correct)
            mae_val = np.mean(np.absolute(diff))     

            # accel
            target_acc = np.diff(target_poses, n=2, axis=1)
            out_acc = np.diff(out_joint_poses, n=2, axis=1)
            acc_val = np.mean(np.abs(target_acc - out_acc))

            joint_mae.update(mae_val, batch_size)
            accel.update(acc_val, batch_size)
            pck.update(pck_val, batch_size)
            diversity.update(div_val, batch_size)

            if embed_space_evaluator:
                embed_space_evaluator.push_samples(out_origin, target_origin)

            print(iter_idx + 1, ' / ', tot_len)
            
    # back to training mode
    generator.train(True)

    # print
    ret_dict = {'joint_mae': joint_mae.avg}
    elapsed_time = time.time() - start
    if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
        frechet_dist, feat_dist = embed_space_evaluator.get_scores()
        print(
            'joint mae: {:.5f}, accel diff: {:.5f}, FGD: {:.3f}, feat_D: {:.3f}, PCK: {:.3f}, div: {:.5f} / {:.1f}s'.format(
                joint_mae.avg, accel.avg, frechet_dist, feat_dist, pck.avg, diversity.avg, elapsed_time))

        ret_dict['frechet'] = frechet_dist
        ret_dict['feat_dist'] = feat_dist

    print('*' * 30)
    return frechet_dist


class EmbeddingSpaceEvaluator:
    def __init__(self, hparams, embed_net_path, device):
        # init embed net
        ckpt = torch.load(embed_net_path, map_location=device)
        mode = 'pose'
        self.pose_dim = 27
        self.net = EmbeddingNet(self.pose_dim, 34, mode).to(device)
        self.net.load_state_dict(ckpt['gen_dict'])
        self.net.train(False)

        # storage
        self.context_feat_list = []
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []

    def reset(self):
        self.context_feat_list = []
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []

    def get_no_of_samples(self):
        return len(self.real_feat_list)

    def push_samples(self, generated_poses, real_poses):
        # convert poses to latent features
        real_feat, _, _ = self.net(real_poses, 'pose', variational_encoding=False)
        generated_feat, _, _ = self.net(generated_poses, 'pose', variational_encoding=False)

        self.real_feat_list.append(real_feat.data.cpu().numpy())
        self.generated_feat_list.append(generated_feat.data.cpu().numpy())

    def get_features_for_viz(self):
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)

        transformed_feats = umap.UMAP().fit_transform(np.vstack((generated_feats, real_feats)))
        n = int(transformed_feats.shape[0] / 2)
        generated_feats = transformed_feats[0:n, :]
        real_feats = transformed_feats[n:, :]

        return real_feats, generated_feats

    def get_scores(self):
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)

        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)
            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10
            return frechet_dist

        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


def eval_FGD():
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                 .replace(":", "")\
                                 .replace(" ", "_")
    hparams_motion = "hparams/preferred/locomotion_test.json"  # args["<hparams>"]
    dataset_motion = "locomotion"  # args["<dataset>"]
    hparams_motion = JsonConfig(hparams_motion)
    dataset_motion = Datasets[dataset_motion]

    log_dir = os.path.join(hparams_motion.Dir.log_root, "log_" + date)
    print("log_dir:" + str(log_dir))
    checkpoint_dir = hparams_motion.Infer.checkpoint_dir

    gpu = hparams_motion.Device.data
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    data_motion = dataset_motion(hparams_motion, False, log_path=checkpoint_dir)
    x_channels_motion, cond_channels_motion = data_motion.n_channels()

    built_motion = build(x_channels_motion, cond_channels_motion, None, None, hparams_motion, False)

    eval_net_path = 'feature_extractor/gesture_autoencoder_checkpoint_best.bin'

    test_data_loader = DataLoader(data_motion.get_test_dataset(),
                                           batch_size=50,
                                           shuffle=False,
                                           drop_last=True)

    embed_space_evaluator = EmbeddingSpaceEvaluator(hparams_motion, eval_net_path, device)

    evaluate_testset(data_motion, test_data_loader, built_motion['graph'], embed_space_evaluator, hparams_motion, device)
    print("finish!")

def train_eval_FGD(eval_loader, graph, dataset, step):

    eval_net_path = 'feature_extractor/gesture_autoencoder_checkpoint_best.bin'
    hparams_motion = "hparams/preferred/locomotion_test.json"
    hparams_motion = JsonConfig(hparams_motion)

    train_hparams = "hparams/preferred/locomotion.json"
    train_hparams = JsonConfig(train_hparams)
    gpu = train_hparams.Device.data
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    
    embed_space_evaluator = EmbeddingSpaceEvaluator(hparams_motion, eval_net_path, device)

    return evaluate_testset(dataset, eval_loader, graph, embed_space_evaluator, hparams_motion, device)

if __name__ == "__main__":
    eval_FGD()

