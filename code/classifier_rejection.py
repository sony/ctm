"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time
import copy
import numpy as np
import torch as th
import torch.distributed as dist
import glob
import torch.nn.functional as F
from tqdm.auto import tqdm
import pickle

from cm import dist_util, logger
from cm.script_util import (
    train_defaults,
    model_and_diffusion_defaults,
    cm_train_defaults,
    ctm_train_defaults,
    ctm_eval_defaults,
    ctm_loss_defaults,
    ctm_data_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_classifier,

)


class vpsde():
    def __init__(self, beta_min=0.1, beta_max=20., multiplier=1., cos_t_classifier=False,):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.multiplier = multiplier
        self.a = (self.beta_1 ** 0.5 - self.beta_0 ** 0.5) ** 2 / 3.
        self.b = (self.beta_0 ** 0.5) * (self.beta_1 ** 0.5 - self.beta_0 ** 0.5)
        self.c = self.beta_0
        self.s = 0.008
        self.f_0 = np.cos(self.s / (1. + self.s) * np.pi / 2.) ** 2
        self.cos_t_classifier = cos_t_classifier

    @property
    def T(self):
        return 1

    def compute_tau(self, std_wve_t, multiplier=-1.):
        if multiplier == -1:
            if self.multiplier == 1.:
                tau = -self.beta_0 + th.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * th.log(1. + std_wve_t ** 2))
                tau /= self.beta_1 - self.beta_0
            elif self.multiplier == 2.:
                d = - th.log(1. + std_wve_t ** 2)
                in_ = (2 * (self.b ** 3) - 9 * self.a * self.b * self.c + 27. * (self.a ** 2) * d) ** 2 - 4 * (((self.b ** 2) - 3 * self.a * self.c) ** 3)
                out_ = 2 * (self.b ** 3) - 9 * self.a * self.b * self.c + 27. * (self.a ** 2) * d
                plus = (out_ + in_ ** 0.5)
                minus = (out_ - in_ ** 0.5)
                sign_plus = th.sign(plus)
                sign_minus = th.sign(minus)
                tau = - self.b / (3. * self.a) - sign_plus * ((th.abs(plus) / 2.) ** (1/3.)) / (3. * self.a) - sign_minus * ((th.abs(minus) / 2.) ** (1/3.)) / (3. * self.a)
        elif multiplier == 1.:
            tau = -self.beta_0 + th.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * th.log(1. + std_wve_t ** 2))
            tau /= self.beta_1 - self.beta_0
        elif multiplier == 2.:
            d = - th.log(1. + std_wve_t ** 2)
            in_ = (2 * (self.b ** 3) - 9 * self.a * self.b * self.c + 27. * (self.a ** 2) * d) ** 2 - 4 * (
                        ((self.b ** 2) - 3 * self.a * self.c) ** 3)
            out_ = 2 * (self.b ** 3) - 9 * self.a * self.b * self.c + 27. * (self.a ** 2) * d
            plus = (out_ + in_ ** 0.5)
            minus = (out_ - in_ ** 0.5)
            sign_plus = th.sign(plus)
            sign_minus = th.sign(minus)
            tau = - self.b / (3. * self.a) - sign_plus * ((th.abs(plus) / 2.) ** (1 / 3.)) / (
                        3. * self.a) - sign_minus * ((th.abs(minus) / 2.) ** (1 / 3.)) / (3. * self.a)
        return tau

    def marginal_prob(self, t, multiplier=-1.):
        log_mean_coeff = - 0.5 * self.integral_beta(t, multiplier)
        #log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = th.exp(log_mean_coeff)
        std = th.sqrt(1. - th.exp(2. * log_mean_coeff))
        return mean, std

    def transform_normalized_vp_to_unnormalized_wve(self, t, multiplier=-1.):
        mean, std = self.marginal_prob(t, multiplier=multiplier)
        return std / mean

    def sampling_std(self, num_step):
        #c = 1000 // num_step
        assert 1000 % num_step == 0
        ddim_timesteps = th.from_numpy(np.array(list(range(0, 1000, 1000 // num_step)))[::-1].copy())
        print(ddim_timesteps)
        steps_out = ddim_timesteps + 1
        std = self.transform_normalized_vp_to_unnormalized_wve(steps_out / 1000.)
        print(std)
        return std

    def transform_unnormalized_wve_to_normalized_vp(self, t, std_out=False, multiplier=-1.):
        tau = self.compute_tau(t, multiplier=multiplier)
        mean_vp_tau, std_vp_tau = self.marginal_prob(tau, multiplier=multiplier)
        #print("tau before: ", tau)
        if self.cos_t_classifier:
            tau = self.compute_t_cos_from_t_lin(tau)
        #print("tau after: ", tau)
        if std_out:
            return mean_vp_tau, std_vp_tau, tau
        return mean_vp_tau, tau

    def from_rescaled_t_to_original_std(self, rescaled_t):
        return th.exp(rescaled_t / 250.) - 1e-44

    def compute_t_cos_from_t_lin(self, t_lin):
        sqrt_alpha_t_bar = th.exp(-0.25 * t_lin ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t_lin * self.beta_0)
        time = th.arccos(np.sqrt(self.f_0) * sqrt_alpha_t_bar)
        t_cos = self.T * ((1. + self.s) * 2. / np.pi * time - self.s)
        return t_cos

    def get_diffusion_time(self, batch_size, batch_device, t_min=1e-5, importance_sampling=True):
        if importance_sampling:
            Z = self.normalizing_constant(t_min)
            u = th.rand(batch_size, device=batch_device)
            return (-self.beta_0 + th.sqrt(self.beta_0 ** 2 + 2 * (self.beta_1 - self.beta_0) *
                    th.log(1. + th.exp(Z * u + self.antiderivative(t_min))))) / (self.beta_1 - self.beta_0), Z.detach()
        else:
            return th.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

    def antiderivative(self, t, stabilizing_constant=0.):
        if isinstance(t, float) or isinstance(t, int):
            t = th.tensor(t).float()
        return th.log(1. - th.exp(- self.integral_beta(t)) + stabilizing_constant) + self.integral_beta(t)

    def normalizing_constant(self, t_min):
        return self.antiderivative(self.T) - self.antiderivative(t_min)

    def integral_beta(self, t, multiplier=-1.):
        if multiplier == -1.:
            if self.multiplier == 1.:
                return 0.5 * t ** 2 * (self.beta_1 - self.beta_0) + t * self.beta_0
            elif self.multiplier == 2.:
                return ((self.beta_1 ** 0.5 - self.beta_0 ** 0.5) ** 2) * (t ** 3) / 3. \
                      + (self.beta_0 ** 0.5) * (self.beta_1 ** 0.5 - self.beta_0 ** 0.5) * (t ** 2) + self.beta_0 * t
        elif multiplier == 1.:
            return 0.5 * t ** 2 * (self.beta_1 - self.beta_0) + t * self.beta_0
        elif multiplier == 2.:
            return ((self.beta_1 ** 0.5 - self.beta_0 ** 0.5) ** 2) * (t ** 3) / 3. \
                + (self.beta_0 ** 0.5) * (self.beta_1 ** 0.5 - self.beta_0 ** 0.5) * (t ** 2) + self.beta_0 * t

def get_classifier_guidance(classifier, vpsde, unnormalized_input, std_wve_t, img_resolution, class_labels, log_prob=False):
    mean_vp_tau, std_vp_tau, tau = vpsde.transform_unnormalized_wve_to_normalized_vp(std_wve_t, std_out=True) ## VP pretrained classifier
    input = mean_vp_tau[:,None,None,None] * unnormalized_input
    with th.no_grad():
        x_ = input.float().clone().detach().requires_grad_()
        tau = th.ones(input.shape[0], device=tau.device) * tau.reshape(-1)
        logits = classifier(x_, timesteps=tau)#, condition=class_labels)
        #print("tau: ", tau)
        #print("log probability: ", torch.softmax(logits, 1)[:,281].log())
        #print("log probability mean: ", torch.softmax(logits, 1)[:,281].log().mean())
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), class_labels.view(-1)]
        if log_prob:
            return selected
        classifier_guidance = th.autograd.grad(outputs=selected.sum(), inputs=x_, retain_graph=False)[0]
        classifier_guidance *= mean_vp_tau[:,None,None,None]
    return classifier_guidance

def main():
    args = create_argparser().parse_args()

    if args.use_MPI:
        dist_util.setup_dist(args.device_id)
    else:
        dist_util.setup_dist_without_MPI(args.device_id)

    logger.configure(args, dir=args.out_dir)

    logger.log("creating model and diffusion...")

    classifier = create_classifier(**args_to_dict(args, list(classifier_defaults().keys()) + ['image_size']))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    vpsde_ = vpsde(cos_t_classifier=(args.image_size == 64))

    num_samples = int(50000 * (1. + args.rejection_ratio))
    if os.path.exists(os.path.join(args.out_dir, 'classifier_rejection_v3.pickle')):
        with open(os.path.join(args.out_dir, 'classifier_rejection_v3.pickle'), 'rb') as handle:
            b = pickle.load(handle)
            log_probs_all = b['log_probs_all']
            log_probs = b['log_probs']
            all_classes = b['all_classes']
            sample_dirs = b['sample_dirs']
    else:
        sample_dirs = glob.glob(os.path.join(args.out_dir, 'ctm_exact_sampler_1_steps_028000_itrs_0.999_ema_/*.npz'))
        log_probs = []
        all_classes = []
        for k in tqdm(range(len(sample_dirs))):
            samples = np.load(sample_dirs[k])['arr_0']
            samples = th.from_numpy(samples).to(dist_util.dev()).permute(0,3,1,2)
            samples = samples / 127.5 - 1.
            s_in = samples.new_ones([samples.shape[0]])
            classes = np.load(sample_dirs[k])['arr_1']
            classes_th = th.from_numpy(classes).to(dist_util.dev())
            log_prob = get_classifier_guidance(classifier, vpsde_, samples, 0.002 * s_in, args.image_size, classes_th, log_prob=True).detach().cpu().numpy().reshape(-1)
            log_probs.extend(log_prob)
            all_classes.extend(classes)
        log_probs_all = {i: {} for i in range(1000)}
        for k in range(1000):
            log_probs_all[k] = {i: log_probs[i] for i in range(len(log_probs)) if all_classes[i] == k}


        with open(os.path.join(args.out_dir, 'classifier_rejection_v3.pickle'), 'wb') as handle:
            pickle.dump({'log_probs_all': log_probs_all, 'log_probs': log_probs, 'all_classes': all_classes, 'sample_dirs': sample_dirs}, handle)
        import sys
        sys.exit()

    samples_per_class = int((1. + args.rejection_ratio) * 50)
    top_50k = []
    for k in range(1000):
        top_50k.extend(
            list(dict(sorted(dict(list(log_probs_all[k].items())[:samples_per_class]).items(), key=lambda item: item[1], reverse=True)).keys())[:50])

    num = 0
    samples_top = []

    for sample_dir in sample_dirs:
        samples = np.load(sample_dir)['arr_0']
        idx = np.array([i for i in top_50k if i >= num and i < num + samples.shape[0]])
        if len(idx) > 0:
            samples_top.append(samples[idx - num])
        num += samples.shape[0]
    samples_top = np.concatenate(samples_top)#.tolist()
    print(len(samples_top))
    n = 1
    while len(samples_top) < 50000:
        sample_dir = sample_dirs[-n]
        samples = np.load(sample_dir)['arr_0']
        samples_top = np.concatenate((samples_top, samples))
        print(len(samples_top))
        n += 1

    samples_top = np.array(samples_top[:50000])
    np.random.shuffle(samples_top)
    np.random.shuffle(samples_top)
    np.random.shuffle(samples_top)
    np.random.shuffle(samples_top)
    np.random.shuffle(samples_top)
    assert len(samples_top) == 50000
    os.makedirs(os.path.join(args.out_dir, f'ctm_exact_sampler_1_steps_028000_itrs_0.999_ema_single_npz_{args.rejection_ratio}_'), exist_ok=True)
    np.savez(os.path.join(args.out_dir, f'ctm_exact_sampler_1_steps_028000_itrs_0.999_ema_single_npz_{args.rejection_ratio}_/sorted_samples'), samples_top)

    dist.barrier()
    logger.log("sampling complete")

def create_argparser():

    defaults = dict(
        generator="determ",
        eval_batch=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        sampling_steps=40,
        model_path="",
        eval_seed=42,
        save_format='png',
        stochastic_seed=False,
        #data_name='cifar10',
        data_name='imagenet64',
        #schedule_sampler="lognormal",
        ind_1=0,
        ind_2=0,
        gamma=0.5,
        classifier_guidance=False,
        classifier_path="",
        cg_scale=1.0,
        generator_type='dummy',
        edm_style=False,
        target_snr=0.16,
        langevin_steps=1,
        rejection_ratio = 0.5,
        log_tau = 0.0,
        class_idx=614,
    )
    defaults.update(train_defaults(defaults['data_name']))
    defaults.update(model_and_diffusion_defaults(defaults['data_name']))
    defaults.update(cm_train_defaults(defaults['data_name']))
    defaults.update(ctm_train_defaults(defaults['data_name']))
    defaults.update(ctm_eval_defaults(defaults['data_name']))
    defaults.update(ctm_loss_defaults(defaults['data_name']))
    defaults.update(ctm_data_defaults(defaults['data_name']))
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
