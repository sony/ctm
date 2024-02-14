"""
Train a diffusion model on images.
"""

import argparse

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    train_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    ctm_train_defaults,
    ctm_eval_defaults,
    ctm_loss_defaults,
    ctm_data_defaults,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
import torch.distributed as dist
import torch
import numpy as np
from scipy import integrate


def normalizing_Z(std_max, std_min, rho):
    return 2. * torch.log(std_max / std_min)

def get_importance_time(batch_size, std_max, std_min, rho):
    u = torch.rand(batch_size, device=std_max.device)
    time = (std_max / std_min) ** (u / rho) - 1.
    time /= (std_max / std_min) ** (1. / rho) - 1.
    Z = normalizing_Z(std_max, std_min, rho)
    return time, Z

def gen_get_importance_time(batch, std_max, std_min, rho, t_min):
    u = torch.rand(batch.shape[0], device=batch.device)
    numerator = sigma(1., std_max, std_min, rho)
    denominator = sigma(t_min, std_max, std_min, rho)
    time = (numerator / denominator) ** u
    time *= denominator
    time = inv_sigma(time, std_max, std_min, rho)
    return time, 2. * rho * torch.log(numerator / denominator)

def sigma(t, std_max, std_min, rho):
    return std_min ** (1. / rho) + t * (std_max ** (1. / rho) - std_min ** (1. / rho))

def inv_sigma(sigma, std_max, std_min, rho):
    return (sigma - std_min ** (1. / rho)) / (std_max ** (1. / rho) - std_min ** (1. / rho))

def prior_logp(std_max, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * torch.log(2 * np.pi * std_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * std_max ** 2)

def residual(args, model, diffusion, images, teacher=False, std_max=80., std_min=0.002, rho=7, t_min=1e-3):
    z = torch.randn_like(images)
    std = sigma(t_min, std_max, std_min, rho)
    perturbed_data = images + std[:, None, None, None] * z
    if teacher:
        denoised = diffusion.get_denoised_and_G(model, perturbed_data, std, s=std, ctm=False, teacher=True)[0].to(
            torch.float64)
    else:
        denoised = diffusion.get_denoised_and_G(model, perturbed_data, std, s=std, ctm=True)[0].to(torch.float64)
    score = (denoised - perturbed_data) / std ** 2

    q_mean = perturbed_data + std[:, None, None, None] ** 2 * score
    q_std = std

    n_dim = np.prod(images.shape[1:])
    p_entropy = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(q_std) + 1.)
    q_recon = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(q_std)) + 0.5 / (q_std ** 2) * torch.square(
        images - q_mean).sum(axis=(1, 2, 3))
    residual = q_recon - p_entropy
    return residual

def elbo(args, model, diffusion, images, teacher=False, std_max=80., std_min=0.002, rho=7, t_min=1e-3):
    time, Z = gen_get_importance_time(images, std_max, std_min, rho, t_min)
    z = torch.randn_like(images)
    std = sigma(time, std_max, std_min, rho) ** rho
    perturbed_data = images + std[:, None, None, None] * z
    with torch.enable_grad():
        perturbed_data = perturbed_data.requires_grad_()
        if teacher:
            denoised = diffusion.get_denoised_and_G(model, perturbed_data, std, s=std, ctm=False, teacher=True)[0].to(torch.float64)
        else:
            denoised = diffusion.get_denoised_and_G(model, perturbed_data, std, s=std, ctm=True)[0].to(torch.float64)
        score = (denoised - perturbed_data) / (std ** 2)[:, None, None, None]
        a = std[:, None, None, None] * score
        mu = (std[:, None, None, None] ** 2) * score
        epsilon = torch.randint_like(images, low=0, high=2).float() * 2 - 1.
        Mu = - (
                torch.autograd.grad(mu, perturbed_data, epsilon, create_graph=False)[0] * epsilon
        ).reshape(images.size(0), -1).sum(1, keepdim=False) * Z

    Nu = - (a ** 2).reshape(images.size(0), -1).sum(1, keepdim=False) * Z / 2

    lp_z = torch.randn_like(images)
    lp_perturbed_data = images + std_max[:, None, None, None] * lp_z
    lp = prior_logp(std_max, lp_perturbed_data)

    elbos = lp + Mu + Nu

    residuals = residual(args, model, diffusion, images, teacher, std_max=std_max, std_min=std_min)
    elbos = - (elbos - residuals) / np.prod(list(images.shape[1:])) / np.log(2) + 7.

    return elbos

def drift_fn(model, diffusion, x, std_max, std_min, rho, t, teacher=False):
    """The drift function of the reverse-time SDE."""
    std = sigma(t, std_max, std_min, rho) ** rho
    volatility_sq = 2. * rho * (std ** 2) * (std_max ** (1. / rho) - std_min ** (1. / rho)) / \
                    (std_min ** (1. / rho) + t * (std_max ** (1. / rho) - std_min ** (1. / rho)))
    if teacher:
        denoised = diffusion.get_denoised_and_G(model, x, std, s=std, ctm=False, teacher=True)[0].to(
            torch.float64)
    else:
        denoised = diffusion.get_denoised_and_G(model, x, std, s=std, ctm=True)[0].to(torch.float64)
    #denoised = model(x, std, None if teacher else std)[0].to(torch.float64)
    score = (denoised - x) / (std ** 2)[:, None, None, None]
    drift = - volatility_sq[:, None, None, None] * score / 2.
    return drift

def div_fn(model, diffusion, x, std_max, std_min, rho, t, noise, teacher=False):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(drift_fn(model, diffusion, x, std_max, std_min, rho, t, teacher=teacher) * noise)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * noise, dim=tuple(range(1, len(x.shape))))

def likelihood(args, model, diffusion, images, teacher=False, std_max=80., std_min=0.002, rho=7, t_min=1e-3, mode='correct'):
    shape = images.shape
    noise = torch.randint_like(images, low=0, high=2).float() * 2 - 1.
    def ode_func(t, x):
        sample = torch.from_numpy(x[:-shape[0]].reshape(shape)).to(images.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = drift_fn(model, diffusion, sample, std_max, std_min, rho, vec_t, teacher=teacher).detach().cpu().numpy().reshape((-1,))
        logp_grad = div_fn(model, diffusion, sample, std_max, std_min, rho, vec_t, noise, teacher=teacher).detach().cpu().numpy().reshape((-1,))
        #print("t: ", t)
        #print("logp_grad: ", logp_grad)
        return np.concatenate([drift, logp_grad], axis=0)

    if mode == 'correct':
        z = torch.randn_like(images)
        std = sigma(t_min, std_max, std_min, rho) ** rho
        perturbed_data = images + std[:, None, None, None] * z
        init = np.concatenate([perturbed_data.detach().cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    elif mode == 'wrong':
        init = np.concatenate([images.detach().cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    else:
        raise NotImplementedError

    solution = integrate.solve_ivp(ode_func, (t_min, 1.), init, rtol=1e-3, atol=1e-3, method='RK45')
    nfe = solution.nfev
    zp = solution.y[:, -1]
    z = torch.from_numpy(zp[:-shape[0]].reshape(shape)).to(images.device).type(torch.float32)
    delta_logp = torch.from_numpy(zp[-shape[0]:].reshape((shape[0],))).to(images.device).type(torch.float32)
    prior_logp_ = prior_logp(std_max, z)
    #print("score bpd: ", - torch.mean(prior_logp + delta_logp) / np.prod(list(data.shape[1:])) / np.log(2) + 7. - inverse_scaler(-1.))

    if mode == 'correct':
        residual_nll = residual(args, model, diffusion, images, teacher, std_max=std_max, std_min=std_min)
        #print("residual bpd: ", torch.mean(residual_nll) / np.prod(list(images.shape[1:])) / np.log(2))
        delta_logp = delta_logp - residual_nll

    #print("logdet bpd: ", - torch.mean(logdet) / np.prod(list(batch.shape[1:])) / np.log(2))
    #print(prior_logp_.shape, prior_logp_.mean() / np.log(2))
    #print(delta_logp.shape, delta_logp.mean() / np.log(2))
    bpd = -(prior_logp_ + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N
    # A hack to convert log-likelihoods to bits/dim
    offset = 7.
    bpd = bpd + offset
    return bpd

def main():
    args = create_argparser().parse_args()
    if args.use_MPI:
        dist_util.setup_dist(args.device_id)
    else:
        dist_util.setup_dist_without_MPI(args.device_id)

    logger.configure(args, dir=args.out_dir)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size() * batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    data = load_data(
        args=args,
        data_name=args.data_name,
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        train_classes=args.train_classes,
        num_workers=args.num_workers,
        type=args.type,
        deterministic=args.deterministic,
    )

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    # Load Model
    model, diffusion = create_model_and_diffusion(args, teacher=args.training_mode=='edm')
    model.to(dist_util.dev())
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()

    resume_checkpoint = args.resume_checkpoint

    if resume_checkpoint:
        if dist.get_rank() == 0:
            logger.log(f"loading pretrained model from checkpoint: {resume_checkpoint}...")
            if dist.get_world_size() > 1:
                state_dict = torch.load(resume_checkpoint, map_location=dist_util.dev())  # "cpu")
            else:
                state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location='cpu',  # dist_util.dev()
                )
            model.load_state_dict(state_dict, strict=True)
            logger.log(f"end loading pretrained model from checkpoint: {resume_checkpoint}...")
            del state_dict

    assert 50000 % args.global_batch_size == 0
    std_max = torch.tensor([80.], device=dist_util.dev())
    std_min = torch.tensor([0.002], device=dist_util.dev())
    total_elbos = []
    for itr in range(args.num_student_elbo):
        elbos = np.array([])
        for k in range(50000 // args.global_batch_size):
            batch, cond = next(data)
            batch = batch.to(dist_util.dev())
            elbo_ = elbo(args, model, diffusion, batch, std_max=std_max, std_min=std_min,
                         teacher = (args.training_mode != 'ctm')).cpu().detach().numpy()
            elbos = np.concatenate((elbos, elbo_))
            #print(f"num samples: {batch.shape[0] * (k+1)}, bpd: {elbos.mean()}")
        total_elbos.append(elbos.mean())
        print(f"student ELBO after {(itr + 1)} runs: {total_elbos}")
    print(f"student ELBO after {args.num_student_elbo} runs: {np.mean(total_elbos)}")

    total_nlls = []
    for itr in range(args.num_student_nll):
        nlls = np.array([])
        for k in range(50000 // args.global_batch_size):
            batch, cond = next(data)
            batch = batch.to(dist_util.dev())
            likelihood_ = likelihood(args, model, diffusion, batch, std_max=std_max, std_min=std_min,
                                     teacher=(args.training_mode != 'ctm')).cpu().detach().numpy()
            nlls = np.concatenate((nlls, likelihood_))
            print(f"num samples: {batch.shape[0] * (k+1)}, bpd: {nlls.mean()}")
        total_nlls.append(nlls.mean())
        print(f"student bpds after {(itr + 1)} runs: {total_nlls}")
    print(f"student bpd after {args.num_student_nll} runs: {np.mean(total_nlls)}")

    assert 50000 % args.global_batch_size == 0
    std_max = torch.tensor([80.], device=dist_util.dev())
    std_min = torch.tensor([0.002], device=dist_util.dev())
    total_elbos = []
    for itr in range(args.num_teacher_elbo):
        elbos = np.array([])
        for k in range(50000 // args.global_batch_size):
            batch, cond = next(data)
            batch = batch.to(dist_util.dev())
            elbo_ = elbo(args, model, diffusion, batch, std_max=std_max, std_min=std_min,
                         teacher=True).cpu().detach().numpy()
            elbos = np.concatenate((elbos, elbo_))
            # print(f"num samples: {batch.shape[0] * (k+1)}, bpd: {elbos.mean()}")
        total_elbos.append(elbos.mean())
        print(f"teacher ELBO after {(itr + 1)} runs: {total_elbos}")
    print(f"teacher ELBO after {args.num_student_elbo} runs: {np.mean(total_elbos)}")

    total_nlls = []
    for itr in range(args.num_teacher_nll):
        nlls = np.array([])
        for k in range(50000 // args.global_batch_size):
            batch, cond = next(data)
            batch = batch.to(dist_util.dev())
            likelihood_ = likelihood(args, model, diffusion, batch, std_max=std_max, std_min=std_min,
                                     teacher=True).cpu().detach().numpy()
            nlls = np.concatenate((nlls, likelihood_))
            print(f"num samples: {batch.shape[0] * (k + 1)}, bpd: {nlls.mean()}")
        total_nlls.append(nlls.mean())
        print(f"teacher bpds after {(itr + 1)} runs: {total_nlls}")
    print(f"teacher bpd after {args.num_student_nll} runs: {np.mean(total_nlls)}")

def create_argparser():
    defaults = dict(
        data_name='cifar10',
        num_student_elbo=10,
        num_teacher_elbo=10,
        num_student_nll=1,
        num_teacher_nll=1,
    )
    defaults.update(train_defaults(defaults['data_name']))
    defaults.update(model_and_diffusion_defaults(defaults['data_name']))
    defaults.update(cm_train_defaults(defaults['data_name']))
    defaults.update(ctm_train_defaults(defaults['data_name']))
    defaults.update(ctm_eval_defaults(defaults['data_name']))
    defaults.update(ctm_loss_defaults(defaults['data_name']))
    defaults.update(ctm_data_defaults(defaults['data_name']))
    defaults.update()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
