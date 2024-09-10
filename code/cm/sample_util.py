import torch as th
import numpy as np
from .random_util import get_generator
from .nn import append_dims, append_zero
import cm.dist_util as dist_util
import cm.logger as logger
from cm.enc_dec_lib import get_classifier_guidance, vpsde
from torchvision.utils import make_grid, save_image
import blobfile as bf
import os
import time

from functools import partial

def get_t(ind, start_scales=40):
    rho = 7.
    sigma_max = 80.
    sigma_min = 0.002
    t = sigma_max ** (1 / rho) + ind / (start_scales - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    )
    t = t ** rho
    return t

def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler:str="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
    x_T=None,
    ctm=False,
    teacher=False,
    clip_output=True,
    train=False,
    ind_1=0,
    ind_2=0,
    gamma=1.,
    classifier=None,
    cg_scale=1.0,
    generator_type='dummy',
    edm_style=False,
    target_snr=0.16,
    langevin_steps=1,
    churn_step_ratio=0.0,
    # guidance=0.5,
    guidance=1.,
    # use_milstein_method=False,
):  
    # assert gamma == 1.
    
    if generator is None:
        if generator_type == 'dummy':
            generator = get_generator("dummy")
        elif generator_type == 'determ':
            generator = get_generator('determ', 10000, 0)

    if sampler in ["progdist", 'euler', 'exact', 'cm_multistep', 'gamma_multistep', 'exact_hybrid']:
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max-1e-4, rho, device=device)
    elif sampler in ['hybrid']:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)
    else:
        # sigmas = get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
    
    
    # if 'x_T' in model_kwargs.keys() and model_kwargs['x_T'] is not None:
    #     x_T = model_kwargs['x_T'].detach().clone()
    # else:
    #     # org:
    #     x_T = generator.randn(*shape, device=device) * sigma_max
    #     # x_T = th.randn(*shape, device=device) * sigma_max
    #     model_kwargs['x_T'] = x_T
        
    if x_T == None:
        # org
        # x_T = generator.randn(*shape, device=device) * sigma_max
        x_T = th.randn(*shape, device=device) * sigma_max
        # x_T = generator.randn(*shape, device=device) * np.sqrt(sigma_max**2 + 0.5**2)
        # if 'x_T' not in model_kwargs.keys():
        #     model_kwargs['x_T'] = x_T

    if 'x_T' not in model_kwargs.keys() or model_kwargs['x_T'] is None:
        model_kwargs['x_T'] = x_T
        
    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "exact": sample_exact,
        "gamma": sample_gamma_,
        "gamma_multistep": sample_gamma_multistep_,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
        "cm_multistep": sample_multistep,
        "hybrid": partial(sample_hybrid, beta_d=diffusion.beta_d, beta_min=diffusion.beta_min,),# use_milstein_method=use_milstein_method),
        "exact_hybrid": sample_exact_hybrid,
        "gamma_from_toy": sample_gamma_from_toy,
        "contri_sampler": partial(sample_contribution_hybrid_sampler, beta_d=diffusion.beta_d, beta_min=diffusion.beta_min, gamma=gamma),
        "contri_sampler2": partial(sample_contribution_hybrid_sampler2, beta_d=diffusion.beta_d, beta_min=diffusion.beta_min, gamma=gamma),
        "bridge_gamma_multistep": sample_bridged_gamma_multistep_,
        "contri_ddpm_pp": sample_contribution_dpm_pp, 
        "contri_ddpm_pp_cm": sample_ddpm_pp_cm,
    }[sampler]
    
    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler in ["multistep", "exact", "cm_multistep", "exact_hybrid"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps
        )
    elif sampler in ["gamma"]:
        sampler_args = dict(ind_1=ind_1, ind_2=ind_2)#, classifier=classifier, class_labels=model_kwargs,
                            #cg_scale=cg_scale, target_snr=target_snr, langevin_steps=langevin_steps,)
    elif sampler in ["gamma_multistep", "bridge_gamma_multistep"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps, gamma=gamma,
            #classifier = classifier, class_labels = model_kwargs, cg_scale = cg_scale, edm_style=edm_style,
        )
    elif sampler in ["hybrid"]:
        sampler_args = dict(
            pred_mode=diffusion.pred_mode, churn_step_ratio=churn_step_ratio, sigma_max=sigma_max, 
        )
    elif sampler in ["gamma_from_toy"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps, gamma=gamma,
        )
    else:
        sampler_args = {}
        
    if sampler in ['heun', 'hybrid', 'contri_sampler', 'contri_sampler2', 'bridge_gamma_multistep', "contri_ddpm_pp", "contri_ddpm_pp_cm"]:
        sampler_args['teacher'] = False if train else teacher
        sampler_args['ctm'] = ctm
    
    if sampler in ['hybrid', 'exact_hybrid', 'gamma_from_toy', 'contri_sampler', 'contri_sampler2', 'bridge_gamma_multistep', 'contri_ddpm_pp', 'contri_ddpm_pp_cm']:
        # TODO!!!!!!!!
        # TODO!: Fix the denoiser func below such that x_T is used 
        # (I think get_denoised_and_G and the network might need to be fixed for that. But make sure to refer to the ddbm code on this matter.)
        def denoiser_ddbm(x_t, t, s=th.zeros(x_T.shape[0]), xT=None, sig_max=80., device=device, return_both=False):
            # _, denoised = diffusion.denoise(model, x_t, t, **model_kwargs)
            denoised, G_theta = diffusion.get_denoised_and_G(model, x_t=x_t, t=t, s=s, ctm=ctm, teacher=teacher, 
                                                             sigma_T=sig_max, is_sampling=True,
                                                            #  **model_kwargs,
                                                            x_T=xT,)            
            if clip_denoised:
                denoised = denoised.clamp(-1, 1)
                
            if not return_both:
                if sampler in ['exact_hybrid']:
                    denoised = G_theta
                return denoised
            else:
                # if clip_denoised:
                #     G_theta = G_theta.clamp(-1, 1)                    
                return denoised, G_theta
            
        # assert churn_step_ratio == 0
        (x_0, path, nfe) = sample_fn(
            denoiser_ddbm,
            x_T,
            sigmas,
            generator,
            progress=progress,
            callback=callback,
            guidance=guidance,
            **sampler_args,
        )
        # print('NFE when using hybrid sampler:', nfe)
        
    else:
        
        def denoiser(x_t, t, s=th.ones(x_T.shape[0]), xT=None, sig_max=80., device=device):

            denoised, G_theta = diffusion.get_denoised_and_G(model, x_t=x_t, t=t, s=s, ctm=ctm, teacher=teacher, 
                                                             sigma_T=sig_max, 
                                                            #  x_T=xT,
                                                             **model_kwargs,)
            
            if sampler in ['exact', 'cm_multistep', 'onestep', 'gamma_multistep', 'gamma']:
                denoised = G_theta
            if clip_denoised:
                #print("clip denoised!!!")
                denoised = denoised.clamp(-1, 1)
            #if sampler in ['gamma']:
            #    return denoised, G_theta
            return denoised
    
        x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    if clip_output:
        #print("clip output")
        return x_0.clamp(-1, 1)
    return x_0

@th.no_grad()
def sample_contribution_hybrid_stepwise_sampler(
    denoiser,
    x: th.Tensor,
    sigmas,
    generator,
    progress=False,
    callback=None,
    sigma_max=80.0,
    churn_step_ratio=0.,
    pred_mode='ve',
    guidance=0.,
    beta_d=2,
    beta_min=0.1,
    ctm=True,
    separate_xT=None,
):
    """Contribution Sampler based on the new Contribution Stepwise Solver."""
    
    if separate_xT is None:
        x_T: th.Tensor = x.detach().clone()
    else:
        x_T = separate_xT.detach().clone()
    path = [x_T.detach().clone().cpu()]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    nfe = 0
    assert 0. <= churn_step_ratio < 1.
    
    if pred_mode.startswith('vp'):
        vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

        logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
        
        std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        
        logsnr = lambda t :  - 2 * th.log(vp_snr_sqrt_reciprocal(t))

        logsnr_T = logsnr(th.as_tensor(sigma_max))
        logs_T = logs(th.as_tensor(sigma_max))
    
    for j, i in enumerate(indices):
        # print('x:', x.shape)
        # exit()
        t1 = append_dims(sigmas[i] * s_in, x.ndim)
        t2 = append_dims(sigmas[i + 1] * s_in, x.ndim)
        # assert th.all(sigmas[i + 1] < sigmas[i])
        assert (t2 < t1).all()
        
        if churn_step_ratio > 0.:
            # Euler step with SDE
            
            # sigma_hat = sigmas[i] + (churn_step_ratio * (sigmas[i+1] - sigmas[i]))
            sigma_hat = t1 + (churn_step_ratio * (t2 - t1))
            
            # ORG:
            # denoised = denoiser(x, sigmas[i] * s_in, x_T)
            denoised = denoiser(x, t1, s=t1, xT=x_T)
            assert not x.isnan().any()
            assert not denoised.isnan().any()
            
            if pred_mode == 've':
                # d_1, gt2 = fixed_hybrid_sample_to_d(x, sigmas[i] , denoised, x_T, sigma_max, w=guidance, stochastic=True)
                d_1, gt2 = ddbm_to_d(x, t1, denoised, x_T, sigma_max, w=guidance, stochastic=True)
            elif pred_mode.startswith('vp'):
                d_1, gt2 = get_d_vp(x, denoised, x_T, std(t1),logsnr(t1), logsnr_T, logs(t1), logs_T, s_deriv(t1), vp_snr_sqrt_reciprocal(t1), vp_snr_sqrt_reciprocal_deriv(t1), guidance, stochastic=True)
            
            dt: th.Tensor = sigma_hat - t1
            # dw: th.Tensor = generator.randn_like(x) * dt.abs().sqrt()
            dw: th.Tensor = th.randn_like(x) * dt.abs().sqrt()
            # x = x + d_1 * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
            
            x_t = x_t + (d_1 * dt) + (gt2.sqrt() * dw)            
            nfe += 1
        else:
            sigma_hat = t1
            
        # = PART B
        # Euler step with PF ODE
        
        # ORG:
        # denoised = denoiser(x, sigma_hat * s_in, x_T)
        if ctm:
            # print('x', x.shape)
            # print('sigma_hat * s_in', (sigma_hat * s_in).shape)
            # print('x_T', (x_T).shape)
            # denoised = denoiser(x, t=sigma_hat * s_in, s=sigma_hat * s_in, xT=x_T)
            denoised = denoiser(x, t=sigma_hat, s=sigma_hat, xT=x_T)
        else:
            raise NotImplementedError("No teacher, so it cannot be used.")
            if teacher:
                denoised = denoiser(x, sigma_hat * s_in, s=None, xT=x_T)
            else:
                denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in, xT=x_T)
        
        nfe += 1
        if pred_mode == 've':
            # = PART C:
            # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
            # d = fixed_hybrid_sample_to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance)
            d = ddbm_to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance, stochastic=False)

        elif pred_mode.startswith('vp'):
            d = get_d_vp(x, denoised, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
        
        assert not x.isnan().any()
        assert not denoised.isnan().any()
            
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = t2 - sigma_hat
        
        x = x + (d * dt)
        path.append(x.detach().clone().cpu())
        
    return x, path, nfe

@th.no_grad()
def sample_contribution_hybrid_sampler(
    denoiser,
    x: th.Tensor,
    sigmas,
    generator,
    progress=False,
    callback=None,
    sigma_max=80.0,
    churn_step_ratio=0.,
    pred_mode='ve',
    guidance=0.,
    beta_d=2,
    beta_min=0.1,
    ctm=True,
    gamma=0.0,
    teacher=False,
):
    """Contribution Sampler."""
    assert 0. <= gamma <= 1.0, gamma
    
    x_T: th.Tensor = x.detach().clone()
    path = [x_T.detach().clone().cpu()]
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    nfe = 0
    
    if len(indices) == 1: # => One-step generation.
        assert (sigmas[1]).unique().item() == 0., sigmas
        sigma_max = append_dims(sigma_max * s_in, x.ndim)
        sigma_zero = append_dims(th.zeros_like(s_in), x.ndim)
        
        _, x = denoiser(x, t=sigma_max, s=sigma_zero, xT=x_T, sig_max=sigma_max, return_both=True)
        path.append(x.detach().clone().cpu())
        nfe += 1
    else:
        # for i in indices[:-1]:
        for i in indices:
            sigma_t1: th.Tensor = append_dims(sigmas[i] * s_in, x.ndim)
            sigma_t2: th.Tensor = append_dims(sigmas[i + 1] * s_in, x.ndim)
            sigma_hat: th.Tensor = append_dims(sigma_t2 * np.sqrt(1. - (gamma**2)), x.ndim)

            if (sigma_t2 > 0.0).any():
                # First approach:
                # _, x_gamma = denoiser(x, t=sigma_t1, s=sigma_hat, xT=x_T, sig_max=sigma_max, return_both=True)
                # t1 -> sigma_hat -> t2
                # x0_pred, _ = denoiser(x_gamma, t=sigma_hat, s=sigma_hat, xT=x_T, sig_max=sigma_max, return_both=True)
                # nfe += 2
                
                # Second approach (seems to perform better in the toy exp.):
                x0_pred, x_gamma = denoiser(x, t=sigma_t1, s=sigma_hat, xT=x_T, sig_max=sigma_max, return_both=True)
                nfe += 1
                
                x = corrected_gamma_sample_xt(x_gamma, gamma, x0_pred, x_T, sigma_t2, sigma_max=sigma_max, gen=generator)
                # 20th AUG: TODO: Make ctm_gamma_sample_xt() for baseline comparisons.

            else:
                
                assert th.all(sigmas[i+1] == 0.), "error!"
                assert len(sigmas[i:]) == 2, sigmas[i:]

                x, _, _nfe = sample_contribution_hybrid_stepwise_sampler(
                    denoiser, x, sigmas[i:], 
                    generator=generator, progress=progress, callback=None, 
                    sigma_max=sigma_max, churn_step_ratio=churn_step_ratio, 
                    pred_mode=pred_mode, beta_d=beta_d, beta_min=beta_min, ctm=ctm,
                    guidance=guidance, separate_xT=x_T,
                )
                nfe += _nfe

            path.append(x.detach().clone().cpu())
                
    return x, path, nfe

def corrected_gamma_sample_xt(x_gamma, gamma, x0, xT, sigma_t, sigma_max, noise=None, gen=None) -> th.Tensor:
    assert 0 <= gamma <= 1
    
    if gamma == 0:
        return x_t
    
    if noise is None:
        noise = th.randn_like(xT)
        
    sqrt_one_min_g_sq = np.sqrt(1. - (gamma**2))
    assert 0 <= sqrt_one_min_g_sq <= 1
    
    a_t: th.Tensor = append_dims((sigma_t**2)/((sigma_max**2) * th.ones_like(sigma_t)), xT.ndim) 
    assert th.all(0. <= a_t) and th.all(a_t <= 1.), f"'snrT_div_snrt' does not belong to [0, 1] !!\nsigma_t: {sigma_t}"
    
    std_t: th.Tensor = (1. - sqrt_one_min_g_sq) / th.sqrt(1. - ( a_t * ( 1. - np.square(gamma) ) ) )
    std_t = append_dims(std_t, xT.ndim)
    std_t = append_dims(sigma_t, xT.ndim) * std_t
    
    if std_t.isnan().any():
        print("here std!\nsigma_t:", sigma_t)
        # print('x_gamma', x_gamma)
        print('std_t', std_t)
        print('inside std_t:', th.sqrt(1. - ( a_t * ( 1. - np.square(gamma) ) ) ))
        print('gamma', gamma)
        print('a_t', a_t)
        # print('x0:', x0)
        exit()
        
    x_t = x_gamma + (std_t * noise)
    
    if th.all(sigma_t == 0):
        assert th.all(std_t == 0), f"sig_t should be 0: {std_t}"
        assert th.all(x_t == x_gamma), f"x_t and x_gamma should be equal."
    return x_t

def dpm_gamma_sample_xt(x_gamma, gamma, x0, xT, sigma_t, sigma_max, noise=None, gen=None) -> th.Tensor:
    assert 0 <= gamma <= 1
    
    if gamma == 0:
        return x_t
    
    if noise is None:
        noise = th.randn_like(xT)
        
    sqrt_one_min_g_sq = np.sqrt(1. - (gamma**2))
    assert 0 <= sqrt_one_min_g_sq <= 1
    
    a_t: th.Tensor = append_dims((sigma_t**2)/((sigma_max**2) * th.ones_like(sigma_t)), xT.ndim) 
    assert th.all(0. <= a_t) and th.all(a_t <= 1.), f"'snrT_div_snrt' does not belong to [0, 1] !!\nsigma_t: {sigma_t}"
    
    std_t: th.Tensor = (1. - sqrt_one_min_g_sq) / th.sqrt(1. - ( a_t * ( 1. - np.square(gamma) ) ) )
    std_t = append_dims(std_t, xT.ndim)
    std_t = append_dims(sigma_t, xT.ndim) * std_t
    
    if std_t.isnan().any():
        print("here std!\nsigma_t:", sigma_t)
        # print('x_gamma', x_gamma)
        print('std_t', std_t)
        print('inside std_t:', th.sqrt(1. - ( a_t * ( 1. - np.square(gamma) ) ) ))
        print('gamma', gamma)
        print('a_t', a_t)
        # print('x0:', x0)
        exit()
        
    x_t = x_gamma + (std_t * noise)
    
    if th.all(sigma_t == 0):
        assert th.all(std_t == 0), f"sig_t should be 0: {std_t}"
        assert th.all(x_t == x_gamma), f"x_t and x_gamma should be equal."
    return x_t

# def corrected_gamma_sample_xt(x_gamma, gamma, x0, xT, sigma_t, sigma_max, noise=None, gen=None) -> th.Tensor:
#     if noise is None:
#         # assert gen is not None
#         # noise = gen.randn_like(xT)
#         noise = th.randn_like(xT)
        
#     snrT_div_snrt: th.Tensor = append_dims((sigma_t**2)/((sigma_max**2) * th.ones_like(sigma_t)), xT.ndim) 
#     assert th.all(0. <= snrT_div_snrt) and th.all(snrT_div_snrt <= 1.), f"'snrT_div_snrt' does not belong to [0, 1] !!\nsigma_t: {sigma_t}"
    
#     a_t = snrT_div_snrt
#     b_t = 1. - snrT_div_snrt

#     gamma_pow2 = gamma**2
#     assert (0 <= gamma_pow2 <= 1) and (0 <= (gamma_pow2 ** 2) <= 1)
    
#     mean_t = x_gamma + (gamma_pow2 * a_t * (xT - x0))
    
#     sigma_sqrt_part: th.Tensor = (
#         (2 * (b_t + (a_t * gamma_pow2)))
#         - gamma_pow2 
#         - ((gamma_pow2**2) * a_t)
#         # - (2 * th.sqrt(b_t + (a_t * gamma_pow2)) * th.sqrt(b_t) * np.sqrt(1. - gamma_pow2))
#         - (2 * th.sqrt(b_t + (a_t * gamma_pow2)) * th.sqrt(b_t) * np.sqrt(1. - gamma_pow2))
#         # - (2 * th.sqrt(b_t + (a_t * gamma_pow2)) * th.sqrt(b_t - (gamma_pow2 * b_t)))
#     )
    
#     if (sigma_sqrt_part.min() < 0.0).any():
#         print("Before Clamping| Min:", sigma_sqrt_part.min(), ", Max:", sigma_sqrt_part.max())
#         sigma_sqrt_part = sigma_sqrt_part.clamp(min=0.)
#         print("After Clamping| Min:", sigma_sqrt_part.min(), ", Max:", sigma_sqrt_part.max())
    
#     std_t = sigma_t * sigma_sqrt_part.sqrt()
#     assert th.all(std_t >= 0)
    
#     if std_t.isnan().any():
#         print("here std!\nsigma_t:", sigma_t)
#         # print('x_gamma', x_gamma)
#         print('std_t', std_t)
#         print('parts of std:')
#         print(((2*b_t + (2*a_t * gamma_pow2))))
#         print(-gamma_pow2)
#         print((-(gamma_pow2**2) * a_t).item())
#         # print(-(2 * torch.sqrt(b_t + (a_t * gamma_pow2)) * torch.sqrt(b_t) * np.sqrt(1. - gamma_pow2)).item())
#         print(-(2 * th.sqrt(b_t + (a_t * gamma_pow2)) * th.sqrt(b_t) * np.sqrt((1. - gamma) * (1. + gamma))))
#         # print(-(2 * torch.sqrt(b_t + (a_t * gamma_pow2)) * torch.sqrt(b_t - (gamma_pow2 * b_t))).item())
        
#         print(
#             (
#                 ((2*b_t + (2*a_t * gamma_pow2)))
#                 - gamma_pow2 
#                 - ((gamma_pow2**2) * a_t)
#                 # - (2 * torch.sqrt(b_t + (a_t * gamma_pow2)) * torch.sqrt(b_t) * np.sqrt(1. - gamma_pow2))
#                 - (2 * th.sqrt(b_t + (a_t * gamma_pow2)) * th.sqrt(b_t) * np.sqrt((1. - gamma) * (1. + gamma)))
#                 # - (2 * torch.sqrt(b_t + (a_t * gamma_pow2)) * torch.sqrt(b_t - (gamma_pow2 * b_t)))
#             ).item()
#         )
#         print('gamma', gamma)
#         print('a_t', a_t)
#         # print('x0:', x0)
#         exit()
        
#     x_t = mean_t + (std_t * noise)
    
#     if th.all(sigma_t == 0):
#         assert th.all(std_t == 0), f"sig_t should be 0: {std_t}"
#         assert th.all(x_t == x_gamma), f"x_t and x_gamma should be equal."
#     return x_t

@th.no_grad()
def sample_contribution_hybrid_sampler2(
    denoiser,
    x: th.Tensor,
    sigmas,
    generator,
    progress=False,
    callback=None,
    sigma_max=80.0,
    churn_step_ratio=0.,
    pred_mode='ve',
    guidance=0.,
    beta_d=2,
    beta_min=0.1,
    ctm=True,
    gamma=0.0,
    teacher=False,
):
    assert gamma == 1
    """Contribution Sampler."""
    assert 0. <= gamma <= 1.0, gamma
    
    x_T: th.Tensor = x.detach().clone()
    path = [x_T.detach().clone().cpu()]
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    nfe = 0
    
    if len(indices) == 1: # => One-step generation.
        assert (sigmas[1]).unique().item() == 0., sigmas
        sigma_max = append_dims(sigma_max * s_in, x.ndim)
        sigma_zero = append_dims(th.zeros_like(s_in), x.ndim)
        
        _, x = denoiser(x, t=sigma_max, s=sigma_zero, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
        path.append(x.detach().clone().cpu())
        nfe += 1
    else:
        # for i in indices[:-1]:
        for i in indices:
            sigma_t1: th.Tensor = append_dims(sigmas[i] * s_in, x.ndim)
            sigma_t2: th.Tensor = append_dims(sigmas[i + 1] * s_in, x.ndim)
            sigma_hat: th.Tensor = append_dims(sigma_t2 * np.sqrt(1. - (gamma**2)), x.ndim)
            assert (sigma_t2 < sigma_t1).all()
            assert (sigma_hat <= sigma_t1).all()
            
            if (sigma_t2 > 0.0).any():
                
                # First approach:
                _, x_gamma = denoiser(x, t=sigma_t1, s=sigma_hat, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                nfe += 1
                
                if gamma == 1:
                    x0_pred = x_gamma.clone()
                else:
                    x0_pred, _ = denoiser(x_gamma, t=sigma_hat, s=sigma_hat, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                    nfe += 1
                
                # Second approach (seems to perform better in the toy exp.):
                # x0_pred, x_gamma = denoiser(x, t=sigma_t1, s=sigma_hat, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                # nfe += 1
                
                x = corrected_gamma_sample_xt(x_gamma, gamma, x0_pred, x_T.clone(), sigma_t=sigma_t2, sigma_max=sigma_max, gen=generator)
                # 20th AUG: TODO: Make ctm_gamma_sample_xt() for baseline comparisons.
            else:
                if th.all(sigmas[i+1] == 0.):
                    assert i + 1 == len(sigmas) - 1
                
                _, x = denoiser(x, t=sigma_t1, s=sigma_t2, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                nfe += 1
                
            path.append(x.detach().clone().cpu())
                
    return x, path, nfe


def get_lambda_t(sigma_t) -> th.Tensor:
    return -th.log(sigma_t)

@th.no_grad()
def sample_contribution_dpm_pp(
    denoiser,
    x: th.Tensor,
    sigmas,
    generator,
    progress=False,
    callback=None,
    sigma_max=80.0,
    churn_step_ratio=0.,
    pred_mode='ve',
    guidance=0.,
    beta_d=2,
    beta_min=0.1,
    ctm=True,
    gamma=0.0,
    teacher=False,
):
    # assert gamma == 1
    """Contribution Sampler."""
    assert 0. <= gamma <= 1.0, gamma
    # NOTE 9/6: If gamma == 1, then this becomes the Heun Solver version of DPMSolver++ !!
    
    x_T: th.Tensor = x.detach().clone()
    path = [x_T.detach().clone().cpu()]
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    nfe = 0
    
    start = time.time()
    
    if len(indices) == 1: # => One-step generation.
        print("NOTE: one-step gen!")
        assert (sigmas[1]).unique().item() == 0., sigmas
        sigma_max = append_dims(sigma_max * s_in, x.ndim)
        sigma_zero = append_dims(th.zeros_like(s_in), x.ndim)
        
        _, x = denoiser(x, t=sigma_max, s=sigma_zero, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
        path.append(x.detach().clone().cpu())
        nfe += 1
    else:
        print(f"NOTE: {len(indices)}-step gen!")
        # for i in indices[:-1]:
        for i in indices:
            sigma_zero = append_dims(th.zeros_like(s_in), x.ndim)
            sigma_t1 = append_dims(sigmas[i] * s_in, x.ndim)
            sigma_t2 = append_dims(sigmas[i + 1] * s_in, x.ndim)
            sigma_u = append_dims(sigma_t2 + ((sigma_t1 - sigma_t2) * (1. - gamma)), x.ndim)
            assert th.all(sigma_t1 >= sigma_u) and th.all(sigma_u >= sigma_t2) and th.all(sigma_t1 > sigma_t2)
            
            l_t1, l_t2, l_u = get_lambda_t(sigma_t1), get_lambda_t(sigma_t2), get_lambda_t(sigma_u)
            h = l_t2 - l_t1
            h_0 = l_u - l_t1
            phi = th.expm1(-h)      # == (t2/t1) - 1
            phi_u = th.expm1(-h_0)  # == (u/t1) - 1
            
            r = h_0 / h # ORG
            # r = (sigma_u - sigma_t1) / (sigma_t2 - sigma_t1)
            assert (r >= 0).all(), f"ERROR: gamma = {gamma}"
            
            if (sigma_t2 > 0.0).any():
                
                # First approach:
                # x0_pred_t1, _ = denoiser(x, t=sigma_t1, s=sigma_t1, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                # TODO: Second approach:
                _, x0_pred_t1 = denoiser(x, t=sigma_t1, s=sigma_zero, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                
                nfe += 1
                    
                if gamma == 0:
                    assert (phi_u == 0).all() and (sigma_u == sigma_t1).all()
                    D = x0_pred_t1
                    
                else:
                    x_u = ((sigma_u/sigma_t1) * x) - (phi_u * x0_pred_t1)
                    
                    x0_pred_u, _ = denoiser(x_u, t=sigma_u, s=sigma_u, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                    nfe += 1
                    
                    D = x0_pred_t1 + (0.5 * (x0_pred_u - x0_pred_t1) / r)
                
                x = ((sigma_t2 / sigma_t1) * x) - (phi * D)
                
            else:
                if th.all(sigmas[i+1] == 0.):
                    assert i + 1 == len(sigmas) - 1
                
                _, x = denoiser(x, t=sigma_t1, s=sigma_t2, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                nfe += 1
                
            path.append(x.detach().clone().cpu())
    
    end = time.time()
    logger.log(f"time taken to generate {x.shape[0]} images: {end - start} sec.")
    return x, path, nfe

@th.no_grad()
def sample_contribution_hybrid_sampler3(
    denoiser,
    x: th.Tensor,
    sigmas,
    generator,
    progress=False,
    callback=None,
    sigma_max=80.0,
    churn_step_ratio=0.,
    pred_mode='ve',
    guidance=0.,
    beta_d=2,
    beta_min=0.1,
    ctm=True,
    gamma=0.0,
    teacher=False,
):
    assert gamma == 1
    """Contribution Sampler."""
    assert 0. <= gamma <= 1.0, gamma
    
    x_T: th.Tensor = x.detach().clone()
    path = [x_T.detach().clone().cpu()]
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    nfe = 0
    
    if len(indices) == 1: # => One-step generation.
        assert (sigmas[1]).unique().item() == 0., sigmas
        sigma_max = append_dims(sigma_max * s_in, x.ndim)
        sigma_zero = append_dims(th.zeros_like(s_in), x.ndim)
        
        _, x = denoiser(x, t=sigma_max, s=sigma_zero, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
        path.append(x.detach().clone().cpu())
        nfe += 1
    else:
        # for i in indices[:-1]:
        for i in indices:
            sigma_t1: th.Tensor = append_dims(sigmas[i] * s_in, x.ndim)
            sigma_t2: th.Tensor = append_dims(sigmas[i + 1] * s_in, x.ndim)
            sigma_hat: th.Tensor = append_dims(sigma_t2 * np.sqrt(1. - (gamma**2)), x.ndim)
            
            if (sigma_t2 > 0).all():
                # First approach:
                # _, x_gamma = denoiser(x, t=sigma_t1, s=sigma_hat, xT=x_T, sig_max=sigma_max, return_both=True)
                # nfe += 1
                
                # Second approach (seems to perform better in the toy exp.):
                x0_pred, x_gamma = denoiser(x, t=sigma_t1, s=sigma_hat, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                nfe += 1
                
                if gamma == 0:
                    assert th.all(sigma_hat == sigma_t2)
                    x = x_gamma.clone()
                else:
                    # First approach's continuation:
                    x0_pred, _ = denoiser(x_gamma, t=sigma_hat, s=sigma_hat, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                    nfe += 1
                    
                    # 22nd aug:
                    assert th.all(sigma_t2 <= sigma_max)
                    a_t2 = append_dims(sigma_t2.square() / (sigma_max**2), x_T.ndim)
                    assert (0. <= a_t2).all() and (a_t2 <= 1.).all(), a_t2
                    b_t2 = 1. - a_t2
                    c_t2 = sigma_t2 * th.sqrt(b_t2)
                    
                    x = (x_T.clone() * a_t2) + (x0_pred.clone() * b_t2) + (c_t2 * th.randn_like(x_T))

                # x = corrected_gamma_sample_xt(x_gamma, gamma, x0_pred, x_T, sigma_t2, sigma_max=sigma_max, gen=generator)
                # 20th AUG: TODO: Make ctm_gamma_sample_xt() for baseline comparisons.
                
            else:
                
                if th.all(sigmas[i+1] == 0.):
                    assert i + 1 == len(sigmas) - 1
                    # assert th.all(x == x_gamma)
                    
                _, x = denoiser(x, t=sigma_t1, s=sigma_t2, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                nfe += 1
                
            path.append(x.detach().clone().cpu())
                
    return x, path, nfe


@th.no_grad()
def sample_ddpm_pp_cm(
    denoiser,
    x: th.Tensor,
    sigmas,
    generator,
    progress=False,
    callback=None,
    sigma_max=80.0,
    churn_step_ratio=0.,
    pred_mode='ve',
    guidance=0.,
    beta_d=2,
    beta_min=0.1,
    ctm=True,
    gamma=0.0,
    teacher=False,
):
    # assert gamma == 1
    """Contribution Sampler."""
    assert 0. <= gamma <= 1.0, gamma
    # NOTE 9/6: If gamma == 1, then this becomes the Heun Solver version of DPMSolver++ !!
    
    x_T: th.Tensor = x.detach().clone()
    path = [x_T.detach().clone().cpu()]
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    nfe = 0
    sigma_zero = append_dims(th.zeros_like(s_in), x.ndim)
    
    if len(indices) == 1: # => One-step generation.
        assert (sigmas[1]).unique().item() == 0., sigmas
        sigma_max = append_dims(sigma_max * s_in, x.ndim)
        
        _, x = denoiser(x, t=sigma_max, s=sigma_zero, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
        path.append(x.detach().clone().cpu())
        nfe += 1
    else:
        # for i in indices[:-1]:
        for i in indices:
            sigma_t1 = append_dims(sigmas[i] * s_in, x.ndim)
            sigma_t2 = append_dims(sigmas[i + 1] * s_in, x.ndim)
            sigma_u = append_dims(sigma_t2 + ((sigma_t1 - sigma_t2) * (1. - gamma)), x.ndim)
            assert th.all(sigma_t1 >= sigma_u) and th.all(sigma_u >= sigma_t2) and th.all(sigma_t1 > sigma_t2)
            
            l_t1, l_t2, l_u = get_lambda_t(sigma_t1), get_lambda_t(sigma_t2), get_lambda_t(sigma_u)
            h = l_t2 - l_t1
            h_0 = l_u - l_t1
            phi = th.expm1(-h)      # == (t2/t1) - 1
            phi_u = th.expm1(-h_0)  # == (u/t1) - 1
            
            r = h_0 / h # ORG
            # r = (sigma_u - sigma_t1) / (sigma_t2 - sigma_t1)
            assert (r >= 0).all(), f"ERROR: gamma = {gamma}"
            
            if (sigma_t2 > 0.0).any():
                
                # First approach:
                # x0_pred_t1, _ = denoiser(x, t=sigma_t1, s=sigma_t1, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                # TODO: Second approach:
                _, x0_pred_t1 = denoiser(x, t=sigma_t1, s=sigma_zero, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                
                nfe += 1
                    
                if gamma == 0:
                    assert (phi_u == 0).all() and (sigma_u == sigma_t1).all()
                    D = x0_pred_t1
                    
                else:
                    x_u = ((sigma_u/sigma_t1) * x) - (phi_u * x0_pred_t1)
                    
                    _, x0_pred_u = denoiser(x_u, t=sigma_u, s=sigma_zero, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                    nfe += 1
                    
                    D = x0_pred_t1 + (0.5 * (x0_pred_u - x0_pred_t1) / r)
                
                x = ((sigma_t2 / sigma_t1) * x) - (phi * D)
                
            else:
                if th.all(sigmas[i+1] == 0.):
                    assert i + 1 == len(sigmas) - 1
                
                _, x = denoiser(x, t=sigma_t1, s=sigma_t2, xT=x_T.clone(), sig_max=sigma_max, return_both=True)
                
                nfe += 1
                
            path.append(x.detach().clone().cpu())
                
    return x, path, nfe

@th.no_grad()
def sample_bridged_gamma_multistep_(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    gamma=0.0,
    ctm=True,
    teacher=False,
    guidance=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    x_T: th.Tensor = x.detach().clone()
    path = [x_T.detach().clone().cpu()]
    nfe = 0
    
    # if ts != [] and ts != None:
    #     sigmas = []
    #     t_max_rho = t_max ** (1 / rho)
    #     t_min_rho = t_min ** (1 / rho)
    #     s_in = x.new_ones([x.shape[0]])

    #     for i in range(len(ts)):
    #         sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
    #     sigmas = th.tensor(sigmas)
    #     sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    assert gamma >= 0.0 and gamma <= 1.0
    
    # for i in indices[:-1]:
    for i in indices:
        sigma = sigmas[i]
        # print(sigma, sigmas[i+1], gamma)
        s = (np.sqrt(1. - gamma ** 2) * (sigmas[i + 1] - 0.002) + 0.002) # Org CTM
        _, x_s = denoiser(x, sigma * s_in, s=s * s_in, xT=x_T, sig_max=t_max, return_both=True)
        nfe += 1
        if i < len(indices) - 2:
            std = th.sqrt((sigmas[i + 1] ** 2) - (s ** 2))
            # x = denoised + std * generator.randn_like(denoised) #th.randn_like(denoised)
            x = x_s + (std * th.randn_like(x_s))
        else:
            x = x_s
        path.append(x.detach().clone().cpu())

    return x, path, nfe

# 20th AUG: TODO:
# def ctm_gamma_sample_xt(x_gamma, gamma, x0, xT, sigma_hat, sigma_t, sigma_max, noise=None, gen=None) -> th.Tensor:
#     if noise is None:
#         assert gen is not None
#         noise = gen.randn_like(x0)
#     std = th.sqrt(sigma_t.square() - sigma_hat.square())
#     x = 
    
def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    # print('min_inv_rho', min_inv_rho)
    # print('max_inv_rho', max_inv_rho)
    
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up

@th.no_grad()
def sample_exact(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    
    x_T: th.Tensor = x.clone()
    
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices[:-1]:
        sigma = sigmas[i]
        # print(sigma, sigmas[i+1])
        if sigmas[i+1] != 0:
            # denoiser(x_t, t, s=th.ones(x_T.shape[0], device=device)):
            denoised = denoiser(x, sigma * s_in, s=sigmas[i + 1] * s_in, xT=x_T)
            x = denoised
        else:
            denoised = denoiser(x, sigma * s_in, s=sigma * s_in, xT=x_T)
            d = to_d(x, sigma, denoised)
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
        #else:
        #    denoised = denoiser(x, sigma * s_in)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        #x = denoised
    return x

@th.no_grad()
def sample_hybrid(
    denoiser,
    x,
    sigmas,
    generator,
    pred_mode='ve',
    progress=False,
    callback=None,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    churn_step_ratio=0.,
    guidance=1,
    teacher=False,
    ctm=False,
    # use_milstein_method=False,
):

    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_T: th.Tensor = x.clone()
    # x.detach().clone() was the part that I added on Sat 6/8!
    # path = [x]
    path = [x_T.detach().clone().cpu()]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0
    assert 0. <= churn_step_ratio < 1.

    if pred_mode.startswith('vp'):
        vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

        logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
        
        std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        
        logsnr = lambda t :  - 2 * th.log(vp_snr_sqrt_reciprocal(t))

        logsnr_T = logsnr(th.as_tensor(sigma_max))
        logs_T = logs(th.as_tensor(sigma_max))
    
    for j, i in enumerate(indices):
        assert th.all(sigmas[i + 1] < sigmas[i])
        
        if churn_step_ratio > 0:
            # 1 step euler
            sigma_hat = sigmas[i] + (churn_step_ratio * (sigmas[i+1] - sigmas[i]))
            
            # ORG:
            # denoised = denoiser(x, sigmas[i] * s_in, x_T)
            denoised = denoiser(x, sigmas[i] * s_in, s=sigmas[i] * s_in, xT=x_T)
            
            if pred_mode == 've':
                # d_1, gt2 = fixed_hybrid_sample_to_d(x, sigmas[i] , denoised, x_T, sigma_max, w=guidance, stochastic=True)
                d_1, gt2 = ddbm_to_d(x, sigmas[i] , denoised, x_T, sigma_max, w=guidance, stochastic=True)
            elif pred_mode.startswith('vp'):
                d_1, gt2 = get_d_vp(x, denoised, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
            
            dt = sigma_hat - sigmas[i]
            assert th.all(dt < 0)
            
            dw: th.Tensor = generator.randn_like(x) * dt.abs().sqrt()
            # x = x + d_1 * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
            x = x + (d_1 * dt) + (dw * gt2.sqrt())
            
            # if use_milstein_method:
            #     # dgt = 1 / torch.sqrt(2 * t1)
            #     dgt = 1 / th.sqrt(2 * sigmas[i])
            #     x += 0.5 * gt2.sqrt() * dgt * (dw.square() - dt)
            #     # print('new x_t:', x_t.shape)
            #     # exit()
            #     print('gt2.sqrt() * dgt |', (gt2.sqrt() * dgt).unique())
                
            nfe += 1
            
            path.append(x.detach().clone().cpu())
        else:
            sigma_hat = sigmas[i]
        
        # = PART B
        # heun step
        # ORG:
        # denoised = denoiser(x, sigma_hat * s_in, x_T)
        # print('-----------------------------------------------------')
        # print('sigma_hat:', sigma_hat, "i:", i, "sigmas", sigmas)
        # print(f'x (<- inp to denoiser func.): max= {x.max().item()}, min= {x.min().item()}')
        if ctm:
            denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in, xT=x_T)
        else:
            raise NotImplementedError("No teacher, so it cannot be used.")
            if teacher:
                denoised = denoiser(x, sigma_hat * s_in, s=None, xT=x_T)
            else:
                denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in, xT=x_T)
        # print('denoised', 'min', denoised.min().item(), 'max', denoised.max().item())
        
        if pred_mode == 've':
            # = PART C:
            # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
            # d = fixed_hybrid_sample_to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance)
            d = ddbm_to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance, stochastic=False)
            # print(f'nf1, i={i}|\nd: max= {d.max().item()}, min= {d.min().item()}')
            # exit()
        elif pred_mode.startswith('vp'):
            d = get_d_vp(x, denoised, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
            
        nfe += 1
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        assert th.all(dt < 0)
        # print(f'dt: max= {dt.max().item()}, min= {dt.min().item()}')
        # print(f'd * dt: max= {(d * dt).max().item()}, min= {(d * dt).min().item()}')
        
        # if (sigmas[i + 1] == 0) or (sigmas[i + 1] == sigma_min):
        if (sigmas[i + 1] == 0):
            # print('i:', i, 'sigmas[i + 1]', sigmas[i + 1])
            # print('sigmas', sigmas)
            # print()
            x = x + d * dt 
            
        else:
            # Heun's method
            x_2 = x + d * dt
            # print(f'x_2 (<- inp to denoiser func.): max= {x_2.max().item()}, min= {x_2.min().item()}')
            # ORG:
            # denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, x_T)
            if ctm:
                denoised_2 = denoiser(x_2, t=sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in, xT=x_T)
            else:
                raise NotImplementedError("No teacher, so it cannot be used.")
                if teacher:
                    denoised_2 = denoiser(x_2, t=sigmas[i + 1] * s_in, s=None, xT=x_T)
                else:
                    denoised_2 = denoiser(x_2, t=sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in, xT=x_T)
            
            if pred_mode == 've':
                # d_2 =  (x_2 - denoised_2) / append_dims(sigmas[i + 1], x.ndim)
                # d_2 = fixed_hybrid_sample_to_d(x_2,  sigmas[i + 1], denoised_2, x_T, sigma_max, w=guidance)
                d_2 = ddbm_to_d(x_2,  sigmas[i + 1], denoised_2, x_T, sigma_max, w=guidance, stochastic=False)
                # print(f'nf2, i={i}|\nd_2: max= {d_2.max().item()}, min= {d_2.min().item()}')
            elif pred_mode.startswith('vp'):
                d_2 = get_d_vp(x_2, denoised_2, x_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
            
            d_prime = (d + d_2) / 2
            # print(f'i={i}| d_prime: max= {d_prime.max().item()}, min= {d_prime.min().item()}')
            
            
            # print(f'x (prior to output): max= {x.max().item()}, min= {x.min().item()}')
            
            # noise = th.zeros_like(x) if 'flow' in pred_mode or pred_mode == 'uncond' else generator.randn_like(x)
            x = x + d_prime * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
            
            # print(f'i={i}| (d_prime * dt): max= {(d_prime * dt).max().item()}, min= {dt * d_prime.min().item()}')
            # print(f'x (output): max= {x.max().item()}, min= {x.min().item()}')
            # print('-----------------------------------------------------')
            # exit()
            
            nfe += 1
        # loss = (denoised.detach().cpu() - x0).pow(2).mean().item()
        # losses.append(loss)

        path.append(x.detach().clone().cpu())
    
    return x, path, nfe

def ddbm_to_d(x, sigma, denoised, x_T, sigma_max, w=1, stochastic=False):
    """Converts a denoiser output to a Karras ODE derivative."""
    sigma_sq: th.Tensor = append_dims(sigma**2, x.ndim)
    sigma_T_sq: th.Tensor = append_dims(th.ones_like(sigma)*(sigma_max**2), x.ndim)
    
    grad_pxtlx0: th.Tensor = (denoised - x) / sigma_sq
    grad_pxTlxt: th.Tensor = (x_T - x) / (sigma_T_sq - sigma_sq)
    # grad_pxTlxt: th.Tensor = (x_T - x) / (sigma_T_sq * ( 1. - ( sigma_sq / sigma_T_sq ) ))
    gt2: th.Tensor = 2 * append_dims(sigma, x.ndim)
    # d = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1))
    if stochastic:
        d: th.Tensor = - gt2 * grad_pxtlx0
        return d, gt2
    else:
        d: th.Tensor = - 0.5 * gt2 * (grad_pxtlx0 - (w * grad_pxTlxt))
        return d

def fixed_hybrid_sample_to_d(x, sigma, denoised, x_T, sigma_max, w=1, stochastic=False):
    """Converts a denoiser output to a Karras ODE derivative."""
    sigma = append_dims(sigma, x.ndim)
    sigma_max = append_dims(th.ones_like(sigma)*sigma_max, x.ndim)
    # _multiplier = (append_dims(th.ones_like(sigma)*sigma_max**2, x.ndim) - append_dims(sigma**2, x.ndim))
    # print('\nmin & max of denom of grad log h-func:', _multiplier.min().item(), _multiplier.max().item())
    # print('sigma max:', sigma_max**2)
    # print('min & max of sigma:', (sigma**2).min().item(), (sigma**2).max().item(), '\n')
    # diff = (x_T - x)
    # print('diff:', diff.min().item(), diff.max().item())
    # exit()
    
    snrT_div_snrt = sigma.square() / sigma_max.square()
    # ddbm_sigma = sigma * (1. - snrT_div_snrt).sqrt()
    ddbm_sigma_sq = sigma.square() * (1. - snrT_div_snrt)
    mu_hat_t = (snrT_div_snrt * x_T) + ((1. - snrT_div_snrt) * denoised)
    
    # grad_pxtlx0 = (x - mu_hat_t) / ddbm_sigma**2
    grad_pxtlx0 = (x - mu_hat_t) / ddbm_sigma_sq
    # grad_pxtlx0 = (denoised - x) / append_dims(sigma**2, x.ndim) # <- COMPLETELY WRONG!!!
    
    # print("grad_pxtlx0", grad_pxtlx0.min().item(), grad_pxtlx0.max().item())
    # grad_pxTlxt = (x_T - x) / (sigma_max**2 - sigma**2)
    grad_pxTlxt = (x_T - x) / (sigma_max.square() * (1. - snrT_div_snrt)) # Done on 6/27 cz it seems to be more correct, acc. to the CTM Toy exps. (that were also done on 6/27)
    # print("grad_pxTlxt", grad_pxTlxt.min().item(), grad_pxTlxt.max().item(), '\n----------------')
    # exit()
    gt2 = 2 * sigma
    # ORG:
    # d = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1))
    # d = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - 2 * w * grad_pxTlxt * (0 if stochastic else 1))
    d = (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 + (2 * w * grad_pxTlxt * (0 if stochastic else 1)))
    if stochastic:
        return d, gt2
    else:
        return d

@th.no_grad()
def sample_exact_hybrid(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    guidance=1,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    
    x_T: th.Tensor = x.clone()
    sigma_max = t_max
    nfe = 0
    path = []
    
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices[:-1]:
        sigma = sigmas[i]
        # print(sigma, sigmas[i+1])
        if sigmas[i+1] != 0:
            # denoiser(x_t, t, s=th.ones(x_T.shape[0], device=device)):
            denoised = denoiser(x_t=x, t=sigma * s_in, s=sigmas[i + 1] * s_in, xT=x_T)
            x = denoised
        else:
            denoised = denoiser(x_t=x, t=sigma * s_in, s=sigma * s_in, xT=x_T)
            # d = to_d(x, sigma, denoised)
            d = fixed_hybrid_sample_to_d(x, sigma, denoised, x_T, sigma_max, guidance)
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
            
            nfe += 1
        #else:
        #    denoised = denoiser(x, sigma * s_in)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        #x = denoised
    return x, path, nfe

# @th.no_grad()
# def sample_new_sampler(
#     denoiser,
#     x,
#     sigmas,
#     generator,
#     progress=False,
#     callback=None,
#     ts=[],
#     t_min=0.002,
#     t_max=80.0,
#     rho=7.0,
#     steps=40,
#     guidance=1,
# ):
#     """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    
#     x_T: th.Tensor = x.detach().clone()
#     sigma_max = t_max
#     nfe = 0
#     path = []
    
#     s_in = x.new_ones([x.shape[0]])
#     if ts != [] and ts != None:
#         sigmas = []
#         t_max_rho = t_max ** (1 / rho)
#         t_min_rho = t_min ** (1 / rho)
#         s_in = x.new_ones([x.shape[0]])

#         for i in range(len(ts)):
#             sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
#         sigmas = th.tensor(sigmas)
#         sigmas = append_zero(sigmas).to(x.device)
#     indices = range(len(sigmas) - 1)
#     if progress:
#         from tqdm.auto import tqdm

#         indices = tqdm(indices)

#     for i in indices[:-1]:
#         sigma = sigmas[i]
#         # print(sigma, sigmas[i+1])
#         sigma_hat = sigmas[i + 1]
        
#         if sigmas[i+1] != 0:
#             # denoiser(x_t, t, s=th.ones(x_T.shape[0], device=device)):
#             denoised = denoiser(x_t=x, t=sigma * s_in, s=sigma_hat * s_in, xT=x_T)
#             x = denoised
#         else:
#             denoised = denoiser(x_t=x, t=sigma * s_in, s=sigma * s_in, xT=x_T)
#             # d = to_d(x, sigma, denoised)
#             d = fixed_hybrid_sample_to_d(x, sigma, denoised, x_T, sigma_max, guidance)
#             dt = sigma_hat - sigma
#             x = x + d * dt
            
#             nfe += 1
#         #else:
#         #    denoised = denoiser(x, sigma * s_in)

#         if callback is not None:
#             callback(
#                 {
#                     "x": x,
#                     "i": i,
#                     "sigma": sigmas[i],
#                     "denoised": denoised,
#                 }
#             )
#         #x = denoised
#     return x, path, nfe

@th.no_grad()
def sample_gamma_from_toy(
    denoiser,
    x: th.Tensor,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    guidance=1,
    gamma=1.,
):

    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    
    x_T: th.Tensor = x.clone()
    nfe = 0
    path = []
    
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices[:-1]:
        sigma = sigmas[i]
        # print(sigma, sigmas[i+1])
        sigma_hat = sigmas[i+1] * np.sqrt(1 - (gamma**2))
        # x_t, t, s=th.ones(x_T.shape[0]), xT=None, sig_max=80., device=device, return_both=False
        _, x_gamma = denoiser(x_t=x, t=sigma * s_in, s=sigma_hat * s_in, xT=x_T, 
                                     sig_max=t_max, device=x.device, return_both=True)
        
        # Newly developed ctm gamma samplers:
        # 1. Getting x_0_pred from x and sigma:
        # x_0_pred, _ = denoiser(x_t=x, t=sigma * s_in, s=sigma * s_in, xT=x_T, 
        #                              sig_max=t_max, device=x.device, return_both=True)
        
        # 2. Getting x_0_pred from x_gamma and sigma_hat:
        x_0_pred, _ = denoiser(x_t=x_gamma, t=sigma_hat * s_in, s=sigma_hat * s_in, xT=x_T, 
                                     sig_max=t_max, device=x.device, return_both=True)
        
        if sigmas[i+1] > 0:
            # denoiser(x_t, t, s=th.ones(x_T.shape[0], device=device)):
            x = new_gamma_sample_xt(x_gamma=x_gamma, gamma=gamma, x0=x_0_pred, xT=x_T, 
                                    sigma_t=sigmas[i + 1], sigma_max=t_max, device=x.device)
            # "ctm_gamma_sampler" in toy exp
            # x = x_gamma + (gamma * sigmas[i+1] * th.randn_like(x).to(x.device))
        else:
            # 1st July: It kinda seems weird to use denoised here since denoised is G0 and not x0.
            # Try just using the above code (similar to how it is in the toy example).
            # denoised = denoiser(x_t=x, t=sigma * s_in, s=sigma * s_in, xT=x_T)
            # denoised, G_theta = denoiser(x_t=x.detach().clone(), t=sigma * s_in, s=sigma * s_in, xT=x_T.detach().clone())
            # d = to_d(x, sigma, denoised)
            # d = fixed_hybrid_sample_to_d(x, sigma, denoised, x_T, sigma_max, guidance)
            # dt = sigmas[i + 1] - sigma
            # x = x + d * dt
            x = x_gamma.clone()
        
        nfe += 1
        #else:
        #    denoised = denoiser(x, sigma * s_in)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": x_0_pred,
                }
            )
        #x = denoised
    return x, path, nfe

def new_gamma_sample_xt(x_gamma, gamma, x0, xT, sigma_t, sigma_max, device, noise=None):
        assert gamma >= 0. and gamma <= 1.
    
        if noise is None:
            noise = th.randn_like(x0).to(device)
        
        snrT_div_snrt: th.Tensor = append_dims((sigma_t**2)/((sigma_max**2) * th.ones_like(sigma_t)), x0.ndim) 
        a_t = snrT_div_snrt
        b_t = 1. - snrT_div_snrt
        
        # g = t * sqrt(1 - gamma_sq)
        gamma_sq = np.sqrt(gamma)
        
        mean_t = x_gamma + (gamma_sq * a_t * (xT - x0))
        std_t = sigma_t * th.sqrt(b_t + (1. - gamma_sq) * (b_t + (a_t * gamma_sq)))

        x_t = mean_t + (std_t * noise)
        return x_t
    
def get_d_vp(x, denoised, x_T, std_t, logsnr_t, logsnr_T, logs_t, logs_T, s_t_deriv, sigma_t, sigma_t_deriv, w, stochastic=False):
    
    a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
    b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
    
    mu_t = a_t * x_T + b_t * denoised 
    
    grad_logq = - (x - mu_t)/std_t**2 / (-th.expm1(logsnr_T - logsnr_t))
    # grad_logpxtlx0 = - (x - logs_t.exp()*denoised)/std_t**2 
    grad_logpxTlxt = -(x - th.exp(logs_t-logs_T)*x_T) /std_t**2  / th.expm1(logsnr_t - logsnr_T)

    f = s_t_deriv * (-logs_t).exp() * x
    gt2 = 2 * (logs_t).exp()**2 * sigma_t * sigma_t_deriv 
    # breakpoint()

    d = f -  gt2 * ((0.5 if not stochastic else 1)* grad_logq - w * grad_logpxTlxt)
    # d = f - (0.5 if not stochastic else 1) * gt2 * (grad_logpxtlx0 - w * grad_logpxTlxt* (0 if stochastic else 1))
    if stochastic:
        return d, gt2
    else:
        return d
    
@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@th.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x



@th.no_grad()
def sample_multistep(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)


    for i in indices[:-1]:
        sigma = sigmas[i]
        print(i, sigma, sigmas[i+1])
        #print(0.002 * s_in)
        denoised = denoiser(x, sigma * s_in, s=0.002 * s_in)
        if i < len(indices) - 2:
            print(th.sqrt(sigmas[i+1] ** 2 - 0.002 ** 2).item())
            x = denoised + th.sqrt(sigmas[i+1] ** 2 - 0.002 ** 2) * th.randn_like(denoised)
        else:
            x = denoised

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        #x = denoised
    return x


@th.no_grad()
def sample_multistep(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)


    for i in indices[:-1]:
        sigma = sigmas[i]
        # print(i, sigma, sigmas[i+1])
        #print(0.002 * s_in)
        denoised = denoiser(x, sigma * s_in, s=0.002 * s_in)
        if i < len(indices) - 2:
            # print(th.sqrt(sigmas[i+1] ** 2 - 0.002 ** 2).item())
            x = denoised + th.sqrt(sigmas[i+1] ** 2 - 0.002 ** 2) * th.randn_like(denoised)
        else:
            x = denoised

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        #x = denoised
    return x


@th.no_grad()
def sample_gamma_multistep_(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    gamma=0.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    assert gamma != 0.0 and gamma != 1.0
    # for i in indices[:-1]:
    for i in indices:
        sigma = sigmas[i]
        print(sigma, sigmas[i+1], gamma)
        s = (np.sqrt(1. - gamma ** 2) * (sigmas[i + 1] - 0.002) + 0.002)
        denoised = denoiser(x, sigma * s_in, s=s * s_in)
        if i < len(indices) - 2:
            std = th.sqrt(sigmas[i + 1] ** 2 - s ** 2)
            x = denoised + std * generator.randn_like(denoised) #th.randn_like(denoised)
        else:
            x = denoised

    return x


@th.no_grad()
def sample_gamma_(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    ind_1=0,
    ind_2=0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    assert ind_1 >= ind_2
    print(f"{get_t(0, steps)} -> {get_t(ind_1, steps)} -> {get_t(ind_2, steps)} -> {get_t(39, steps)}")
    ones = th.ones(x.shape[0], device=x.device)
    denoised = denoiser(x, get_t(0, steps) * ones, s=get_t(ind_1, steps) * ones)
    denoised = denoised + th.randn_like(denoised) * np.sqrt(get_t(ind_2, steps) ** 2 - get_t(ind_1, steps) ** 2)
    x = denoiser(denoised, get_t(ind_2, steps) * ones, s=get_t(39, steps) * ones)

    return x


@th.no_grad()
def sample_gamma_multistep(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    gamma=0.0,
    classifier=None,
    class_labels=None,
    cg_scale=1.,
    edm_style=False,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    vpsde_ = vpsde(cos_t_classifier=(64==x.shape[-2]))

    assert gamma != 0.0 and gamma != 1.0
    for i in indices[:-1]:
        sigma = sigmas[i]
        if edm_style:
            #s = gamma * (sigmas[i] - sigmas[i+1]) + sigmas[i+1]
            s = sigmas[i+1]
            if i > 0:
                sigma = gamma * (sigmas[i-1] - sigmas[i]) + sigmas[i]
        else:
            s = (np.sqrt(1. - gamma ** 2) * (sigmas[i + 1] - 0.002) + 0.002)
        print(sigmas[i], sigmas[i + 1], sigma, s, gamma)
        denoised = denoiser(x, sigma * s_in, s=s * s_in)
        if classifier != None and i < len(indices) - 2:
            denoised = denoised + cg_scale * (s * s_in)[:,None,None,None] * get_classifier_guidance(classifier, vpsde_, denoised, s * s_in, x.shape[-2], class_labels)
        if i < len(indices) - 2:
            if edm_style:
                sigma = gamma * (sigmas[i] - sigmas[i + 1]) + sigmas[i + 1]
                print(f"diffuse to {sigma.item()}")
                std = th.sqrt(sigma ** 2 - s ** 2)
            else:
                std = th.sqrt(sigmas[i + 1] ** 2 - s ** 2)
            x = denoised + std * generator.randn_like(denoised) #th.randn_like(denoised)
        else:
            x = denoised
    log_prob = get_classifier_guidance(classifier, vpsde_, x, 0.002 * s_in, x.shape[-2], class_labels, log_prob=True)
    print("log_prob: ", log_prob)
    return x


@th.no_grad()
def sample_gamma(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    ind_1=0,
    ind_2=0,
    classifier=None,
    class_labels=None,
    cg_scale=1.,
    target_snr=0.16,
    langevin_steps=1,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    assert ind_1 >= ind_2
    vpsde_ = vpsde(cos_t_classifier=(64 == x.shape[-2]))
    #print(f"{get_t(0)} -> {get_t(ind_1)} -> {get_t(ind_2)} -> {get_t(17)}")
    ones = th.ones(x.shape[0], device=x.device)
    G_s = denoiser(x, get_t(0, steps) * ones, s=get_t(ind_1, steps) * ones)[1]
    for _ in range(langevin_steps):
        if cg_scale != 0.0:
            cg = get_classifier_guidance(classifier, vpsde_, G_s, get_t(ind_1, steps) * ones, x.shape[-2], class_labels)
        else:
            cg = 0.
        denoised = denoiser(G_s, get_t(ind_1, steps) * ones, s=get_t(ind_1, steps) * ones)[0]
        score = (denoised - G_s) / ((get_t(ind_1, steps) ** 2) * ones)[:,None,None,None]
        noise = generator.randn_like(G_s)
        log_prob = score + cg_scale * cg
        grad_norm = th.norm(log_prob.reshape(log_prob.shape[0], -1), dim=-1).mean()
        noise_norm = th.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2
        G_s = G_s + step_size * log_prob + th.sqrt(2. * step_size) * generator.randn_like(G_s)

    G_s = G_s + generator.randn_like(G_s) * np.sqrt(get_t(ind_2, steps) ** 2 - get_t(ind_1, steps) ** 2)
    x = denoiser(G_s, get_t(ind_2, steps) * ones, s=get_t(39, steps) * ones)[1]
    #log_prob = get_classifier_guidance(classifier, vpsde_, x, 0.002 * ones, x.shape[-2], class_labels, log_prob=True)
    #print("log_prob: ", log_prob.mean())
    return x


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    teacher=False,
    ctm=False,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        # print("sigmas: ", sigmas[i], ctm, teacher)
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        # = PART B
        if ctm:
            denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        else:
            if teacher:
                denoised = denoiser(x, sigma_hat * s_in, s=None)
            else:
                denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        #print("denoised: ", denoised[0][0][0][:3])
        # = PART C:
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
            #print("last")
        else:
            #print("no last")
            # Heun's method
            x_2 = x + d * dt
            if ctm:
                denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
            else:
                if teacher:
                    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=None)
                else:
                    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@th.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@th.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in, None)


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in, None)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@th.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x



def application_sample(
    images,
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
    ctm=False,
    teacher=False,
    clip_output=True,
    train=False,
    ind_1=0,
    ind_2=0,
    gamma=0.5,
    generator_type='dummy',
    classifier=None,
    num_gradient_descent=1,
    scale=1.0,
    out_dir='',
):
    if generator is None:
        if generator_type == 'dummy':
            generator = get_generator("dummy")
        elif generator_type == 'determ':
            generator = get_generator('determ', num_samples=10000)

    sample_fn = {
        "colorization": iterative_colorization,
        "inpainting": iterative_inpainting,
        "inpainting_flip": iterative_inpainting,
        "superres": iterative_superres,
        "stroke": iterative_stroke_painting,
    }[sampler]

    def denoiser(x_t, t, s=th.ones(shape[0], device=device)):
        denoised, G_theta = diffusion.get_denoised_and_G(model, x_t, t, s, ctm, teacher, **model_kwargs)
        #denoised = G_theta
        return denoised, G_theta

    if 'inpainting' in sampler:
        sampler_args = {'flip': False if sampler == 'inpainting' else True}
    elif 'stroke' in sampler:
        sampler_args = {'classifier': classifier, 'num_gradient_descent': num_gradient_descent, 'scale': scale,
                        'out_dir': out_dir}
    else:
        sampler_args = {}

    x_out, x_in = sample_fn(
        denoiser,
        images,
        images,
        ts=ts,
        generator=generator,
        gamma=gamma,
        **sampler_args
    )
    if clip_output:
        #print("clip output")
        return x_out.clamp(-1, 1), x_in.clamp(-1, 1)
    return x_out, x_in

@th.no_grad()
def iterative_colorization(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
    gamma = 0.0,
):
    def obtain_orthogonal_matrix():
        vector = np.asarray([0.2989, 0.5870, 0.1140])
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(3)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)
    mask = th.zeros(*x.shape[1:], device=dist_util.dev())
    mask[0, ...] = 1.0

    def replacement(x0, x1):
        x0 = th.einsum("bchw,cd->bdhw", x0, Q).contiguous()
        x1 = th.einsum("bchw,cd->bdhw", x1, Q).contiguous()

        x_mix = x0 * mask + x1 * (1.0 - mask)
        x_mix = th.einsum("bdhw,cd->bchw", x_mix, Q).contiguous()
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    monochrome_images = replacement(images, th.zeros_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in, s=None)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(monochrome_images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)
    colored_images = x
    return colored_images, monochrome_images


@th.no_grad()
def iterative_inpainting(
    distiller,
    reference_images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
    gamma=0.0,
    flip=False,
):
    from PIL import Image, ImageDraw, ImageFont

    image_size = x.shape[-1]

    # create a blank image with a white background
    img = Image.new("RGB", (image_size, image_size), color="white")

    # get a drawing context for the image
    draw = ImageDraw.Draw(img)

    # load a font
    #font = ImageFont.truetype("Arial.ttf", 250)
    font = ImageFont.truetype("Arial.ttf", 500)

    # draw the letter "C" in black
    #draw.text((50, 0), "S", font=font, fill=(0, 0, 0))
    draw.rectangle((50,50, 200,200), fill=(0, 0, 0))

    # convert the image to a numpy array
    img_np = np.array(img)
    img_np = img_np.transpose(2, 0, 1)
    img_th = th.from_numpy(img_np).to(dist_util.dev())

    mask = th.zeros(*x.shape, device=dist_util.dev())
    #mask = mask.reshape(-1, 7, 3, image_size, image_size)
    mask = mask.reshape(-1, 1, 3, image_size, image_size)

    mask[::2, :, img_th > 0.5] = 1.0
    mask[1::2, :, img_th < 0.5] = 1.0
    mask = mask.reshape(-1, 3, image_size, image_size)

    def replacement(x0, x1, flip=False):
        if flip:
            x_mix = x0 * (1 - mask) + x1 * mask
        else:
            x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    masked_images = replacement(reference_images, -th.ones_like(reference_images), flip=flip)
    t = (t_max_rho + ts[0] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
    z = generator.randn_like(x)
    x = masked_images + t * z

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)

        if gamma == 1.0:
            try:
                x0 = distiller(x, t * s_in, s=0.002)
            except:
                x0 = distiller(x, t * s_in, s=None)
        else:
            s = (np.sqrt(1. - gamma ** 2) * (next_t - 0.002) + 0.002)
            x0 = distiller(x, t * s_in, s=s * s_in)
            print(t, next_t, s)

        if gamma == 1.0:
            x0 = replacement(masked_images, x0, flip=flip) # mask again
            x = x0 + generator.randn_like(x) * np.sqrt(next_t ** 2 - t_min ** 2)
        else:
            #masked_images = replacement(reference_images + s * generator.randn_like(masked_images), -th.ones_like(reference_images), flip=flip)
            x0 = replacement(masked_images + s * generator.randn_like(masked_images), x0, flip=flip) # mask again
            x = x0 + generator.randn_like(x) * next_t * gamma
    inpainted_images = x
    return inpainted_images, masked_images


@th.no_grad()
def iterative_superres(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
    gamma=0.0,
):
    patch_size = 8

    def obtain_orthogonal_matrix():
        vector = np.asarray([1] * patch_size**2)
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(patch_size**2)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)

    image_size = x.shape[-1]

    def replacement(x0, x1):
        x0_flatten = (
            x0.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x1_flatten = (
            x1.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x0 = th.einsum("bcnd,de->bcne", x0_flatten, Q).contiguous()
        x1 = th.einsum("bcnd,de->bcne", x1_flatten, Q).contiguous()
        x_mix = x0.new_zeros(x0.shape)
        x_mix[..., 0] = x0[..., 0]
        x_mix[..., 1:] = x1[..., 1:]
        x_mix = th.einsum("bcne,de->bcnd", x_mix, Q).contiguous()
        x_mix = (
            x_mix.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )
        return x_mix

    def average_image_patches(x):
        x_flatten = (
            x.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
        return (
            x_flatten.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    low_res_images = average_image_patches(images)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        if gamma == 1.0:
            try:
                x0 = distiller(x, t * s_in, s = 0.002 * s_in)
            except:
                x0 = distiller(x, t * s_in, s=None)
            x0 = th.clamp(x0, -1.0, 1.0)
        else:
            s = (np.sqrt(1. - gamma ** 2) * (next_t - 0.002) + 0.002)
            x0 = distiller(x, t * s_in, s=s * s_in)

        if gamma == 1.0:
            x0 = replacement(low_res_images, x0) # mask again
            x = x0 + generator.randn_like(x) * np.sqrt(next_t ** 2 - t_min ** 2)
        else:
            #low_images = replacement(low_res_images + s * generator.randn_like(low_res_images), -th.ones_like(low_res_images))
            x0 = replacement(low_res_images + s * generator.randn_like(low_res_images), x0) # mask again
            x = x0 + generator.randn_like(x) * next_t * gamma

        #x0 = replacement(low_res_images, x0)
        #x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)
    high_res_images = x
    return high_res_images, low_res_images


@th.no_grad()
def iterative_stroke_painting(
    distiller,
    reference_images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
    gamma=0.0,
    classifier=None,
    num_gradient_descent=1,
    scale=1.0,
    out_dir=''
):

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    t = (t_max_rho + ts[0] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
    z = generator.randn_like(x)
    x = reference_images + t * z

    next_t = (t_max_rho + ts[1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
    next_t = np.clip(next_t, t_min, t_max)

    print("t: ", t)
    print("next_t: ", next_t)
    vpsde_ = vpsde(cos_t_classifier=(64 == x.shape[-2]))
    if gamma == 1.0:
        try:
            x = distiller(x, t * s_in, s=0.002 * s_in)[1]
        except:
            x = distiller(x, t * s_in, s=None)[1]
        x = x + generator.randn_like(x) * next_t

    else:
        s = (np.sqrt(1. - gamma ** 2) * (next_t - 0.002) + 0.002)
        print("s: ", s)
        x = distiller(x, t * s_in, s=s * s_in)[1]
        if classifier != None:

            with th.enable_grad():
                for k in range(num_gradient_descent):
                    stroke_images = distiller(x, s * s_in, s=0.002 * s_in)[1]
                    sample = ((stroke_images + 1) * 127.5).clamp(0, 255).to(th.uint8)
                    sample = sample.permute(0, 2, 3, 1)
                    sample = sample.contiguous()
                    np.savez(os.path.join(out_dir,
                                          f"stroke_sample_{str(t)[:6]},{str(next_t)[:6]}_{str(s)[:6]}_numgrad_{num_gradient_descent}_scale_{scale}_{k}th.npz"),
                             sample.detach().cpu().numpy())
                    nrow = int(np.sqrt(stroke_images.shape[0]))
                    image_grid = make_grid((th.clamp(stroke_images, -1., 1.) + 1.) / 2., nrow, padding=2)
                    with bf.BlobFile(os.path.join(out_dir,
                                                  f"stroke_sample_{str(t)[:6]},{str(next_t)[:6]}_{str(s)[:6]}_numgrad_{num_gradient_descent}_scale_{scale}_{k}th.png"),
                                     "wb") as fout:
                        save_image(image_grid, fout)
                    '''with th.no_grad():
                        stroke_images = th.clamp(distiller(x, s * s_in, s=0.002 * s_in)[1], -1.0, 1.0)
                        nrow = int(np.sqrt(stroke_images.shape[0]))
                        image_grid = make_grid((th.clamp(stroke_images, -1., 1.) + 1.) / 2., nrow, padding=2)
                        with bf.BlobFile(os.path.join(out_dir, f"stroke_sample_{ts[0]},{ts[1]}_{k}.png"),
                                         "wb") as fout:
                            save_image(image_grid, fout)'''
                    x_ = x.float().clone().detach().requires_grad_()
                    mean_vp_tau, tau = vpsde_.transform_unnormalized_wve_to_normalized_vp(th.tensor([s], device=x.device))
                    fake = mean_vp_tau[:, None, None, None] * x_
                    true = mean_vp_tau[:, None, None, None] * (reference_images + s * generator.randn_like(x))
                    #true = mean_vp_tau[:, None, None, None] * (reference_images + s * z)
                    tau = tau.reshape(-1) * s_in
                    #print(fake.shape, tau.shape)
                    fake_feat = classifier(fake, timesteps=tau, feature=True)  # , condition=class_labels)
                    true_feat = classifier(true, timesteps=tau, feature=True)  # , condition=class_labels)
                    feat_diff = (fake_feat - true_feat) ** 2
                    print("loss: ", feat_diff.sum().item())
                    grad = th.autograd.grad(outputs=feat_diff.sum(), inputs=x_, retain_graph=False)[0]
                    #th.cuda.empty_cache()
                    with th.no_grad():
                        denoised = distiller(x, s * s_in, s=s * s_in)[0]
                        #print("denoised: ", denoised.shape)
                        score = (denoised - x) / ((s ** 2) * s_in)[:, None, None, None]
                        noise = generator.randn_like(x)
                        #print(score.shape, th.norm(score.reshape(score.shape[0],-1),1).shape,
                        #      grad.shape, th.norm(grad.reshape(grad.shape[0],-1),1).shape)
                        print("loss scale: ", (th.norm(score.reshape(score.shape[0],-1),dim=1) / \
                                   th.norm(grad.reshape(grad.shape[0],-1),dim=1)))
                        log_prob = score - scale * (th.norm(score.reshape(score.shape[0],-1),dim=1) / \
                                   th.norm(grad.reshape(grad.shape[0],-1),dim=1))[:,None,None,None] * grad
                        grad_norm = th.norm(log_prob.reshape(log_prob.shape[0], -1), dim=-1).mean()
                        noise_norm = th.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                        step_size = (0.1 * noise_norm / grad_norm) ** 2 * 2
                        x = x + step_size * log_prob + th.sqrt(2. * step_size) * generator.randn_like(x)
                        #th.cuda.empty_cache()
                    #print("x, grad scale: ", (x ** 2).mean(), (grad ** 2).mean())
                    #x = x - scale * grad
        stroke_images = th.clamp(distiller(x, s * s_in, s=0.002 * s_in)[1], -1.0, 1.0)
        sample = ((stroke_images + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        np.savez(os.path.join(out_dir,
                              f"stroke_sample_{str(t)[:6]},{str(next_t)[:6]}_{str(s)[:6]}_numgrad_{num_gradient_descent}_scale_{scale}_{num_gradient_descent}th.npz"),
                 sample.detach().cpu().numpy())
        nrow = int(np.sqrt(stroke_images.shape[0]))
        image_grid = make_grid((th.clamp(stroke_images, -1., 1.) + 1.) / 2., nrow, padding=2)
        with bf.BlobFile(os.path.join(out_dir, f"stroke_sample_{str(t)[:6]},{str(next_t)[:6]}_{str(s)[:6]}_numgrad_{num_gradient_descent}_scale_{scale}_{num_gradient_descent}th.png"),
                         "wb") as fout:
            save_image(image_grid, fout)
        x = x + generator.randn_like(x) * np.sqrt(next_t ** 2 - s ** 2)
    x = distiller(x, next_t * s_in, s=0.002 * s_in)[1]
    stroke_images = x
    stroke_images = th.clamp(stroke_images, -1.0, 1.0)
    sample = ((stroke_images + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    np.savez(os.path.join(out_dir,
                          f"stroke_sample_{str(t)[:6]},{str(next_t)[:6]}_{str(s if gamma != 1.0 else 0)[:6]}_numgrad_{num_gradient_descent}_scale_{scale}_{num_gradient_descent}th_after_diffusion.npz"),
             sample.detach().cpu().numpy())
    nrow = int(np.sqrt(stroke_images.shape[0]))
    image_grid = make_grid((th.clamp(stroke_images, -1., 1.) + 1.) / 2., nrow, padding=2)
    with bf.BlobFile(os.path.join(out_dir, f"stroke_sample_{str(t)[:6]},{str(next_t)[:6]}_{str(s if gamma != 1.0 else 0)[:6]}_numgrad_{num_gradient_descent}_scale_{scale}_{num_gradient_descent}th_after_diffusion.png"),
                     "wb") as fout:
        save_image(image_grid, fout)
    return stroke_images, reference_images
