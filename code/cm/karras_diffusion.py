"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .sample_util import fixed_hybrid_sample_to_d, ddbm_to_d
from . import dist_util, logger
import torch.distributed as dist

from .nn import mean_flat, append_dims, append_zero
import cm.script_util as script_util
from .enc_dec_lib import get_xl_feature, gaussian_blur
import blobfile as bf

def vp_logsnr(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return - th.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)
    
def vp_logs(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min

class KarrasDenoiser:
    def __init__(
        self,
        args,
        schedule_sampler,
        diffusion_schedule_sampler,
        feature_extractor=None,
        discriminator_feature_extractor=None,
        pred_mode='both',
        beta_d=2,
        beta_min=0.1,
    ):
        self.args = args
        self.schedule_sampler = schedule_sampler
        self.diffusion_schedule_sampler = diffusion_schedule_sampler
        self.feature_extractor = feature_extractor
        self.discriminator_feature_extractor = discriminator_feature_extractor
        self.num_timesteps = args.start_scales
        self.dist = nn.MSELoss(reduction='none')
        
        self.pred_mode = pred_mode
        self.beta_d = beta_d
        self.beta_min = beta_min
        assert not self.args.is_I2I, "(9/6) Error: Doing N2I exp for now, but is_I2I is True...!"

    def get_snr(self, sigmas):
        if self.pred_mode is not None and self.pred_mode.startswith('vp') and 'ddbm' in self.args.inner_parametrization:
            return vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
        else:
            return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas
    
    def get_weightings(self, weight_schedule:str, snrs, sigma_data, t, s, schedule_multiplier=None,):
        if weight_schedule == "snr":
            weightings = snrs
        elif weight_schedule == "snr+1":
            weightings = snrs + 1
        elif weight_schedule == "karras":
            weightings = snrs + 1.0 / sigma_data**2
        elif weight_schedule == 'cm':
            sigma = (1 / snrs) ** 0.5
            weightings = (sigma - self.args.sigma_min)**2 + sigma_data**2
            weightings = weightings / ((sigma - self.args.sigma_min)**2 * sigma_data**2)
        elif weight_schedule == "truncated-snr":
            weightings = th.clamp(snrs, min=1.0)
        elif weight_schedule == "uniform":
            weightings = th.ones_like(snrs)
        elif weight_schedule.startswith("ict"):
            # 20th Aug: 
            # NOTE: Note that 'sigma_s' is 'sigma_u' here actually!
            assert t is not None
            assert s is not None
            
            assert (t > s).all(), "sigma_t is not greater than sigma_u! This is a problem!"
            weightings: th.Tensor = 1 / (t - s)
            
            assert (not weightings.isnan().any()) and (weightings > 0).all(), "Error! Weightings is nan, or <= 0."
            
        elif weight_schedule.startswith("bridge_karras"):
            assert t is None
            assert s is None 
            
            sigma = (1 / snrs) ** 0.5
            sigma_max = self.args.sigma_max
            sigma_data_end = self.args.sigma_data_end # for cifar uncond, it is = np.sqrt(sigma_data**2 + sigma_max**2)
            cov_xy = self.args.cov_xy # for cifar uncond, it is = sigma_data**2
            c = 1
            
            if self.pred_mode == 've':
                # Might as well use the EDM scalings! 
                snrT_div_snrt = (sigma**2) / (sigma_max**2)
                a_t = snrT_div_snrt
                b_t = 1. - snrT_div_snrt
                c_t = (sigma**2) * (1. - snrT_div_snrt)
                
                A = th.square(a_t * sigma_data_end) + th.square(b_t * sigma_data) + (2 * a_t * b_t * cov_xy) + (c**2) * c_t
                weightings = A / (th.square(a_t) * (np.square(sigma_data_end * sigma_data) - cov_xy**2) + sigma_data**2 * c**2 * c_t )
                            
            elif self.pred_mode == 'vp':
                
                logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
                logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
                logs_T = vp_logs(1, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                c_t = -th.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

                A = a_t**2 * sigma_data_end**2 + b_t**2 * sigma_data**2 + 2*a_t * b_t * cov_xy + c**2 * c_t
                weightings = A / (a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + sigma_data**2 * c**2 * c_t )
                
            elif self.pred_mode == 'vp_simple' or  self.pred_mode == 've_simple':

                weightings = th.ones_like(snrs)
                    
        elif weight_schedule == "uniform_g":
            return 1./(1. - s / t) ** schedule_multiplier
        elif weight_schedule == "karras_weight":
            sigma = (1 / snrs) ** 0.5
            weightings = (sigma ** 2 + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
        elif weight_schedule == "sq-t-inverse":
            weightings = 1. / snrs ** 0.25
        else:
            raise NotImplementedError()
        return weightings

    def get_k_in(self):
        return (1. / self.args.sigma_data_end) if self.args.do_xT_precond else 1.
    
    def get_lambda_t(self, sigma_t):
        return -th.log(sigma_t)
    
    def get_c_in(self, sigma, inner_parametrization='edm'):
        sigma_max = self.args.sigma_max
        sigma_data = self.args.sigma_data
        sigma_data_end = self.args.sigma_data_end # for cifar uncond, it is = np.sqrt(sigma_data**2 + sigma_max**2)
        cov_xy = self.args.cov_xy # for cifar uncond, it is = sigma_data**2
        c = 1.
        
        if inner_parametrization == 'edm':
            c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        elif inner_parametrization == 'cm':
            c_in = 1 / ((sigma - self.args.sigma_min)**2 + sigma_data**2) ** 0.5
        elif inner_parametrization == 'no':
            c_in = th.ones_like(sigma)
        elif inner_parametrization == 'cm_ddbm':
            sigma = sigma - self.args.sigma_min
            snrT_div_snrt = sigma**2 / sigma_max**2
            a_t = snrT_div_snrt
            b_t = 1. - snrT_div_snrt
            c_t = (sigma ** 2) * (1. - snrT_div_snrt)
            
            # A = sigma**4 / sigma_max**4 * sigma_data_end**2 + (1 - sigma**2 / sigma_max**2)**2 * sigma_data**2 + 2*sigma**2 / sigma_max**2 * (1 - sigma**2 / sigma_max**2) * cov_xy + c **2 * sigma**2 * (1 - sigma**2 / sigma_max**2)
            A = th.square(a_t * sigma_data_end) + th.square(b_t * sigma_data) + (2 * a_t * b_t * cov_xy) + (c**2 * c_t)
            c_in = 1 / A.sqrt()
        elif inner_parametrization == 'ddbm':
            # print('get_c_in ddbm')
            snrT_div_snrt = sigma**2 / sigma_max**2
            a_t = snrT_div_snrt
            b_t = 1. - snrT_div_snrt
            c_t = (sigma ** 2) * (1. - snrT_div_snrt)
            
            # A = (snrT_div_snrt**2) * sigma_data_end**2 + (1 - snrT_div_snrt)**2 * sigma_data**2 + 2*snrT_div_snrt * (1 - snrT_div_snrt) * cov_xy + c **2 * sigma**2 * (1 - snrT_div_snrt)
            A: th.Tensor = (a_t.square() * (sigma_data_end**2)) + (b_t.square() * (sigma_data**2)) + (2 * a_t * b_t * cov_xy) + (c**2 * c_t)
            c_in = 1 / A.sqrt()
        else:
            raise ValueError(f'Param "inner_parametrization" is an invalid value.')
        
        return c_in

    def get_inner_scalings(self, t, inner_parametrization='no'):
        
        if inner_parametrization == 'edm':
            c_skip, c_out = self.get_edm_scalings(t)
        elif inner_parametrization == 'cm':
            c_skip, c_out = self.get_cm_scalings(t)
        elif inner_parametrization == 'no':
            c_skip, c_out = th.zeros_like(t), th.ones_like(t)
        elif inner_parametrization == 'cm_ddbm':
            c_skip, c_out = self.get_cm_ddbm_scalings(t)
        elif inner_parametrization == 'ddbm':
            c_skip, c_out = self.get_ddbm_scalings(t)
        else:
            raise ValueError(f'Param inner_parametrization is an invalid value "{inner_parametrization}".')
        
        return c_skip, c_out

    def get_outer_scalings(self, t, s=None, outer_parametrization='euler'):
        if outer_parametrization == 'euler':
            c_skip = s / t
        elif outer_parametrization == 'variance':
            c_skip = (((s - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2) / (
                        (t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)).sqrt()
        elif outer_parametrization == 'euler_variance_mixed':
            c_skip = s / (t + 1.) + \
                     (((s - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2) /
                      ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)).sqrt() / (t + 1.)
        c_out = (1. - s / t)
        return c_skip, c_out

    def get_edm_scalings(self, sigma):
        c_skip = self.args.sigma_data**2 / (sigma**2 + self.args.sigma_data**2)
        c_out = sigma * self.args.sigma_data / (sigma**2 + self.args.sigma_data**2) ** 0.5
        return c_skip, c_out

    def get_cm_scalings(self, sigma):
        c_skip = self.args.sigma_data**2 / (
            (sigma - self.args.sigma_min) ** 2 + self.args.sigma_data**2
        )
        # c_out = (
        #     (sigma - self.args.sigma_min)
        #     * self.args.sigma_data
        #     / (sigma**2 + self.args.sigma_data**2) ** 0.5
        # )
        c_out = (
            (sigma - self.args.sigma_min)
            * self.args.sigma_data
            / ((sigma - self.args.sigma_min)**2 + self.args.sigma_data**2) ** 0.5
        )
        return c_skip, c_out
    
    def get_cm_ddbm_scalings(self, sigma):
        return self.get_ddbm_scalings(sigma - self.args.sigma_min)
        
    def get_ddbm_scalings(self, sigma):
        if self.pred_mode is None:
            raise ValueError("Use EDM Scalings instead of DDBM Scalings; And/or set the 'inner_parametrization' param to 've' or 'vp'.")
        
        sigma_max = self.args.sigma_max
        sigma_data = self.args.sigma_data
        sigma_data_end = self.args.sigma_data_end # for cifar uncond, it is = np.sqrt(sigma_data**2 + sigma_max**2)
        cov_xy = self.args.cov_xy # for cifar uncond, it is = sigma_data**2
        c = 1.

        if self.pred_mode == 've':
            # Might as well use the EDM scalings!
            
            # print("sigma max at get_bridge_scalings", self.sigma_max)
            # A = sigma**4 / sigma_max**4 * sigma_data_end**2 + (1 - sigma**2 / sigma_max**2)**2 * sigma_data**2 + 2*sigma**2 / sigma_max**2 * (1 - sigma**2 / sigma_max**2) * cov_xy + c **2 * sigma**2 * (1 - sigma**2 / sigma_max**2)
            # c_in = 1 / (A) ** 0.5
            # c_skip = ((1 - sigma**2 / sigma_max**2) * sigma_data**2 + sigma**2 / sigma_max**2 * cov_xy)/ A
            # c_out =((sigma/sigma_max)**4 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + sigma_data**2 *  c **2 * sigma**2 * (1 - sigma**2/sigma_max**2) )**0.5 * c_in
            
            snrT_div_snrt = sigma**2 / sigma_max**2
            a_t = snrT_div_snrt
            b_t = 1. - snrT_div_snrt
            c_t = (sigma ** 2) * (1. - snrT_div_snrt)
            
            A = th.square(a_t * sigma_data_end) + th.square(b_t * sigma_data) + (2 * a_t * b_t * cov_xy) + c**2 * c_t
            c_in: th.Tensor = 1 / (A ** 0.5)
            c_skip: th.Tensor = ((b_t * (sigma_data**2)) + (a_t * cov_xy))/ A
            c_out: th.Tensor = th.sqrt(
                (th.square(a_t) * (np.square(sigma_data_end * sigma_data) - np.square(cov_xy))) + (np.square(sigma_data * c) * c_t)
            ) * c_in
            
            # print('-' * 20)
            # print("Might as well use the EDM scalings!")
            # print('-' * 20)
            
            return c_skip, c_out
    
        elif self.pred_mode == 'vp':

            logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
            logs_T = vp_logs(1, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
            b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -th.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

            A = a_t**2 * sigma_data_end**2 + b_t**2 * sigma_data**2 + 2*a_t * b_t * cov_xy + c**2 * c_t

            c_in = 1 / (A) ** 0.5
            c_skip = (b_t * sigma_data**2 + a_t * cov_xy)/ A
            c_out =(a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + sigma_data**2 *  c **2 * c_t )**0.5 * c_in
            return c_skip, c_out
            
    
        elif self.pred_mode == 've_simple' or self.pred_mode == 'vp_simple':
            # c_in = th.ones_like(sigma)
            c_out = th.ones_like(sigma) 
            c_skip = th.zeros_like(sigma)
            return c_skip, c_out, c_in

    def calculate_adaptive_weight(self, loss1, loss2, last_layer=None):
        loss1_grad = th.autograd.grad(loss1, last_layer, retain_graph=True)[0]
        loss2_grad = th.autograd.grad(loss2, last_layer, retain_graph=True)[0]
        d_weight = th.norm(loss1_grad) / (th.norm(loss2_grad) + 1e-4)
        d_weight = th.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def rescaling_t(self, t):
        rescaled_t = 1000 * 0.25 * th.log(t + 1e-44)
        return rescaled_t

    # def get_t(self, ind, sigma_min=None, sigma_max=None, rho=None, num_scales=None):
    #     assert num_scales is not None and self.args.start_scales <= num_scales <= self.args.end_scales
    #     rho = self.args.rho if rho is None else rho
    #     sigma_max = self.args.sigma_max if sigma_max is None else sigma_max
    #     sigma_min = self.args.sigma_min if sigma_min is None else sigma_min
    #     assert sigma_min >= self.args.sigma_min, f"The min sigma value must be at least args.sigma_min={self.args.sigma_min}."
    #     # sigma_max = sigma_max-1e-4
        
    #     if self.args.time_continuous:
    #         t = sigma_max ** (1 / rho) + ind * (
    #                 sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    #         )
    #         t = t ** rho
    #     else:
    #         # t = sigma_max ** (1 / rho) + ind / (self.args.start_scales - 1) * (
    #         #         sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    #         # )
    #         t = sigma_max ** (1 / rho) + ind / (num_scales - 1) * (
    #                 sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    #         )
    #         t = t ** rho
    #     return t
    
    # def karras_schedule(self, sigma_min=None, sigma_max=None, rho=None, num_scales=None) -> th.Tensor:
    #     assert num_scales is not None, num_scales
    #     if self.args.scale_mode == 'ict_exponential':
    #         assert self.args.start_scales <= num_scales <= self.args.end_scales, num_scales
    #     rho = self.args.rho if rho is None else rho
    #     sigma_max = self.args.sigma_max if sigma_max is None else sigma_max
    #     sigma_min = self.args.sigma_min if sigma_min is None else sigma_min
    #     assert sigma_min >= self.args.sigma_min, f"The min sigma value must be at least args.sigma_min={self.args.sigma_min}."
    #     # sigma_max = sigma_max-1e-4
        
    #     # steps = th.arange(num_scales) / max(num_scales - 1, 1)
    #     steps = th.arange(num_scales)

    #     if self.args.time_continuous:
    #         sigmas: th.Tensor = sigma_max ** (1 / rho) + steps * (
    #                 sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    #         )
    #         sigmas = sigmas ** rho
    #     else:
    #         # t = sigma_max ** (1 / rho) + ind / (self.args.start_scales - 1) * (
    #         #         sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    #         # )
    #         sigmas: th.Tensor = sigma_max ** (1 / rho) + steps / (num_scales - 1) * (
    #                 sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    #         )
    #         sigmas = sigmas ** rho
        
    #     sigmas = append_zero(sigmas).to(sigmas.device)
    #     # print('sigmas', sigmas)
    #     # exit()
    #     return sigmas
    
    def karras_schedule(self, sigma_min=None, sigma_max=None, rho=7.0, n=None, device="cpu"):
        assert n is not None
        sigma_max = self.args.sigma_max if sigma_max is None else sigma_max
        sigma_min = self.args.sigma_min if sigma_min is None else sigma_min
        
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        return append_zero(sigmas).to(device)

    def get_num_heun_step(self, start_scales=-1, num_heun_step=-1, num_heun_step_random=None, heun_step_strategy='', time_continuous=None):
        if start_scales == -1:
            start_scales = self.args.start_scales
            
        if num_heun_step == -1:
            num_heun_step = self.args.hun_heun_step
            
        if num_heun_step_random == None:
            num_heun_step_random = self.args.num_heun_step_random
            
        if heun_step_strategy == '':
            heun_step_strategy = self.args.heun_step_strategy
            
        if time_continuous == None:
            time_continuous = self.args.time_continuous
            
        if num_heun_step_random:
            
            if time_continuous:
                num_heun_step = np.random.rand() * num_heun_step / start_scales
            else:
                if heun_step_strategy == 'uniform':
                    num_heun_step = np.random.randint(1,1+num_heun_step)
                elif heun_step_strategy == 'weighted':
                    p = np.array([i ** self.args.heun_step_multiplier for i in range(1,1+num_heun_step)])
                    p = p / sum(p)
                    num_heun_step = np.random.choice([i+1 for i in range(len(p))], size=1, p=p)[0]
        
        else:
            if time_continuous:
                num_heun_step = num_heun_step / start_scales
            
            else:
                num_heun_step = num_heun_step
        return num_heun_step

    def get_gan_time(self, x_start, noise, x_t, t, t_dt, s, indices, num_heun_step, gan_num_heun_step):
        if gan_num_heun_step != -1:
            indices, _ = self.schedule_sampler.sample_t(self.args, x_start.shape[0], x_start.device, gan_num_heun_step,
                                                            self.args.time_continuous)
            new_indices = self.schedule_sampler.sample_s(self.args, x_start.shape[0], x_start.device,
                                                             indices,
                                                             gan_num_heun_step, self.args.time_continuous,
                                                             N=self.args.start_scales)
            t = self.get_t(indices, num_scales=self.args.start_scales)
            x_t = x_start + noise * append_dims(t, x_start.ndim)
            t_dt = self.get_t(indices + gan_num_heun_step, num_scales=self.args.start_scales)
            s = self.get_t(new_indices, num_scales=self.args.start_scales)
            num_heun_step = gan_num_heun_step
        return x_t, t, t_dt, s, indices, num_heun_step

    @th.no_grad()
    def heun_solver(self, target_model, x, ind, dims, t, t_dt, x_0=None, ctm=True, num_step=1, use_x0_as_denoised_in_solver=True, num_scales=None, sigmas=None, **model_kwargs):
        assert num_scales is not None
        assert sigmas is not None
        assert use_x0_as_denoised_in_solver
        assert (t == sigmas[ind]).all() 
        assert (t_dt == sigmas[ind + num_step]).all()
                    
        with th.no_grad():
            x_T: th.Tensor = model_kwargs['x_T']
            if self.args.self_learn:
                target_model.eval()
                if self.args.self_learn_iterative:
                    
                    for k in range(num_step):
                        # sigmas = self.karras_schedule(sigma_max=self.args.sigma_max, num_scales=num_scales)
                        t1 = append_dims(sigmas[ind + k], dims)
                        t2 = append_dims(sigmas[ind + k + 1], dims)
                        
                        if use_x0_as_denoised_in_solver:
                            assert x_0 is not None
                            denoised = x_0
                        else:
                            target_model.eval()
                            denoised, _ = self.get_denoised_and_G(target_model, x, t1, t1, ctm=ctm, **model_kwargs)
                        
                        assert not denoised.isnan().any()
                        
                        # d = (x - denoised) / append_dims(t, dims)
                        d = ddbm_to_d(x, t1, denoised, x_T, self.args.sigma_max, w=self.args.guidance_scale, stochastic=False)
                        assert not d.isnan().any()
                        K1 = d * (t2 - t1)
                        
                        if use_x0_as_denoised_in_solver:
                            assert x_0 is not None
                            denoised = x_0
                        else:
                            target_model.eval()
                            denoised, _ = self.get_denoised_and_G(target_model, x + K1, t2, t2, ctm=ctm, **model_kwargs)
                        
                        assert not denoised.isnan().any()
                        
                        # next_d = (x_phi_ODE_1st - denoised2) / append_dims(t2, dims)
                        next_d = ddbm_to_d(x + K1, t2, denoised, x_T, self.args.sigma_max, w=self.args.guidance_scale, stochastic=False)
                        assert not next_d.isnan().any()
                        K2 = next_d * (t2 - t1)
                        
                        d_prime = K1 + K2
                        x = x + (0.5 * d_prime)
                
                #     for k in range(num_step):
                #         t = sigmas[ind + k]
                #         t2 = sigmas[ind + k + 1]
                #         # t = self.get_t(ind + k, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)
                #         # t2 = self.get_t(ind + k + 1, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)
                #         _, x = self.get_denoised_and_G(target_model, x, t, s=t2, ctm=ctm, 
                #                                        **model_kwargs)
                # else:
                #     _, x = self.get_denoised_and_G(target_model, x, t, s=t_dt, ctm=ctm, 
                #                                    **model_kwargs)
                
            else:
                
                if use_x0_as_denoised_in_solver:
                    assert x_0 is not None
                    denoised = x_0
                else:
                    target_model.eval()
                    denoised, _ = self.get_denoised_and_G(target_model, x, t, t, ctm=ctm, **model_kwargs)
                
                assert not denoised.isnan().any()
                
                d = ddbm_to_d(x, t, denoised, x_T, self.args.sigma_max, w=self.args.guidance_scale, stochastic=False)
                x = x + (d * (t_dt - t))
                
                # print("is there the teacher model here?")
                # assert self.teacher_model is not None
                # print('yes!!! teacher model is here. But IDK how...')
                # exit()
                # self.teacher_model.eval()
                # for k in range(num_step):
                #     sigmas = self.karras_schedule(sigma_max=self.args.sigma_max, num_scales=num_scales)
                #     t = sigmas[ind + k]
                #     # t = self.get_t(ind + k, num_scales=num_scales)
                #     denoised, _ = self.get_denoised_and_G(self.teacher_model, x, t, None, ctm=False, teacher=True, **model_kwargs)
                #     d = (x - denoised) / append_dims(t, dims)
                #     # t2 = self.get_t(ind + k + 1, num_scales=num_scales)
                #     t2 = sigmas[ind + k + 1]
                #     x_phi_ODE_1st = x + d * append_dims(t2 - t, dims)
                #     denoised2, _ = self.get_denoised_and_G(self.teacher_model, x_phi_ODE_1st, t2, None, ctm=False, teacher=True, **model_kwargs)
                #     next_d = (x_phi_ODE_1st - denoised2) / append_dims(t2, dims)
                #     x_phi_ODE_2nd = x + (d + next_d) * append_dims((t2 - t) / 2, dims)
                #     x = x_phi_ODE_2nd
                
            return x
        
    # @th.no_grad()
    # def dpmpp_2nd_order_solver(self, target_model, x, ind, dims, t, t_dt, x_0=None, ctm=True, num_step=1, use_x0_as_denoised_in_solver=True, num_scales=None, sigmas=None, gamma=1.0, **model_kwargs):
    #     assert num_scales is not None
    #     assert sigmas is not None
    #     assert use_x0_as_denoised_in_solver
    #     assert (t == sigmas[ind]).all() 
    #     assert (t_dt == sigmas[ind + num_step]).all()
                    
    #     with th.no_grad():
    #         x_T: th.Tensor = model_kwargs['x_T']
    #         if self.args.self_learn:
    #             target_model.eval()
    #             if self.args.self_learn_iterative:
                    
    #                 for k in range(num_step):
    #                     # sigmas = self.karras_schedule(sigma_max=self.args.sigma_max, num_scales=num_scales)
    #                     t1 = append_dims(sigmas[ind + k], dims)
    #                     t2 = append_dims(sigmas[ind + k + 1], dims)
    #                     # NOTE: If gamma == 1, then it becomes Heun version of DPMSolver++ (2S)!
    #                     u = append_dims(t2 + ((t1 - t2) * (1. - gamma)), x.ndim)
    #                     assert th.all(t1 >= u) and th.all(u >= t2) and th.all(t1 > t2)
                        
    #                     l_t1, l_t2, l_u = self.get_lambda_t(t1), self.get_lambda_t(t2), self.get_lambda_t(u)
    #                     h = l_t2 - l_t1
    #                     h_0 = l_u - l_t1
    #                     phi = th.expm1(-h)      # == (t2/t1) - 1
    #                     phi_u = th.expm1(-h_0)  # == (u/t1) - 1
                        
    #                     r = h_0 / h # ORG
    #                     # r = (u - t1) / (t2 - t1)
            
    #                     if use_x0_as_denoised_in_solver:
    #                         assert x_0 is not None
    #                         denoised = x_0
    #                     else:
    #                         target_model.eval()
    #                         denoised, _ = self.get_denoised_and_G(target_model, x, t1, t1, ctm=ctm, **model_kwargs)
                        
    #                     assert not denoised.isnan().any()
                                                
    #                     D = denoised
                        
    #                     x = ((t2 / t1) * x) - (phi * D)
                        
    #         else:
                
    #             if use_x0_as_denoised_in_solver:
    #                 assert x_0 is not None
    #                 denoised = x_0
    #             else:
    #                 target_model.eval()
    #                 denoised, _ = self.get_denoised_and_G(target_model, x, t, t, ctm=ctm, **model_kwargs)
                
    #             assert not denoised.isnan().any()
                
    #             d = ddbm_to_d(x, t, denoised, x_T, self.args.sigma_max, w=self.args.guidance_scale, stochastic=False)
    #             x = x + (d * (t_dt - t))
                
    #         return x
        
    @th.no_grad()
    def euler_solver(self, target_model, x, ind, dims, t, t_dt, x_0=None, ctm=True, num_step=1, use_x0_as_denoised_in_solver=True, num_scales=None, sigmas=None, **model_kwargs):
        assert num_scales is not None
        assert sigmas is not None
        assert use_x0_as_denoised_in_solver
        assert (t == sigmas[ind]).all() 
        assert (t_dt == sigmas[ind + num_step]).all()
                    
        with th.no_grad():
            x_T: th.Tensor = model_kwargs['x_T']
            if self.args.self_learn:
                target_model.eval()
                if self.args.self_learn_iterative:
                    
                    for k in range(num_step):
                        # sigmas = self.karras_schedule(sigma_max=self.args.sigma_max, num_scales=num_scales)
                        t1 = append_dims(sigmas[ind + k], dims)
                        t2 = append_dims(sigmas[ind + k + 1], dims)
                        assert th.all(t1 > t2)
                        
                        if use_x0_as_denoised_in_solver:
                            assert x_0 is not None
                            denoised = x_0
                        else:
                            target_model.eval()
                            denoised, _ = self.get_denoised_and_G(target_model, x, t1, t1, ctm=ctm, **model_kwargs)
                        
                        assert not denoised.isnan().any()
                        
                        # d = (x - denoised) / append_dims(t, dims)
                        d = ddbm_to_d(x, t1, denoised, x_T, self.args.sigma_max, w=self.args.guidance_scale, stochastic=False)
                        assert not d.isnan().any()
                        x = x + (d * (t2 - t1))
                
            else:
                
                if use_x0_as_denoised_in_solver:
                    assert x_0 is not None
                    denoised = x_0
                else:
                    target_model.eval()
                    denoised, _ = self.get_denoised_and_G(target_model, x, t, t, ctm=ctm, **model_kwargs)
                
                assert not denoised.isnan().any()
                
                d = ddbm_to_d(x, t, denoised, x_T, self.args.sigma_max, w=self.args.guidance_scale, stochastic=False)
                x = x + (d * (t_dt - t))
                
            return x
        
    @th.no_grad()
    def euler_hybrid_solver(self, target_model, x, ind, dims, t, t_dt, x_0, ctm=True, num_step=1, num_scales=None, sigmas=None, **model_kwargs):
        assert num_scales is not None
        assert sigmas is not None
        with th.no_grad():
            if self.args.self_learn:
                target_model.eval()
                if self.args.self_learn_iterative:
                    for k in range(num_step):
                        t = sigmas[ind + k]
                        t2 = sigmas[ind + k + 1]
                        # t = self.get_t(ind + k, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)
                        # t2 = self.get_t(ind + k + 1, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)
                        
                        # 0703: Currently, this is solving by iterating the CTM model multiple times.
                        #       Rather, make it such that the DM model (in the CTM model) is used instead.
                        #       Therefore, we use the DM model's output to get the gradients and solve it using Heun's method.
                        #       Refer to GCTM code! It's done like that there!
                        #       BUT! The CM code seems to use x0 directly (so no "self_learn" at all!) That's also not bad.
                        _, x = self.get_denoised_and_G(target_model, x, t, s=t2, ctm=ctm, 
                                                       use_org_solver=True, **model_kwargs)
                else:
                    _, x = self.get_denoised_and_G(target_model, x, t, s=t_dt, ctm=ctm, 
                                                   use_org_solver=True, **model_kwargs)
                                   
            else:
     
                assert 'x_T' in model_kwargs.keys()
                x_T = model_kwargs['x_T']
     
                for k in range(num_step):
                    # t = self.get_t(ind + k, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)
                    sigmas = self.karras_schedule(sigma_max=self.args.sigma_max, n=num_scales)
                    t = sigmas[ind + k]
                    
                    if self.teacher_model is None:
                        denoised = x_0 # Similar to OpenAI's CM code.
                        print('yes!')
                        exit()
                    else:
                        self.teacher_model.eval()
                        denoised, _ = self.get_denoised_and_G(self.teacher_model, x, t, None, ctm=False, teacher=True, sigma_T=self.args.sigma_max, 
                                                            use_org_solver=True, **model_kwargs)
                    
                    d = fixed_hybrid_sample_to_d(
                        x=x,
                        sigma=t,
                        denoised=denoised,
                        x_T=x_T,
                        sigma_max=self.args.sigma_max,
                        w=self.args.guidance_scale,
                        stochastic=False,
                    )
                    
                    t2 = sigmas[ind + k + 1]
                    # t2 = self.get_t(ind + k + 1, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)
                    x = x + d * append_dims(t2 - t, dims)
                    
                    assert not x.isnan().any()
                    
            return x
        
    @th.no_grad()
    def contri_solver(self, target_model, x, ind, dims, t, t_dt, x_0=None, ctm=True, num_step=1, churn_step_ratio=None, num_scales=None, use_x0_as_denoised_in_solver=True, sigmas=None, **model_kwargs):
        assert self.args.self_learn == True
        assert sigmas is not None
        assert 'x_T' in model_kwargs.keys()
        assert churn_step_ratio is not None and 0. <= churn_step_ratio <= 1., "Pass valid 'churn_step_ratio' argument."
        x_T: th.Tensor = model_kwargs['x_T']
        assert use_x0_as_denoised_in_solver
        with th.no_grad():
            target_model.eval()
            assert (t == sigmas[ind]).all() 
            assert (t_dt == sigmas[ind + num_step]).all()
            if self.args.self_learn_iterative:
                for k in range(num_step):
                    t1 = append_dims(sigmas[ind + k], dims)
                    t2 = append_dims(sigmas[ind + k + 1], dims)
                    assert (t2 < t1).all()
                    
                    if churn_step_ratio > 0.0:
                        # 1-step Euler (SDE)
                        t1_hat: th.Tensor = ((t2 - t1) * churn_step_ratio) + t1
                        
                        assert (t1_hat != 0).any(), f"t1: {t1} | t2: {t2} | churn: {churn_step_ratio}" 
                        
                        if use_x0_as_denoised_in_solver:
                            assert x_0 is not None
                            denoised = x_0
                        else:
                            target_model.eval()
                            denoised, _ = self.get_denoised_and_G(target_model, x, t1, s=t1, ctm=ctm, 
                                                       use_org_solver=True,  x_T=x_T)
                                                    #    **model_kwargs)
                        
                        assert not denoised.isnan().any()
                        d, gt2 = ddbm_to_d(x, t1, denoised, x_T, self.args.sigma_max, w=self.args.guidance_scale, stochastic=True)
                        assert not d.isnan().any()
                        
                        dt: th.Tensor = t1_hat - t1
                        dw: th.Tensor = th.randn_like(x) * dt.abs().sqrt()
                        x = x + (d * dt) + (gt2.sqrt() * dw)
                    else:
                        t1_hat: th.Tensor = t1

                    # 1-step Euler (PF ODE)
                    if use_x0_as_denoised_in_solver:
                        assert x_0 is not None
                        denoised = x_0
                    else:
                        target_model.eval()
                        denoised, _ = self.get_denoised_and_G(target_model, x, t1_hat, s=t1_hat, ctm=ctm, 
                                                use_org_solver=True,  x_T=x_T)
                        # **model_kwargs)
                    
                    assert not x.isnan().any()
                    assert not denoised.isnan().any()
                    
                    d = ddbm_to_d(x, t1_hat, denoised=denoised, x_T=x_T, sigma_max=self.args.sigma_max, w=self.args.guidance_scale, stochastic=False)
                    assert not d.isnan().any()
                    
                    dt: th.Tensor = t2 - t1_hat
                    
                    x = x + (d * dt)
                    
                    # 0703: Currently, this is solving by iterating the CTM model multiple times.
                    #       Rather, make it such that the DM model (in the CTM model) is used instead.
                    #       Therefore, we use the DM model's output to get the gradients and solve it using Heun's method.
                    #       Refer to GCTM code! It's done like that there!
                    #       BUT! The CM code seems to use x0 directly (so no "self_learn" at all!) That's also not bad.
            else:
                if use_x0_as_denoised_in_solver:
                    assert x_0 is not None
                    denoised = x_0
                else:
                    target_model.eval()
                    denoised, _ = self.get_denoised_and_G(target_model, x, t, s=t, ctm=ctm, 
                                            use_org_solver=True, x_T=x_T)
                                            # **model_kwargs)
                
                assert not denoised.isnan().any()
                d = ddbm_to_d(x, t, denoised, x_T, self.args.sigma_max, w=self.args.guidance_scale, stochastic=False)
                
                x = x + (d * (t_dt - t))
                # _, x = self.get_denoised_and_G(target_model, x, t, s=t_dt, ctm=ctm, x_T=x_T,
                #                                 use_org_solver=True, 
                #                                 # **model_kwargs,
                #                             )
                
            assert not x.isnan().any()
            return x
        
    # @th.no_grad()
    # def hybrid_solver(self, target_model, x, ind, dims, t, t_dt, x_0, ctm=True, num_step=1, churn_step_ratio=None, num_scales=None, **model_kwargs):
    #     with th.no_grad():
    #         if self.args.self_learn:
    #             target_model.eval()
    #             if self.args.self_learn_iterative:
    #                 for k in range(num_step):
    #                     sigmas = self.karras_schedule(sigma_max=self.args.sigma_max, num_scales=num_scales)
    #                     t = sigmas[ind + k]
    #                     t2 = sigmas[ind + k + 1]
    #                     # t = self.get_t(ind + k, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)
    #                     # t2 = self.get_t(ind + k + 1, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)
                        
    #                     # 0703: Currently, this is solving by iterating the CTM model multiple times.
    #                     #       Rather, make it such that the DM model (in the CTM model) is used instead.
    #                     #       Therefore, we use the DM model's output to get the gradients and solve it using Heun's method.
    #                     #       Refer to GCTM code! It's done like that there!
    #                     #       BUT! The CM code seems to use x0 directly (so no "self_learn" at all!) That's also not bad.
    #                     _, x = self.get_denoised_and_G(target_model, x, t, s=t2, ctm=ctm, 
    #                                                    use_org_solver=True, **model_kwargs)
    #             else:
    #                 _, x = self.get_denoised_and_G(target_model, x, t, s=t_dt, ctm=ctm, 
    #                                                use_org_solver=True, **model_kwargs)
                    
    #         else:
    #             assert 'x_T' in model_kwargs.keys()
    #             assert churn_step_ratio is not None and 0. <= churn_step_ratio <= 1., "Pass valid 'churn_step_ratio' argument."
    #             x_T = model_kwargs['x_T']
    #             # print("is there the teacher model here?")
    #             # assert self.teacher_model is not None
    #             # print('yes!!! teacher model is here. But IDK how...')
    #             # exit()
                
    #             for k in range(num_step):
    #                 # t1 = self.get_t(ind + k, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)

    #                 # t2 = self.get_t(ind + k + 1, sigma_max=self.args.sigma_max-1e-4, num_scales=num_scales)
    #                 sigmas = self.karras_schedule(sigma_max=self.args.sigma_max, num_scales=num_scales)
    #                 t = sigmas[ind + k]
    #                 t2 = sigmas[ind + k + 1]
    #                 if k == 0:
    #                     assert th.all(t1 == t), "Error. Rename 't1' back to 't'."
    #                 assert th.all(t2 < t1)
                    
    #                 if churn_step_ratio > 0:
    #                     t1_hat = t1 + churn_step_ratio * (t2 - t1)
                        
    #                     if self.teacher_model is None:
    #                         denoised = x_0.clone() # Similar to OpenAI's CM code.
    #                         print('yes! same as openai cm')
    #                         exit()
    #                     else:
    #                         self.teacher_model.eval()
    #                         denoised, _ = self.get_denoised_and_G(self.teacher_model, x, t1, None, ctm=False, teacher=True, sigma_T=self.args.sigma_max, 
    #                                                             use_org_solver=True, **model_kwargs)
                        
    #                     d, gt2 = ddbm_to_d(x=x, sigma=t1, denoised=denoised, x_T=x_T, sigma_max=self.args.sigma_max, w=self.args.guidance_scale, stochastic=True)
    #                     dw = th.sqrt( append_dims(t1_hat - t1, dims).abs() ) * th.randn_like(x) 
    #                     x = x + (d * append_dims(t1_hat - t1, dims)) + (gt2.sqrt() * dw)
    #                     if self.args.use_milstein_method:
    #                         dgt = 1 / th.sqrt(2 * t1)
    #                         x += 0.5 * gt2.sqrt() * dgt * (dw.square() - append_dims(t1_hat - t1, dims))
    #                     assert not x.isnan().any()
    #                 else:
    #                     t1_hat = t1
    #                 # denoised, _ = self.get_denoised_and_G(self.teacher_model, x, t, None, ctm=False, teacher=True, **model_kwargs)
    #                 # denoised, _ = self.get_denoised_and_G(self.teacher_model, x, t, None, ctm=False, teacher=False, **model_kwargs)
                    
    #                 if self.teacher_model is None:
    #                     denoised = x_0.clone() # Similar to OpenAI's CM code.
    #                     print('yes! same as openai cm')
    #                     exit()
    #                 else:
    #                     self.teacher_model.eval()
    #                     denoised, _ = self.get_denoised_and_G(self.teacher_model, x, t1_hat, None, ctm=False, teacher=True, sigma_T=self.args.sigma_max, 
    #                                                         use_org_solver=True, **model_kwargs)
                    
    #                 # ---- make changes here such that fixed_hybrd.... is used! (06/08) (cz hybrid solver is being used, even in the case of teacher DM)
    #                 # d = (x - denoised) / append_dims(t, dims)
    #                 # d = fixed_hybrid_sample_to_d(
    #                 #     x=x,
    #                 #     sigma=t1,
    #                 #     denoised=denoised,
    #                 #     x_T=x_T,
    #                 #     sigma_max=self.args.sigma_max,
    #                 #     w=self.args.guidance_scale,
    #                 #     stochastic=False,
    #                 # )
                    
    #                 d = ddbm_to_d(x=x, sigma=t1_hat, denoised=denoised, x_T=x_T, sigma_max=self.args.sigma_max, w=self.args.guidance_scale, stochastic=False)
                    
    #                 dt = append_dims(t2 - t1_hat, dims)
    #                 # print()
    #                 # print('d: max', d.max().item(), 'min', d.min().item())
    #                 # print('d_second: max', d_second.max().item(), 'min', d_second.min().item())
    #                 # min_ratio = d.min() / d_second.min()
    #                 # max_ratio = d.max() / d_second.max()
    #                 # print('d/d_second:', f"min = {min_ratio.item()} | max = {max_ratio.item()}")
    #                 # print()
    #                 # exit()
                    
    #                 if th.all(t2 == 0):
    #                     x = x + (d * dt)
                        
    #                 else:
                        
    #                     x_phi_ODE_1st = x + (d * dt)
    #                     # denoised2, _ = self.get_denoised_and_G(self.teacher_model, x_phi_ODE_1st, t2, None, ctm=False, teacher=True, **model_kwargs)
    #                     # denoised2, _ = self.get_denoised_and_G(self.teacher_model, x_phi_ODE_1st, t2, None, ctm=False, teacher=False, **model_kwargs)
                        
    #                     # print('x_phi_ODE_1st: max', x_phi_ODE_1st.max().item(), 'min', x_phi_ODE_1st.min().item())
    #                     assert not x_phi_ODE_1st.isnan().any()
    #                     # exit()
    #                     # print()
    #                     # print('!'*20)
    #                     if self.teacher_model is None:
    #                         denoised2 = x_0.clone() # Similar to OpenAI's CM code.
    #                     else:
    #                         self.teacher_model.eval()
    #                         denoised2, _ = self.get_denoised_and_G(self.teacher_model, x_phi_ODE_1st, t2, None, ctm=False, teacher=True, sigma_T=self.args.sigma_max, 
    #                                                             use_org_solver=True, **model_kwargs)
                            
    #                     # next_d = (x_phi_ODE_1st - denoised2) / append_dims(t2, dims)
    #                     # next_d = fixed_hybrid_sample_to_d(
    #                     #     x=x_phi_ODE_1st,
    #                     #     denoised=denoised2,
    #                     #     sigma=t2,
    #                     #     x_T=x_T,
    #                     #     sigma_max=self.args.sigma_max,
    #                     #     w=self.args.guidance_scale,
    #                     #     stochastic=False,
    #                     # ) 
                        
    #                     next_d = ddbm_to_d(x=x_phi_ODE_1st, denoised=denoised2, sigma=t2, x_T=x_T, sigma_max=self.args.sigma_max, w=self.args.guidance_scale, stochastic=False) 
                        
    #                     # print()
    #                     # # print('next_d: max', next_d.max().item(), 'min', next_d.min().item())
    #                     # print('next_d2: max', next_d2.max().item(), 'min', next_d2.min().item())
    #                     # # min_ratio = next_d.min() / next_d2.min()
    #                     # # max_ratio = next_d.max() / next_d2.max()
    #                     # # print('next_d/next_d2:', f"min = {min_ratio.item()} | max = {max_ratio.item()}")
    #                     # # print()
    #                     # exit()
                        
    #                     # print(th.isnan(denoised2).any())
    #                     # exit()
    #                     # next_d = fixed_hybrid_sample_to_d(
    #                     #     x=x_phi_ODE_1st,
    #                     #     denoised=denoised2,
    #                     #     sigma=t2,
    #                     #     x_T=x_T,
    #                     #     sigma_max=self.args.sigma_max,
    #                     #     w=self.args.guidance_scale,
    #                     #     stochastic=False,
    #                     # )
    #                     # x = x + (d + next_d) * append_dims((t2 - t1) / 2, dims)
    #                     d_prime = 0.5 * (d + next_d) 
    #                     x = x + (d_prime * dt)
    #         return x
    
    def get_gan_fake(self, estimate, x_t, t, t_dt, s, model, target_model, ctm, step, **model_kwargs):
        if self.args.gan_fake_outer_type == 'no':
            _, fake = self.get_denoised_and_G(model, x_t, t, s=th.ones_like(s) * self.args.sigma_min, ctm=ctm, **model_kwargs)
        else:
            assert self.args.gan_fake_outer_type in ['model', 'target_model_sg']
            assert self.args.gan_fake_inner_type in ['model', 'model_sg', 'target_model_sg']
            if (self.args.gan_fake_outer_type == self.args.ctm_estimate_outer_type) \
                and (self.args.gan_fake_inner_type == self.args.ctm_estimate_inner_type):
                    fake = estimate
            else:
                fake = self.get_ctm_estimate(x_t, t, t_dt, s, model, target_model, ctm,
                                             outer_type=self.args.gan_fake_outer_type,
                                             inner_type=self.args.gan_fake_inner_type,
                                             target_matching=self.args.gan_target_matching, **model_kwargs)
        if self.args.gaussian_filter:
            fake = gaussian_blur(self.args, fake, step)
        return fake

    @th.no_grad()
    def get_gan_real(self, x_start, x_t, t, t_dt, s, indices, dims, num_heun_step, model, target_model, ctm, step, **model_kwargs):
        with th.no_grad():
            if self.args.gan_real_free:
                real = x_start
            else:
                x_t_dt = self.heun_solver(target_model, x_t, indices, dims, t, t_dt, ctm=ctm, num_step=num_heun_step,
                                          **model_kwargs).detach()
                #if self.args.gan_real_inner_type == 'no':
                #    _, real = self.get_denoised_and_G(target_model, x_t_dt, t_dt, s=th.ones_like(s) * self.args.sigma_min,
                #                                      ctm=ctm, **model_kwargs)
                #else:
                assert self.args.gan_real_inner_type in ['model_sg', 'target_model_sg', 'no']
                real = self.get_ctm_target(x_t_dt, t_dt, s, model, target_model, ctm,
                                           self.args.gan_real_inner_type, **model_kwargs)
            if self.args.gaussian_filter:
                real = gaussian_blur(self.args, real, step).detach()
        return real.detach()

    def get_ctm_estimate(self, x_t, t, t_dt, s, model, target_model, ctm, outer_type, inner_type, target_matching, **model_kwargs) -> th.Tensor:
        if not self.args.traditional_ctm:
            assert th.all(s == 0) # 20th aug
        if self.args.large_log:
            print("CTM estimate inner type, outer type, ctm: ", inner_type, outer_type, ctm)
        if target_matching:
            s = t_dt
        if inner_type == 'model': # <- inner (1)
            _, estimate = self.get_denoised_and_G(model, x_t, t, s=s, ctm=ctm, **model_kwargs)
        elif inner_type == 'model_sg':
            with th.no_grad():
                _, estimate = self.get_denoised_and_G(model, x_t, t, s=s, ctm=ctm, **model_kwargs)
        elif inner_type == 'target_model_sg':
            with th.no_grad():
                _, estimate = self.get_denoised_and_G(target_model, x_t, t, s=s, ctm=ctm, **model_kwargs)
        else:
            raise NotImplementedError
        
        if self.args.traditional_ctm:
            # 20th Aug (commented this out):
            if self.args.skip_final_ctm_step == False:
                if self.args.training_mode == 'ctm':
                    if outer_type == 'model':
                        raise NotImplementedError()
                        _, estimate = self.get_denoised_and_G(model, estimate, s, s=th.ones_like(s) * self.args.sigma_min,
                                                    ctm=ctm, **model_kwargs)
                    elif outer_type == 'target_model_sg': # <- outer (2)
                        # 20th Aug:
                        # _, estimate = self.get_denoised_and_G(target_model, estimate, s, s=th.ones_like(s) * self.args.sigma_min,
                        #                             ctm=ctm, **model_kwargs)
                        _, estimate = self.get_denoised_and_G(target_model, estimate, s, s=th.zeros_like(s),
                                                    ctm=ctm, **model_kwargs)
                    else:
                        raise NotImplementedError()
        return estimate

    @th.no_grad()
    def get_ctm_target(self, x_t_dt, t_dt, s, model, target_model, ctm, inner_type, **model_kwargs) -> th.Tensor:
        if not self.args.traditional_ctm:
            assert th.all(s == 0) # 20th aug
        assert inner_type == 'target_model_sg'
        if self.args.large_log:
            print("CTM target inner type, ctm: ", inner_type, ctm)
        with th.no_grad():
            if th.all(s == t_dt):
                target = x_t_dt.detach().clone()
            else:
                if inner_type == 'model_sg': # (July 3: NOT THIS ANYMORE) 
                    _, target = self.get_denoised_and_G(model, x_t=x_t_dt, t=t_dt, s=s, ctm=ctm, **model_kwargs)
                elif inner_type == 'target_model_sg': # (FROM July 3) <- inner (1) 
                    # UPDATE 19th aug: 'target_model_sg' seems to be correct.
                    _, target = self.get_denoised_and_G(target_model, x_t=x_t_dt, t=t_dt, s=s, ctm=ctm, **model_kwargs)
                elif inner_type == 'no':
                    target = x_t_dt
                    s = t_dt
                else:
                    raise NotImplementedError
            
            if self.args.traditional_ctm:
                # 20th Aug (commented this out):
                if self.args.skip_final_ctm_step == False:
                    if self.args.training_mode == 'ctm': # => cz target_model_sg <- outer (2)
                        # 20th Aug:
                        _, target = self.get_denoised_and_G(target_model, target, t=s, s=th.ones_like(s) * self.args.sigma_min, ctm=ctm, **model_kwargs)
                        # _, target = self.get_denoised_and_G(target_model, target, t=s, s=th.zeros_like(s), ctm=ctm, **model_kwargs)
            return target.detach()

    def get_denoised(self, g_theta, x_t, t):
        if self.args.outer_parametrization.lower() == 'euler':
            denoised = g_theta
        elif self.args.outer_parametrization.lower() == 'variance':
            denoised = g_theta + append_dims((self.args.sigma_min ** 2 + self.args.sigma_data ** 2
                                              - self.args.sigma_min * t) / \
                                             ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2),
                                             x_t.ndim) * x_t
        elif self.args.outer_parametrization.lower() == 'euler_variance_mixed':
            denoised = g_theta + x_t - append_dims(t / (t + 1.) * (1. + (t - self.args.sigma_min) /
                                                                   ((
                                                                                t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)),
                                                   x_t.ndim) * x_t
        else:
            raise NotImplementedError
        return denoised


    # def get_denoised_and_G(self, model, x_t, t, s=None, ctm=False, teacher=False, sigma_T=None, **model_kwargs):
    #     rescaled_t = self.rescaling_t(t)
    #     if s != None:
    #         rescaled_s = self.rescaling_t(s)
    #     else:
    #         rescaled_s = None
    #     c_in = append_dims(self.get_c_in(t), x_t.ndim)
    #     # print(model_kwargs.keys())
    #     # print('yes good')
    #     # print()
    #     # print('x_t', x_t.max().item(), x_t.min().item())
    #     # print('c_in', c_in.max().item(), c_in.min().item())
    #     # print('c_in * x_t', (c_in * x_t).max().item(), (c_in * x_t).max().item())
    #     if ctm and teacher:
    #         assert not model.module.teacher 
    #     model_output = model(c_in * x_t, rescaled_t, s=rescaled_s, teacher=teacher, **model_kwargs)
        
    #     # assert not x_t.isnan().any, f'{x_t}'
    #     # assert not c_in.isnan().any, f'{c_in}'
    #     # assert not model_output.isnan().any(), f'{model_output}'
    
    #     assert not model_output.isnan().any()

    #     if ctm: 
    #         if self.args.target_subtract:
    #             with th.no_grad():
    #                 teacher_denoised = self.teacher_model(c_in * x_t, rescaled_t, s=None, teacher=True, **model_kwargs)
    #             if self.args.rescaling:
    #                 model_output = model_output * append_dims((t ** 2 - s ** 2) ** 0.5 / t, x_t.ndim)
    #             model_output = model_output + teacher_denoised
                
    #         c_skip, c_out = [
    #             append_dims(x, x_t.ndim)
    #             for x in self.get_inner_scalings(t, self.args.inner_parametrization)
    #         ]
    #         g_theta = c_out * model_output + c_skip * x_t # = denoised in ddbm's denoise() function.
    #         #z = th.randn_like(x_t)
    #         #x_t_ = x_t + 0.001 * z
    #         #model_output_ = model(c_in * x_t_, rescaled_t, s=rescaled_s, teacher=teacher, **model_kwargs)
    #         #g_theta_ = c_out * model_output_ + c_skip * x_t
    #         #np.savez(bf.join(logger.get_dir(), f"g/{np.random.randint(10000000)}.npz"),
    #         #         {'x_t': x_t, 't': t, 's': s, 'g': g_theta, 'g_': g_theta_, 'z': z})
    #         denoised = self.get_denoised(g_theta, x_t, t) # 'denoised' is 'x_start' 
    #         # print('denoised', (denoised).max().item(), (denoised).min().item())
            
    #         # h_theta:
    #         # g_theta = g_theta + append_dims(t, x_t.ndim) * self.h_transform_t(x_t, t, model_kwargs, sigma_T)
            
    #         # # ------------------------------------------------------------------------------------
    #         # # 0523: just for now:
    #         # sigma_T = self.args.sigma_max
            
    #         # h_transform = self.h_transform_t(x_t, t, model_kwargs, sigma_T)
    #         # # print()
    #         # # print('c_out', (c_out).max().item(), (c_out).min().item())
    #         # # print('h_transform', (h_transform).max().item(), (h_transform).min().item())
    #         # s_min_t = append_dims(s-t, x_t.ndim)
    #         # # print('s_min_t', (s_min_t).max().item(), (s_min_t).min().item())
    #         # s_min_t_h_transform = s_min_t * h_transform
    #         # # print('s_min_t_h_transform', (s_min_t_h_transform).max().item(), (s_min_t_h_transform).min().item())
    #         # _diff = append_dims(th.ones_like(t) * sigma_T, x_t.ndim) - append_dims(t, x_t.ndim)
            
    #         # corrected_h_part = s_min_t_h_transform * _diff
    #         # corrected_h_part = corrected_h_part / append_dims(t, x_t.ndim)
    #         # # print('corrected_h_part', (corrected_h_part).max().item(), (corrected_h_part).min().item())
    #         # # ------------------------------------------------------------------------------------
            
    #         # ------------------------------------------------------------------------------------
    #         # print('g_theta', (g_theta).max().item(), (g_theta).min().item())
    #         # g_theta = g_theta + h_transform
    #         # print('new g_theta', (g_theta).max().item(), (g_theta).min().item())
    #         # if self.args.condition_mode == 'concat':
    #         #     assert 'x_T' in model_kwargs.keys()
    #         # denoised = denoised + append_dims(t, x_t.ndim) * self.h_transform_t(x_t, t, model_kwargs, sigma_T) # when h-transform is used, this needs to be done to make 'denoised' eq to 'x_start'.
    #         # ------------------------------------------------------------------------------------
            
    #         c_skip, c_out = [
    #             append_dims(x, x_t.ndim)
    #             for x in self.get_outer_scalings(t, s, self.args.outer_parametrization)
    #         ]
    #         G_theta = c_out * g_theta + c_skip * x_t
    #         # print('G_theta', (G_theta).max().item(), (G_theta).min().item())
    #         # G_theta = G_theta + corrected_h_part
    #         # G_theta = G_theta + append_dims(s-t, x_t.ndim) * self.h_transform_t(x_t, t, model_kwargs, sigma_T)
    #         # G_theta = G_theta + append_dims(s*(s-t)/t, x_t.ndim) * self.h_transform_t(x_t, t, model_kwargs, sigma_T)
    #         # print('new G_theta', (G_theta).max().item(), (G_theta).min().item())
    #     else:
    #         # assert sigma_T is not None
            
    #         if teacher:
    #             c_skip, c_out = [
    #                 append_dims(x, x_t.ndim) for x in self.get_edm_scalings(t)
    #             ]
    #         else:
    #             c_skip, c_out = [
    #                 append_dims(x, x_t.ndim)
    #                 for x in self.get_cm_scalings(t)
    #                 # for x in self.get_cm_ddbm_scalings(t)
    #             ]
            
    #         # print('model_output', model_output)
    #         denoised = c_out * model_output + c_skip * x_t
            
    #         # # if self.args.condition_mode == 'concat':
    #         # # assert 'x_T' in model_kwargs.keys()
    #         # # print('max', denoised.max(), 'min', denoised.min())
    #         # print('denoised (before):')
    #         # print('max', denoised.max(), 'min', denoised.min())
    #         # denoised = denoised + t * self.h_transform_t(x_t, t, model_kwargs, sigma_T)
    #         # print('denoised (after):')
    #         # # print('after:')
    #         # print('max', denoised.max(), 'min', denoised.min())
    #         G_theta = denoised
    #     return denoised, G_theta

    def get_denoised_and_G(self, model, x_t, t, s=None, ctm=False, teacher=False, sigma_T=None, is_sampling=False, x_T=None, 
                           use_org_solver=True, **model_kwargs):
        # with th.autograd.detect_anomaly():
        assert x_T is not None # or ('x_T' in model_kwargs.keys() and model_kwargs['x_T'] is not None)
        assert model is not None, "The model cannot be None."
        
        # is_nan = th.stack([th.isnan(p).any() or th.isinf(p).any() for p in model.parameters()]).any()
        # assert not is_nan
        # exit()
        c_in = append_dims(self.get_c_in(t, self.args.inner_parametrization), x_t.ndim)
        # print(model_kwargs.keys())
        # print('yes good')
        # if rescaled_s is not None and th.equal(rescaled_s, rescaled_t):
        # if c_in.max().item() == 2 and c_in.min().item() == 2:
        # print()
        # if is_sampling:
        #     print('x', x_t.max().item(), x_t.min().item())
        #     print('c_in', c_in.max().item(), c_in.min().item())
        #     print('c_in * x', (c_in * x_t).max().item(), (c_in * x_t).min().item())
        #     print('t', t.max().item(), t.min().item())
        #     print('s', s.max().item(), s.min().item())
        #     print()
        if ctm and teacher:
            assert not model.module.teacher 
        
        rescaled_t = self.rescaling_t(t)
        if s != None:
            rescaled_s = self.rescaling_t(s)
        else:
            rescaled_s = None
        
        # TODO: Multiply x_T with x_T's re-normalization constant 
        k_in = self.get_k_in()
        # assert not c_in.isnan().any()
        model_output = model(c_in * x_t, rescaled_t, rescaled_s, teacher=teacher, 
                            x_T=k_in*x_T if x_T is not None else None, **model_kwargs)
        assert not model_output.isnan().any()
        # assert not x_t.isnan().any, f'{x_t}'
        # assert not c_in.isnan().any, f'{c_in}'
        # assert not model_output.isnan().any(), f'{model_output}'
    
        # assert not model_output.isnan().any()
        
        assert ctm == True

        if ctm:            
            if self.args.target_subtract:
                with th.no_grad():
                    teacher_denoised = self.teacher_model(c_in * x_t, rescaled_t, s=None, teacher=True, **model_kwargs)
                if self.args.rescaling:
                    model_output = model_output * append_dims((t ** 2 - s ** 2) ** 0.5 / t, x_t.ndim)
                model_output = model_output + teacher_denoised
                
            c_skip, c_out = [
                append_dims(x, x_t.ndim)
                for x in self.get_inner_scalings(t, self.args.inner_parametrization)
            ]
            
            # c_skip2, c_out2 = [
            #     append_dims(x, x_t.ndim)
            #     for x in self.get_inner_scalings(t, 'cm_ddbm')
            # ]
            # c_in2 = append_dims(self.get_c_in(t, 'cm_ddbm'), x_t.ndim)
            # print('diff_c_in:', th.allclose(c_in , c_in2))
            # print('diff_c_skip:', th.allclose(c_skip , c_skip2))
            # print('diff_c_out:', th.allclose(c_out, c_out2))
            # exit()
            
            assert not c_skip.isinf().any().item(), f'sigma0: {self.args.sigma_data} | t: {t} | sigmaT: {self.args.sigma_max}'
            assert not c_out.isinf().any().item(), f'sigma0: {self.args.sigma_data} | t: {t} | sigmaT: {self.args.sigma_max}'
            
            # ------------------------------------------------------------------------------------
            if use_org_solver:
                # ORG:
                g_theta = c_out * model_output + c_skip * x_t # = denoised in ddbm's denoise() function.
                
            else:
                # Using h-transform:
                D_theta = c_out * model_output + c_skip * x_t
                
                b_theta = self.q_part_t(D_theta, x_t, t, x_T, sigma_max=sigma_T)
                h_transform_t = self.h_transform_t(x_t, t, x_T, sigma_max=sigma_T)
                g_theta = x_t + h_transform_t + b_theta
                
            # ------------------------------------------------------------------------------------
            
            #z = th.randn_like(x_t)
            #x_t_ = x_t + 0.001 * z
            #model_output_ = model(c_in * x_t_, rescaled_t, s=rescaled_s, teacher=teacher, **model_kwargs)
            #g_theta_ = c_out * model_output_ + c_skip * x_t
            #np.savez(bf.join(logger.get_dir(), f"g/{np.random.randint(10000000)}.npz"),
            #         {'x_t': x_t, 't': t, 's': s, 'g': g_theta, 'g_': g_theta_, 'z': z})
            
            denoised = self.get_denoised(g_theta, x_t, t) # 'denoised' is 'x_start' 
            
            # denoised2 = self.get_denoised(D_theta, x_t, t) # 'denoised' is 'x_start' 
            # is_same = g_theta - D_theta
            # print('is same:')
            
            # print('denoised', (denoised).max().item(), (denoised).min().item())

            c_skip, c_out = [
                append_dims(x, x_t.ndim)
                for x in self.get_outer_scalings(t, s, self.args.outer_parametrization)
            ]
            assert not c_skip.isinf().any().item(), f's: {s} | t: {t}'
            assert not c_out.isinf().any().item(), f's: {s} | t: {t}'
            assert not g_theta.isinf().any().item()
            assert not x_t.isinf().any().item()
            
            G_theta = c_out * g_theta + c_skip * x_t
            # print('G_theta', (G_theta).max().item(), (G_theta).min().item())
        else:
            # assert sigma_T is not None
            print('should not happen!')
            exit()
            if teacher:
                c_skip, c_out = [
                    append_dims(x, x_t.ndim) for x in self.get_edm_scalings(t)
                ]
            else:
                c_skip, c_out = [
                    append_dims(x, x_t.ndim)
                    for x in self.get_cm_scalings(t)
                    # for x in self.get_cm_ddbm_scalings(t)
                ]
            
            denoised = c_out * model_output + c_skip * x_t
            
            G_theta = denoised
        
        # # a = c_skip * x_t
        # # b = c_out * g_theta
        # print('c_skip', c_skip.isinf().any().item())
        # # print('x_t', x_t.isinf().any().item())
        # print('c_out', c_out.isinf().any().item())
        assert not c_out.isinf().any().item(), f'sigma0: {self.args.sigma_data} | t: {t} | sigmaT: {self.args.sigma_max}'
        assert not c_skip.isinf().any().item(), f'sigma0: {self.args.sigma_data} | t: {t} | sigmaT: {self.args.sigma_max}'
        # print('g_theta', g_theta.isinf().any().item())
        # print('a', a.max(), a.min())
        # print('b', b.max(), b.min())
        # print('a+b', (a+b).max(), (a+b).min())
        # assert not G_theta.isnan().any()
        return denoised, G_theta
    
    def q_part_t(self, D_theta, x_t, t, x_T, sigma_max):
        assert x_T is not None
        sigma_max = self.args.sigma_max if sigma_max is None else sigma_max
        
        sigma_max_sq = append_dims(th.ones_like(t)*sigma_max**2, x_t.ndim)
        sigma_t_sq = append_dims(t**2, x_t.ndim)
        
        _denom = sigma_max_sq - sigma_t_sq
        assert not (th.any(_denom.isnan()) or th.any(_denom == 0))
        
        _first = sigma_max_sq * (x_t - D_theta)
        _second = sigma_t_sq * (x_T - D_theta)
    
        q_part_t = -1 * (_first - _second) / _denom
        return q_part_t
        
    def h_transform_t(self, x_t, t, x_T, sigma_max):
        assert x_T is not None
        w = np.float32(self.args.guidance_scale)
        sigma_max = self.args.sigma_max if sigma_max is None else sigma_max
        
        # _denom = append_dims(th.ones_like(t)*sigma_T**2, x_t.ndim) - append_dims(t**2, x_t.ndim)
        # # print(f'_denom: max = {_denom.max().item()}, min = {_denom.min().item()}')
        
        # assert not (th.any(_denom.isnan()) or th.any(_denom == 0))
        
        # _diff = x_T - x_t
        # # print(f'_diff: max = {_diff.max().item()}, min = {_diff.min().item()}')
        # assert not _diff.isnan().any()
        
        # # print(f't**2: max = {(t**2).max().item()}, min = {(t**2).min().item()}')
        
        # # _nearly_h_part = append_dims(t**2, x_t.ndim) * (x_T - x_t) / _denom
        # # print(f'_nearly_h_part: max = {_nearly_h_part.max().item()}, min = {_nearly_h_part.min().item()}')
        
        # # print('w:', w)
        
        # # # h_part = w * (t**2) * (x_T - x_t) / (sigma_T**2 - t**2)
        grad_pxTlxt = self._get_grad_pxTlxt(x_t, t, x_T, sigma_max)
        h_part = - 2 * w * append_dims(t**2, x_t.ndim) * grad_pxTlxt
        
        return h_part
    
    def _get_grad_pxTlxt(self, x_t, t, x_T, sigma_max):
        
        _denom = append_dims(th.ones_like(t)*sigma_max**2, x_t.ndim) - append_dims(t**2, x_t.ndim)
        assert not (th.any(_denom.isnan()) or th.any(_denom == 0))
        
        _diff = x_T - x_t
        assert not _diff.isnan().any(), f"-- x_T has nan: {x_T.isnan().any()}\n-- x_t has nan: {x_t.isnan().any()}"
        
        grad_pxTlxt = _diff / _denom
        return grad_pxTlxt

    def get_CTM_loss(self, estimate, target, weights, step):
        if self.args.loss_norm == 'pseudo-huber':
            dim = estimate[0].flatten().shape[0]
            c = 0.00054 * np.sqrt(dim)
            consistency_loss = th.sqrt((estimate - target) ** 2 + c**2) - c
            assert th.all(weights > 0)
            consistency_loss = weights * consistency_loss
        elif self.args.loss_norm == 'lpips':
            if estimate.shape[-2] < 256:
                estimate = F.interpolate(estimate, size=224, mode="bilinear")
                target = F.interpolate(
                    target, size=224, mode="bilinear"
                )
            consistency_loss = (self.feature_extractor(
                (estimate + 1) / 2.0,
                (target + 1) / 2.0, ) * weights)
        # elif self.args.loss_norm == "cnn_vit":
        #     distances, estimate_features, target_features = [], [], []
        #     estimate_features, target_features = get_xl_feature(self.args, estimate, target,
        #                                                              feature_extractor=self.feature_extractor, step=step)
        #     cnt = 0
        #     for _, _ in self.feature_extractor.items():
        #         for fe in list(estimate_features[cnt].keys()):
        #             norm_factor = th.sqrt(th.sum(estimate_features[cnt][fe] ** 2, dim=1, keepdim=True))
        #             est_feat = estimate_features[cnt][fe] / (norm_factor + 1e-10)
        #             norm_factor = th.sqrt(th.sum(target_features[cnt][fe] ** 2, dim=1, keepdim=True))
        #             tar_feat = target_features[cnt][fe] / (norm_factor + 1e-10)
        #             distances.append(self.dist(est_feat, tar_feat))
        #         cnt += 1
        #     consistency_loss = th.cat([d.mean(dim=[2, 3]) for d in distances], dim=1).sum(dim=1)
        else:
            raise NotImplementedError
        
        return consistency_loss

    def get_DSM_loss(self, model, x_start, model_kwargs, consistency_loss,
                           step, init_step):
        sigmas, denoising_weights = self.diffusion_schedule_sampler.sample(x_start.shape[0], dist_util.dev())
        noise = th.randn_like(x_start)
        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        denoised, _ = self.get_denoised_and_G(model, x_t, sigmas, s=sigmas, ctm=True, teacher=True, **model_kwargs)
        snrs = self.get_snr(sigmas)
        denoising_weights = append_dims(self.get_weightings(self.args.diffusion_weight_schedule, snrs, self.args.sigma_data, None, None), dims)
        denoising_loss = mean_flat(denoising_weights * (denoised - x_start) ** 2)
        if self.args.apply_adaptive_weight:
            if self.args.data_name.lower() == 'cifar10':
                balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), denoising_loss.mean(),
                                                                last_layer=model.module.model.dec[
                                                                    '32x32_aux_conv'].weight)
            else:
                balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), denoising_loss.mean(),
                                                                last_layer=
                                                                model.module.output_blocks[15][0].out_layers[
                                                                    3].weight)
        else:
            balance_weight = 1.
        # print('balance', balance_weight)
        # exit()
        if self.args.large_log:
            logger.log("denoising weight: ", balance_weight)
        balance_weight = self.adopt_weight(balance_weight, step, threshold=init_step, value=1.)
        denoising_loss = denoising_loss * balance_weight
        return denoising_loss
    
    def get_DBSM_loss(self, model, x_start, noise, model_kwargs, consistency_loss,
                           step, init_step, ctm_sigma_t=None, ctm_xt=None,
                           use_bridge_sample=True, sigma_T=None):
        
        assert noise is not None
        assert 'x_T' in model_kwargs.keys() and model_kwargs['x_T'] is not None
        x_T = model_kwargs['x_T'].clone()
        
        dims = x_start.ndim
        assert sigma_T is not None
        
        # 0908:
        # if ctm_sigma_t is None:
        sigma_t, denoising_weights = self.diffusion_schedule_sampler.sample(x_start.shape[0], dist_util.dev())
        while (sigma_t > sigma_T).any():
            sigma_t, denoising_weights = self.diffusion_schedule_sampler.sample(x_start.shape[0], dist_util.dev())
        
        assert th.all( ((sigma_t**2) / (sigma_T**2)) <= 1. )
        # 0908:
        if use_bridge_sample:
            x_t = self.bridge_sample(x0=x_start, x_T=x_T, sig_t=sigma_t, 
                                        gaussian_noise=noise, sig_max=sigma_T,)
        else:
            x_t = x_start + (noise * append_dims(sigma_t, dims))
        # else:
        #     # Use the same xt here as used for the CTM Loss.
        #     assert ctm_xt is not None 
            
        #     sigma_t = ctm_sigma_t
        #     assert th.all((1. - (sigma_t**2/sigma_T**2)) >= 0.)
        #     x_t = ctm_xt.clone()
        
        denoised, _ = self.get_denoised_and_G(model, x_t, t=sigma_t, s=sigma_t, ctm=True, 
                                              teacher=False, sigma_T=sigma_T, **model_kwargs)
        snrs = self.get_snr(sigma_t)
        denoising_weights = append_dims(self.get_weightings(self.args.diffusion_weight_schedule, snrs, self.args.sigma_data, None, None), dims)
        
        denoising_loss = mean_flat(denoising_weights * ((denoised - x_start) ** 2))
        # denoising_loss = denoising_weights * th.nn.functional.mse_loss(denoised, x_start, reduction='none')

        if self.args.apply_adaptive_weight:
            if self.args.data_name.lower() == 'cifar10':
                balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), denoising_loss.mean(),
                                                                last_layer=model.module.model.dec[
                                                                    '32x32_aux_conv'].weight)
            else:
                balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), denoising_loss.mean(),
                                                                last_layer=
                                                                model.module.output_blocks[15][0].out_layers[
                                                                    3].weight)
        else:
            balance_weight = 1.
        # print('balance', balance_weight)
        # print('denoising_loss', denoising_loss.mean().item())
        # exit()
        if self.args.large_log:
            logger.log("bridge denoising weight: ", balance_weight)
        balance_weight = self.adopt_weight(balance_weight, step, threshold=init_step, value=1.)
        # print('balance_weight', balance_weight)
        # exit()
        denoising_loss = denoising_loss * balance_weight
        # print('denoising_loss', denoising_loss)
        # exit()
        return denoising_loss, x_t, denoised, sigma_t, denoising_weights

    def get_q_part_loss(self, model, x_start, x_t, denoised, denoising_weights, sigma_t, model_kwargs, consistency_loss, step, init_step, sigma_T=None):
        x_T = model_kwargs['x_T']
        
        dims = x_start.ndim
        # sigma_t = append_dims(sigma_t, dims)
        # sigma_T = append_dims(th.ones_like(sigma_t)*sigma_T, dims)
        
        sigma_max_sq = append_dims(th.ones_like(sigma_t)*sigma_T**2, dims)
        sigma_t_sq = append_dims(sigma_t**2, dims)
        
        _denom = sigma_max_sq - sigma_t_sq
        # print('_denom:', _denom.mean().item())

        actual_q_part = (-1/_denom) * (sigma_max_sq * (x_t - x_start) - sigma_t_sq * (x_T - x_start))
        # print('actual_q_part:', actual_q_part.mean().item())
        
        h_part = self.h_transform_t(x_t, sigma_t, x_T, sigma_T)
        q_part = denoised - x_t - h_part
        
        q_part_loss = mean_flat(denoising_weights * ((actual_q_part - q_part) ** 2))

        if self.args.apply_adaptive_weight:
            if self.args.data_name.lower() == 'cifar10':
                balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), q_part_loss.mean(),
                                                                last_layer=model.module.model.dec[
                                                                    '32x32_aux_conv'].weight)
            else:
                balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), q_part_loss.mean(),
                                                                last_layer=
                                                                model.module.output_blocks[15][0].out_layers[
                                                                    3].weight)
        else:
            balance_weight = 1.
        # print('balance', balance_weight)
        # print('q_part_loss:', q_part_loss.mean().item())
        # exit()
        if self.args.large_log:
            logger.log("q_part denoising weight: ", balance_weight)
        
        if self.args.apply_adaptive_weight:
            balance_weight = self.adopt_weight(balance_weight, step, threshold=init_step, value=1.)
        q_part_loss = q_part_loss * balance_weight
        # print('balance_weight', balance_weight)
        # print('q_part_loss:', q_part_loss.mean().item())
        # exit()
        return q_part_loss

    def get_GAN_loss(self, model, real=None, fake=None, consistency_loss=None,
                               learn_generator=True, discriminator=None, step=0, init_step=0, **model_kwargs):

        if learn_generator:
            logits_fake = get_xl_feature(self.args, fake, feature_extractor=self.discriminator_feature_extractor,
                                                  discriminator=discriminator, **model_kwargs)
            g_loss = sum([(-l).mean() for l in logits_fake]) / len(logits_fake)
            if self.args.large_log:
                logger.log("g_loss: ", g_loss.mean().item())
            if self.args.d_apply_adaptive_weight:
                CTM_loss = consistency_loss.mean()
                if self.args.data_name.lower() == 'cifar10':
                    d_weight = self.calculate_adaptive_weight(CTM_loss.mean(),
                                                              g_loss.mean(),
                                                              last_layer=model.module.model.dec[
                                                                  '32x32_aux_conv'].weight)
                else:
                    d_weight = self.calculate_adaptive_weight(CTM_loss.mean(),
                                                              g_loss.mean(),
                                                              last_layer=
                                                              model.module.output_blocks[15][0].out_layers[
                                                                  3].weight)
                d_weight = th.clip(d_weight, 0.01, 10.)
            else:
                d_weight = 1.
            discriminator_loss = self.adopt_weight(d_weight, step,
                                                   threshold=init_step + self.args.discriminator_start_itr) * g_loss
        else:
            logits_fake, logits_real = get_xl_feature(self.args, fake.detach(), target=real.detach(),
                                                      feature_extractor=self.discriminator_feature_extractor,
                                                      discriminator=discriminator, step=step, **model_kwargs)
            loss_Dgen = sum([(F.relu(th.ones_like(l) + l)).mean() for l in logits_fake]) / len(logits_fake)
            loss_Dreal = sum([(F.relu(th.ones_like(l) - l)).mean() for l in logits_real]) / len(logits_real)
            discriminator_loss = loss_Dreal + loss_Dgen
            if self.args.large_log:
                print("logits_real: ", sum([l.mean() for l in logits_real]).item() / len(logits_real), len(logits_real))
                print("logits_fake: ", sum([l.mean() for l in logits_fake]).item() / len(logits_fake), len(logits_fake))
        return discriminator_loss

    def ctm_losses(
        self,
        step,
        model,
        x_start,
        model_kwargs=None,
        target_model=None,
        noise=None,
        discriminator=None,
        init_step=0,
        ctm=True,
        num_heun_step=-1,
        gan_num_heun_step=-1,
        diffusion_training_=False,
        gan_training_=False,
        sigma_T=None,
        current_timestep=None,
        num_scales=None,
    ):
        assert num_scales is not None
        assert current_timestep is not None
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        dims = x_start.ndim
        s = None
        terms = {}
        if num_heun_step == -1:
            num_heun_step = [self.get_num_heun_step(num_heun_step=self.args.num_heun_step)]
            dist.broadcast_object_list(num_heun_step, 0)
            num_heun_step = num_heun_step[0]
        if self.args.large_log:
            print("x batch size: ", x_start.shape)
            print("num heun step: ", num_heun_step)
            print("diffusion training: ", diffusion_training_)
            print("gan training: ", gan_training_)
        
        sigmas = self.karras_schedule(sigma_max=self.args.sigma_max, n=num_scales)
        # indices, _ = self.schedule_sampler.sample_t(self.args, x_start.shape[0], x_start.device, num_heun_step,
        #                                             self.args.time_continuous)
        indices = self.schedule_sampler.sample_t(self.args, batch_size=x_start.shape[0], device=x_start.device, num_heun_step=num_heun_step,
                                                 sigmas=sigmas, num_scales=num_scales)
        t = self.get_t(indices, num_scales=current_timestep)
        t_dt = self.get_t(indices + num_heun_step, num_scales=current_timestep)
        if ctm:
            new_indices = self.schedule_sampler.sample_s(self.args, x_start.shape[0], x_start.device, indices,
                                                         num_heun_step, self.args.time_continuous,
                                                         N=num_scales)
            s = self.get_t(new_indices)
        x_t = x_start + noise * append_dims(t, dims)
        dropout_state = th.get_rng_state()
        th.set_rng_state(dropout_state)
        ctm_estimate = self.get_ctm_estimate(x_t, t, t_dt, s, model, target_model, ctm=ctm,
                                             outer_type=self.args.ctm_estimate_outer_type,
                                             inner_type=self.args.ctm_estimate_inner_type,
                                             target_matching=self.args.ctm_target_matching,
                                             **model_kwargs)
                
        if step % self.args.g_learning_period == 0 or not self.args.gan_training:
            assert (discriminator == None) == (self.args.g_learning_period == 1)
            x_t_dt = self.heun_solver(target_model, x_t, indices, dims, t, t_dt, ctm=ctm, num_step=num_heun_step,
                                      **model_kwargs).detach()
            ctm_target = self.get_ctm_target(x_t_dt, t_dt, s, model, target_model, ctm=ctm,
                                             inner_type=self.args.ctm_target_inner_type, **model_kwargs)
            
            if self.args.save_png and step % self.args.save_period == 0:
                img_save_dir = logger.get_dir() + '/images/'
                script_util.save(ctm_estimate, img_save_dir, f'ctm_estimate_{step}')  # _{r}')
                script_util.save(ctm_target, img_save_dir, f'ctm_target_{step}')  # _{r}')
                script_util.save(x_t, img_save_dir, f'xt_NonDenoised_{step}')  # _{r}')
                script_util.save(x_t_dt, img_save_dir, f'xu_Solved_{step}')  # _{r}')

            snrs = self.get_snr(t)
            weights = self.get_weightings(self.args.weight_schedule, snrs, self.args.sigma_data, t, s, self.args.weight_schedule_multiplier)

            terms["consistency_loss"] = self.get_CTM_loss(ctm_estimate, ctm_target, weights, step - init_step,)
            if self.args.large_log:
                if s != None:
                    print(f"{step}-th step, t, t-dt, s, weight, loss: ", t[0].item(), t_dt[0].item(), s[0].item(), weights[0].item(), terms["consistency_loss"][0].item())
                else:
                    print(f"{step}-th step, t, t-dt, weight, loss: ", t[0].item(), t_dt[0].item(), terms["consistency_loss"][0].item(), weights[0].item())
            if self.args.diffusion_training:
                if diffusion_training_:
                    terms['denoising_loss'] = self.get_DSM_loss(model, x_start, model_kwargs,
                                                                      terms["consistency_loss"],
                                                                      step, init_step)

            if self.args.gan_training and step - init_step >= self.args.discriminator_start_itr:
                if gan_training_:
                    gan_x_t, gan_t, gan_t_dt, gan_s, _, _ = self.get_gan_time(x_start, noise, x_t, t, t_dt, s, indices,
                                                                              num_heun_step, gan_num_heun_step)
                    gan_fake = self.get_gan_fake(ctm_estimate, gan_x_t, gan_t, gan_t_dt, gan_s, model, target_model, ctm,
                                                 step - init_step, **model_kwargs)
                    terms['d_loss'] = self.get_GAN_loss(model, fake=gan_fake,
                                                                  consistency_loss=terms["consistency_loss"],
                                                                  discriminator=discriminator,
                                                                  step=step, init_step=init_step)
        else:
            gan_x_t, gan_t, gan_t_dt, gan_s, gan_indices, gan_num_heun_step = \
                self.get_gan_time(x_start, noise, x_t, t, t_dt, s, indices, num_heun_step, gan_num_heun_step)
            gan_real = self.get_gan_real(x_start, gan_x_t, gan_t, gan_t_dt, gan_s, gan_indices, dims, gan_num_heun_step,
                                         model, target_model, ctm, step - init_step, **model_kwargs)
            gan_fake = self.get_gan_fake(ctm_estimate, gan_x_t, gan_t, gan_t_dt, gan_s, model, target_model, ctm,
                                         step - init_step, **model_kwargs)
            terms['d_loss'] = self.get_GAN_loss(model, fake=gan_fake, real=gan_real,
                                                learn_generator=False, discriminator=discriminator,
                                                step=step, init_step=init_step, **model_kwargs)
        return terms

    def bridge_ctm_losses(
        self,
        step,
        model,
        x_start,
        model_kwargs=None,
        target_model=None,
        noise=None,
        discriminator=None,
        init_step=0,
        ctm=True,
        num_heun_step=-1,
        gan_num_heun_step=-1,
        diffusion_training_=False,
        gan_training_=False,
        sigma_max=80.,
        current_timestep=None,
        num_scales=None,
    ):
        
        # for params, params2 in zip(model.parameters(), target_model.parameters()):
        #     assert th.all(params == params2)
        # if step % 100 == 0:
        #     print('good!')
        # exit()
        assert current_timestep is not None
        assert num_scales is not None
        # if model_kwargs is None:
        #     model_kwargs = {}
        # for n1, p1 in target_model.named_parameters():
        #     for n2, p2 in model.module.named_parameters():
        #         assert n1 == n2, f"{n1}, {n2}"
        #         targetparam = p1.data.detach().cpu().reshape(-1)[:3]
        #         normalparam = p2.data.detach().cpu().reshape(-1)[:3]
        #         if (targetparam != normalparam).any():
        #             print('target - normal:', targetparam - normalparam)
        #         break
        #     break
            
        assert model_kwargs is not None
        assert 'x_T' in model_kwargs.keys() and model_kwargs['x_T'] is not None
        
        if num_heun_step >= num_scales:
            # num_heun_step = num_scales - 1
            # num_heun_step = num_scales // 2
            num_heun_step = np.random.randint(low=1, high=num_scales,) # 21 Aug
        assert num_heun_step >= 1

        # x_T = model_kwargs['x_T']
        
        if noise is None:
            noise = th.randn_like(x_start)
        dims = x_start.ndim
        s = None
        terms = {}
        if num_heun_step == -1:
            num_heun_step = [self.get_num_heun_step(num_heun_step=self.args.num_heun_step)]
            dist.broadcast_object_list(num_heun_step, 0)
            num_heun_step = num_heun_step[0]
        if self.args.large_log:
            print("x batch size: ", x_start.shape)
            print("num heun step: ", num_heun_step)
            print("diffusion training: ", diffusion_training_)
            print("gan training: ", gan_training_)
            
        sigmas = self.karras_schedule(sigma_max=sigma_max, n=num_scales).to(noise.device)

        # indices, _ = self.schedule_sampler.sample_t(self.args, x_start.shape[0], x_start.device, num_heun_step,
        #                                             self.args.time_continuous, discretized_sigmas, )
        indices = self.schedule_sampler.sample_t(self.args, batch_size=x_start.shape[0], device=x_start.device, num_heun_step=num_heun_step,
                                                 sigmas=sigmas, num_scales=num_scales)
        
        if isinstance(indices, tuple):
            indices = indices[0]
            
        # print('indices', indices)
        # print('sigmas[indices]', sigmas[indices])
        # print('sigmas[indices + num_heun_step]', sigmas[indices + num_heun_step])
        # exit()
        
        # t = self.get_t(indices, sigma_max=sigma_max, num_scales=num_scales) # It's ok to provide a value that is < sigma_T (for ex. sigma_max)
        # t_dt = self.get_t(indices + num_heun_step, sigma_max=sigma_max, num_scales=num_scales) # It's ok to provide a value that is < sigma_T (for ex. sigma_max)
        
        t = sigmas[indices]
        assert (t != 0).all()
        t_dt = sigmas[indices + num_heun_step]
        # print('t:', t)
        # print('u:', t_dt)
        # print('indices:', indices)
        # print('num_heun_step:', num_heun_step)
        # print('sigmas shape', sigmas.shape, num_scales)
        # exit()
        assert (t_dt < t).all()
        if ctm:
            new_indices = self.schedule_sampler.sample_s(self.args, x_start.shape[0], x_start.device, indices,
                                                         num_heun_step, self.args.time_continuous,
                                                         N=num_scales)
            # s = self.get_t(new_indices, sigma_max=sigma_max, num_scales=num_scales) # It's ok to provide a value that is < sigma_T (for ex. sigma_max)
            # print('new_indices:', new_indices)
            # print('sigmas:', sigmas)
            # print('u:', t_dt)
            # print('s:', sigmas[new_indices])
            # exit()
            s = sigmas[new_indices]
            if not self.args.traditional_ctm:
                assert th.all(s == 0) # 20th aug
            # assert (s < t_dt).all() #  TODO: THIS SEEMS TO BE FALSE....
            # assert th.all(s != 0)
            # if self.args.sample_s_strategy in ['smallest', 'sigma_s_is_zero']:
                # assert new_indices.unique().item() == (num_scales - 1), new_indices.unique().item()
                # assert s.unique().item() == 0., s
            # print('s:', s)
            # exit()
        
        # 0908: made false
        use_bridge_sampling = True
        if use_bridge_sampling:
            # print('gonna call bridge sample in bridge ctm losses')
            x_t = self.bridge_sample(x0=x_start, x_T=model_kwargs['x_T'], sig_t=t, 
                                     gaussian_noise=noise, 
                                     sig_max=sigma_max,
                                     )
        else:
            x_t = x_start + (noise * append_dims(t, dims))
        # print("model_kwargs['x_T']:", model_kwargs['x_T'].max().item(), model_kwargs['x_T'].min().item())
        # exit()
        
        # assert not x_t.isnan().any(), f"\n-- x_t has nans: {x_t.isnan().any()}\n--x_0 has nans: {x_start.isnan().any()}\n-- x_T has nans: {model_kwargs['x_T'].isnan().any()}\n-- t has nans: {t.isnan().any()}\n-- noise has nans: {noise.isnan().any()}"
        
        dropout_state = th.get_rng_state()
        th.set_rng_state(dropout_state)
        # print('1', sigmas[num_heun_step:-1])
        # print('2',sigmas[:-num_heun_step-1])
        # print(sigmas)
        # exit()
        ctm_estimate = self.get_ctm_estimate(x_t, t, t_dt, s, model, target_model, ctm=ctm,
                                             outer_type=self.args.ctm_estimate_outer_type,
                                             inner_type=self.args.ctm_estimate_inner_type,
                                             target_matching=self.args.ctm_target_matching,
                                             **model_kwargs)
        
        assert not th.stack([th.isnan(p).any() for p in model.parameters()]).any()
        assert not th.stack([th.isnan(p).any() for p in target_model.parameters()]).any()
        assert not ctm_estimate.isnan().any(), current_timestep
        
        if step % self.args.g_learning_period == 0 or not self.args.gan_training:
            assert (discriminator == None) == (self.args.g_learning_period == 1)
            if use_bridge_sampling:
                # print('gonna call bridge sample in bridge ctm losses')
                x_t_dt = self.bridge_sample(x0=x_start, x_T=model_kwargs['x_T'], 
                                            sig_t=t_dt, gaussian_noise=noise, sig_max=sigma_max, 
                                            )
            else:
                x_t_dt = x_start + (noise * append_dims(t_dt, dims))
                
            # # NOTE: If Heun Solver, NFE = num_heun_steps x 2.
            # x_t_dt = self.heun_solver(target_model, x_t, indices, dims, t, t_dt, ctm=ctm, num_step=num_heun_step, num_scales=num_scales,
            #                           use_x0_as_denoised_in_solver=self.args.use_x0_as_denoised_in_solver, x_0=x_start,
            #                           sigmas=sigmas, **model_kwargs).detach()
            # x_t_dt = self.euler_solver(target_model, x_t, indices, dims, t, t_dt, ctm=ctm, num_step=num_heun_step, num_scales=num_scales,
            #                           use_x0_as_denoised_in_solver=self.args.use_x0_as_denoised_in_solver, x_0=x_start,
            #                           sigmas=sigmas, **model_kwargs).detach()
            # NOTE: If Contri Solver, NFE = num_heun_steps.
            # x_t_dt = self.contri_solver(target_model, x_t, indices, dims, t, t_dt, ctm=ctm, num_step=num_heun_step,
            #                           use_x0_as_denoised_in_solver=self.args.use_x0_as_denoised_in_solver, x_0=x_start,
            #                           churn_step_ratio=self.args.churn_step_ratio, num_scales=num_scales,
            #                           sigmas=sigmas, **model_kwargs).detach()
            # x_t_dt = self.hybrid_solver(target_model, x_t, indices, dims, t, t_dt, x_0=x_start, ctm=ctm, num_step=num_heun_step,
                                # use_x0_as_denoised_in_solver=self.args.use_x0_as_denoised_in_solver, 
            #                     **model_kwargs).detach()
            assert not x_t_dt.isnan().any()
            ctm_target = self.get_ctm_target(x_t_dt, t_dt, s, model, target_model, ctm=ctm,
                                             inner_type=self.args.ctm_target_inner_type, **model_kwargs)
            assert not ctm_target.isnan().any()
            if self.args.save_png and step % self.args.save_period == 0:
                img_save_dir = logger.get_dir() + '/images/'
                script_util.save(x_start, img_save_dir, f'x0_GroundTruth_{step}')  # _{r}')
                script_util.save(ctm_estimate, img_save_dir, f'ctm_estimate_{step}')  # _{r}')
                script_util.save(ctm_target, img_save_dir, f'ctm_target_{step}')  # _{r}')
                script_util.save(x_t, img_save_dir, f'xt_NonDenoised_{step}')  # _{r}')
                script_util.save(x_t_dt, img_save_dir, f'xu_{num_heun_step}Steps_Solved_{step}')  # _{r}')
                script_util.save(model_kwargs['x_T'], img_save_dir, f'xT_{step}')  # _{r}')
                
                # script_util.save(ctm_estimate, logger.get_dir(), f'ctm_estimate_{step}')  # _{r}')
                # script_util.save(ctm_target, logger.get_dir(), f'ctm_target_{step}')  # _{r}')
                # script_util.save(x_t, logger.get_dir(), f'xt_NonDenoised_{step}')  # _{r}')
                # script_util.save(x_t_dt, logger.get_dir(), f'xu_Solved_{step}')  # _{r}')

            snrs = self.get_snr(append_dims(t, x_t.ndim))
            # org_weights = self.get_weightings(self.args.weight_schedule, snrs, self.args.sigma_data, t, s, self.args.weight_schedule_multiplier)
            weights = self.get_weightings(self.args.weight_schedule, snrs, self.args.sigma_data, 
                                          append_dims(t, x_t.ndim), append_dims(t_dt, x_t.ndim),
                                          self.args.weight_schedule_multiplier,)
            
            # print('org weights:', (org_weights).mean(), 'should have been:', (weights).mean())
            # print('should have been:', (1/weights).mean())
            # print(weights.mean())
            # print((1/(t-t_dt)).mean())
            # exit()
            
            terms["consistency_loss"] = self.get_CTM_loss(ctm_estimate, ctm_target, weights, step - init_step,)
            # print('pred:', ctm_estimate.requires_grad, "| target:", ctm_target.requires_grad, "| loss:", ctm_loss.requires_grad)
             
            # ctm_loss.register_hook(lambda g: print(g))
            # terms["consistency_loss"] = ctm_loss
            
            # print(ctm_loss.mean(), (ctm_loss/weights).mean())
            # exit()
            
            if self.args.large_log:
                if s != None:
                    print(f"{step}-th step, t, t-dt, s, weight, loss: ", t[0].item(), t_dt[0].item(), s[0].item(), weights[0].item(), terms["consistency_loss"][0].item())
                else:
                    print(f"{step}-th step, t, t-dt, weight, loss: ", t[0].item(), t_dt[0].item(), terms["consistency_loss"][0].item(), weights[0].item())
            # if self.args.diffusion_training:
            #     if diffusion_training_:
            #         terms['denoising_loss'] = self.get_DSM_loss(model, x_start, model_kwargs,
            #                                                           terms["consistency_loss"],
            #                                                           step, init_step)

            if self.args.diffusion_training:
                if diffusion_training_:
                    # terms['bridge_denoising_loss'] = self.get_DBSM_loss(model, x_start,
                        # model_kwargs=model_kwargs,
                        # consistency_loss=terms["consistency_loss"],
                        # step=step, init_step=init_step, 
                        # bridge_sample=bridge_sample, 
                        # use_bridge_sample=True,
                        # sigma_T=sigma_max,)
                    (terms['bridge_denoising_loss'], dbsm_x_t, dbsm_denoised, dbsm_sigma_t, dbsm_weights) = self.get_DBSM_loss(
                        model, x_start, 
                        noise=noise, model_kwargs=model_kwargs,
                        consistency_loss=terms["consistency_loss"], step=step, init_step=init_step,
                        ctm_sigma_t=t, ctm_xt=x_t,
                        use_bridge_sample=use_bridge_sampling,
                        sigma_T=sigma_max,
                    )
                    
                    if self.args.save_png and step % self.args.save_period == 0:
                        script_util.save(dbsm_denoised, img_save_dir, f'x0_Pred_DM{step}')  # _{r}')
                    # # !!!! (06/06) Apparently not a contribution.
                    # #              Cz DBSM/DSM loss literally does the same thing as this loss. 
                    # #              So, can't really say it's a contribution... :(
                    # if self.args.qpart_loss:
                    #     # One of the main contributions. 
                    #     terms['q_part_loss'] = self.get_q_part_loss(model=model,
                    #         x_start=x_start, 
                    #         x_t=dbsm_x_t,
                    #         denoised=dbsm_denoised,
                    #         denoising_weights=dbsm_weights,
                    #         sigma_t=dbsm_sigma_t,
                    #         model_kwargs=model_kwargs,
                    #         consistency_loss=terms["consistency_loss"], step=step, init_step=init_step,
                    #         sigma_T=sigma_max,
                    #     )
                    
            # if self.args.gan_training and step - init_step >= self.args.discriminator_start_itr:
            #     if gan_training_:
            #         gan_x_t, gan_t, gan_t_dt, gan_s, _, _ = self.get_gan_time(x_start, noise, x_t, t, t_dt, s, indices,
            #                                                                   num_heun_step, gan_num_heun_step)
            #         gan_fake = self.get_gan_fake(ctm_estimate, gan_x_t, gan_t, gan_t_dt, gan_s, model, target_model, ctm,
            #                                      step - init_step, **model_kwargs)
            #         terms['d_loss'] = self.get_GAN_loss(model, fake=gan_fake,
            #                                                       consistency_loss=terms["consistency_loss"],
            #                                                       discriminator=discriminator,
            #                                                       step=step, init_step=init_step)
        else:
            gan_x_t, gan_t, gan_t_dt, gan_s, gan_indices, gan_num_heun_step = \
                self.get_gan_time(x_start, noise, x_t, t, t_dt, s, indices, num_heun_step, gan_num_heun_step)
            gan_real = self.get_gan_real(x_start, gan_x_t, gan_t, gan_t_dt, gan_s, gan_indices, dims, gan_num_heun_step,
                                         model, target_model, ctm, step - init_step, **model_kwargs)
            gan_fake = self.get_gan_fake(ctm_estimate, gan_x_t, gan_t, gan_t_dt, gan_s, model, target_model, ctm,
                                         step - init_step, **model_kwargs)
            terms['d_loss'] = self.get_GAN_loss(model, fake=gan_fake, real=gan_real,
                                                learn_generator=False, discriminator=discriminator,
                                                step=step, init_step=init_step, **model_kwargs)
        return terms
    
    def bridge_sample(self, x0, x_T, sig_t, gaussian_noise, sig_max=None):
        # print('bridge sampling!')
        dims = x0.ndim
        sig_t = append_dims(sig_t, dims)
        # std_t = th.sqrt(t)* th.sqrt(1 - t / self.sigma_max)
        sig_max = self.args.sigma_max if sig_max is None else sig_max
        sig_max = append_dims(th.ones_like(sig_t) * sig_max, dims)
        
        if self.pred_mode.startswith('ve'):
            # print("sigma at bridge_sample", sig_t.max().item(), sig_t.min().item())
            # print("sigma max at bridge_sample", self.args.sigma_max)
            # exit()
            snrT_div_snrt = (sig_t**2) / (sig_max**2)
            assert not snrT_div_snrt.isnan().any(), f"{'-'*15}\nbridge_sample:--SNRT/t has nans: {snrT_div_snrt.isnan().any()}\n"
            
            std_t = sig_t * th.sqrt(1. - snrT_div_snrt)
            mu_t = (snrT_div_snrt * x_T) + ((1. - snrT_div_snrt) * x0)
            
            samples = mu_t + (std_t * gaussian_noise)
            
            if samples.isnan().any():
                msg = f"{'-'*15}\nbridge_sample:\n-- samples has nans: {samples.isnan().any()}\n--x0 has nans: {x0.isnan().any()}\n-- x_T has nans: {x_T.isnan().any()}\n-- sig_t has nans: {sig_t.isnan().any()}\n-- noise has nans: {gaussian_noise.isnan().any()}"
                msg += f"\n-- mu_t has nans: {mu_t.isnan().any()}"
                msg += f"\n-- std_t has nans: {std_t.isnan().any()}"
                msg += f"\n-- th.sqrt(1 - snrT_div_snrt) has nans: {th.sqrt(1 - snrT_div_snrt).isnan().any()}"
                msg += f"\n-- (1 - snrT_div_snrt) values that cause nan when sqrt is taken:\n{(1 - snrT_div_snrt)[th.sqrt(1 - snrT_div_snrt).isnan()]}"
                msg += f"\n-- sig_t values that caused the nan:\n{sig_t[th.sqrt(1 - snrT_div_snrt).isnan()]}"
                msg += f"\n-- snrT_div_snrt * x_T has nans: {(snrT_div_snrt * x_T).isnan().any()}"
                msg += f"\n-- (1 - snrT_div_snrt) * x0 has nans: {((1 - snrT_div_snrt) * x0).isnan().any()}"
                raise ValueError(msg)
            # assert not samples.isnan().any()
            
        elif self.pred_mode.startswith('vp'):
            logsnr_t = vp_logsnr(sig_t, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(sig_max, self.beta_d, self.beta_min)
            logs_t = vp_logs(sig_t, self.beta_d, self.beta_min)
            logs_T = vp_logs(sig_max, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
            b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            std_t = (-th.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
            
            samples = (a_t * x_T) + (b_t * x0) + (std_t * gaussian_noise)

        return samples
