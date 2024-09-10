from abc import ABC, abstractmethod

import numpy as np
import torch as th
from scipy.stats import norm
import torch.distributed as dist


def create_named_schedule_sampler(args, name: str, num_timesteps: int):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(num_timesteps)
    # elif name == "loss-second-moment":
    #     return LossSecondMomentResampler(num_timesteps)
    elif name == "lognormal":
        return LogNormalSampler()
    elif name == 'halflognormal':
        return HalfLogNormalHalfUniformSampler(args)
    # elif name == "real-uniform":
    #     return RealUniformSampler(args)
    elif name.lower() == 'ict':
        return iCTSampler(num_timesteps)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self, num_heun_step):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample_t(self, args, batch_size, device, num_heun_step=1, sigmas=None, num_scales=None, time_continuous=False):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        if time_continuous:
            indices_np = np.random.rand(batch_size)
            indices = th.from_numpy(indices_np).to(device) * (1. - num_heun_step)
            weights = th.ones_like(indices).float().to(device)
        else:
            w = self.weights(num_heun_step)
            p = w / np.sum(w)
            indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
            indices = th.from_numpy(indices_np).long().to(device)
            weights_np = 1 / (len(p) * p[indices_np])
            weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights

    def sample_s(self, args, batch_size, device, indices, num_heun_step=1, time_continuous=False, N=40):
        if time_continuous:
            new_indices = th.from_numpy(np.random.rand(indices.shape[0])) * \
                            (1. - indices - num_heun_step) + indices + num_heun_step
        else:
            if args.sample_s_strategy == 'smallest':
                # new_indices = th.ones(indices.shape[0]) * (N - 1)
                new_indices = th.ones(indices.shape[0], dtype=th.int32) * N
            elif args.sample_s_strategy == 'uniform':
                new_indices = th.from_numpy(np.random.randint(
                    low=(indices + num_heun_step).detach().cpu().numpy(), 
                    high=N,
                    size=(indices.shape[0],),
                    dtype=int))
            elif args.sample_s_strategy == 'sigma_s_is_zero':
                # new_indices = th.ones(indices.shape[0]) * (N - 1)
                new_indices = th.ones(indices.shape[0], dtype=th.int32) * N
        return new_indices.to(indices.device)

# class RealUniformSampler:
#     def __init__(self, sigma_max=80., sigma_min=0.002):
#         self.sigma_max = sigma_max
#         self.sigma_min = sigma_min

#     def sample(self, batch_size, device):
#         ts = th.rand(batch_size).to(device) * (self.sigma_max - self.sigma_min) + self.sigma_min
#         return ts, th.ones_like(ts)

class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self._weights = np.ones([num_timesteps])
        self.num_timesteps = num_timesteps

    def weights(self, num_heun_step=1):
        #if num_heun_step == 1:
        #    return self._weights
        #else:
        return np.ones([self.num_timesteps - num_heun_step])

class iCTSampler:
    def __init__(self, num_timesteps, p_mean=-1.1, p_std=2.0):
        self._weights = np.ones([num_timesteps])
        self.num_timesteps = num_timesteps
        self.p_mean = p_mean
        self.p_std = p_std

    def weights(self, num_heun_step=1):
        #if num_heun_step == 1:
        #    return self._weights
        #else:
        return np.ones([self.num_timesteps - num_heun_step])
    
    # def sample_t(self, args, batch_size, device, indices, num_heun_step=1, time_continuous=False, N=40):
    #     'discretized_sigmas' must be received as one of the tokens.
    #     if time_continuous:
    #         new_indices = th.from_numpy(np.random.rand(indices.shape[0])) * \
    #                         (1. - indices - num_heun_step) + indices + num_heun_step
    #     else:
    #         if args.sample_s_strategy == 'smallest':
    #             new_indices = th.ones(indices.shape[0]) * (N - 1)
    #         elif args.sample_s_strategy == 'uniform':
    #             new_indices = th.from_numpy(np.random.randint(
    #                 low=(indices + num_heun_step).detach().cpu().numpy(), 
    #                 high=N,
    #                 size=(indices.shape[0],),
    #                 dtype=int))
    #         elif args.sample_s_strategy == 'sigma_s_is_zero':
    #             new_indices = th.ones(indices.shape[0]) * (N - 1) 
    #     return new_indices.to(indices.device)
    
    def lognormal_timestep_distribution(self, batch_size: int, sigmas: th.Tensor, num_scales: int, num_heun_step: int):
        # pdf = th.erf((th.log(sigmas[1:]) - self.p_mean) / (self.p_std * np.sqrt(2))) - th.erf(
        #     (th.log(sigmas[:-1]) - self.p_mean) / (self.p_std * np.sqrt(2))
        # )
        # pdf = pdf / pdf.sum()
        # timesteps = th.multinomial(pdf, batch_size, replacement=True)
        # return timesteps
        
        # if num_heun_step >= num_scales:
        #     num_heun_step = num_scales - 1
        #     assert num_heun_step >= 1
        
        # print('num_scales', num_scales)
        # print('num_heun_step', num_heun_step)
        # exit()
        # print('sigmas:', sigmas)
        # print('sigmas[:-num_heun_step-1]', sigmas[:-num_heun_step-1])
        # exit()
        
        # pdf_new = th.erf(
        #     (th.log(sigmas[:-num_heun_step-1]) - self.p_mean) / (self.p_std * np.sqrt(2))
        # )
        pdf_new = th.erf(
            (th.log(sigmas[:-num_heun_step-1]) - self.p_mean) / (self.p_std * np.sqrt(2))
        )

        # pdf_new = pdf_new - th.erf(
        #                 (th.log(sigmas[num_heun_step:-1]) - self.p_mean) / (self.p_std * np.sqrt(2))
        #             )
        pdf_new = pdf_new - th.erf(
                        (th.log(sigmas[num_heun_step:-1]) - self.p_mean) / (self.p_std * np.sqrt(2))
                    )
        
        pdf_new = pdf_new / pdf_new.sum()
        assert not pdf_new.isnan().any()

        timesteps_t = th.multinomial(pdf_new, batch_size, replacement=True)

        assert (timesteps_t < num_scales - num_heun_step).all()

        # u = timesteps_t.detach().clone() + num_heun_step
        
        # assert (u < num_scales).all() 
        
        return timesteps_t
        
    
    def sample_t(self, args, batch_size, device, sigmas, num_scales, num_heun_step):
        timesteps = self.lognormal_timestep_distribution(batch_size, sigmas, num_scales, num_heun_step)
        return timesteps.to(device)
    
    def sample_s(self, args, batch_size, device, indices, num_heun_step=1, time_continuous=False, N=40):
        if time_continuous:
            new_indices = th.from_numpy(np.random.rand(indices.shape[0])) * \
                            (1. - indices - num_heun_step) + indices + num_heun_step
        else:
            if args.sample_s_strategy == 'smallest':
                # new_indices = th.ones(indices.shape[0], dtype=th.int32) * (N - 1)
                new_indices = th.ones(indices.shape[0], dtype=th.int32) * N
            elif args.sample_s_strategy == 'uniform':
                new_indices = th.from_numpy(np.random.randint(
                    low=(indices + num_heun_step).detach().cpu().numpy(), 
                    high=N,
                    size=(indices.shape[0],),
                    dtype=int))
            elif args.sample_s_strategy == 'sigma_s_is_zero':
                # new_indices = th.ones(indices.shape[0], dtype=th.int32) * (N - 1)
                new_indices = th.ones(indices.shape[0], dtype=th.int32) * N
        
        return new_indices.to(indices.device)


# class LossAwareSampler(ScheduleSampler):
#     def update_with_local_losses(self, local_ts, local_losses):
#         """
#         Update the reweighting using losses from a model.

#         Call this method from each rank with a batch of timesteps and the
#         corresponding losses for each of those timesteps.
#         This method will perform synchronization to make sure all of the ranks
#         maintain the exact same reweighting.

#         :param local_ts: an integer Tensor of timesteps.
#         :param local_losses: a 1D Tensor of losses.
#         """
#         batch_sizes = [
#             th.tensor([0], dtype=th.int32, device=local_ts.device)
#             for _ in range(dist.get_world_size())
#         ]
#         dist.all_gather(
#             batch_sizes,
#             th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
#         )

#         # Pad all_gather batches to be the maximum batch size.
#         batch_sizes = [x.item() for x in batch_sizes]
#         max_bs = max(batch_sizes)

#         timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
#         loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
#         dist.all_gather(timestep_batches, local_ts)
#         dist.all_gather(loss_batches, local_losses)
#         timesteps = [
#             x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
#         ]
#         losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
#         self.update_with_all_losses(timesteps, losses)

#     @abstractmethod
#     def update_with_all_losses(self, ts, losses):
#         """
#         Update the reweighting using losses from a model.

#         Sub-classes should override this method to update the reweighting
#         using losses from the model.

#         This method directly updates the reweighting without synchronizing
#         between workers. It is called by update_with_local_losses from all
#         ranks with identical arguments. Thus, it should have deterministic
#         behavior to maintain state across workers.

#         :param ts: a list of int timesteps.
#         :param losses: a list of float losses, one per timestep.
#         """


# class LossSecondMomentResampler(LossAwareSampler):
#     def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
#         self.history_per_term = history_per_term
#         self.uniform_prob = uniform_prob
#         self.num_timesteps = num_timesteps
#         self._loss_history = np.zeros(
#             [num_timesteps, history_per_term], dtype=np.float64
#         )
#         self._loss_counts = np.zeros([num_timesteps], dtype=np.int)

#     def weights(self):
#         if not self._warmed_up():
#             return np.ones([self.num_timesteps], dtype=np.float64)
#         weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
#         weights /= np.sum(weights)
#         weights *= 1 - self.uniform_prob
#         weights += self.uniform_prob / len(weights)
#         return weights

#     def update_with_all_losses(self, ts, losses):
#         for t, loss in zip(ts, losses):
#             if self._loss_counts[t] == self.history_per_term:
#                 # Shift out the oldest loss term.
#                 self._loss_history[t, :-1] = self._loss_history[t, 1:]
#                 self._loss_history[t, -1] = loss
#             else:
#                 self._loss_history[t, self._loss_counts[t]] = loss
#                 self._loss_counts[t] += 1

#     def _warmed_up(self):
#         return (self._loss_counts == self.history_per_term).all()


class LogNormalSampler:
    def __init__(self, p_mean=-1.2, p_std=1.2, even=False):
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        if self.even:
            self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
            self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def sample(self, bs, device, num_heun_step=1):
        if self.even:
            # buckets = [1/G]
            start_i, end_i = self.rank * bs, (self.rank + 1) * bs
            global_batch_size = self.size * bs
            locs = (th.arange(start_i, end_i) + th.rand(bs)) / global_batch_size
            log_sigmas = th.tensor(self.inv_cdf(locs), dtype=th.float32, device=device)
        else:
            log_sigmas = self.p_mean + self.p_std * th.randn(bs, device=device)
        sigmas = th.exp(log_sigmas)
        weights = th.ones_like(sigmas)
        return sigmas, weights

class HalfLogNormalHalfUniformSampler:
    def __init__(self, args, p_mean=-1.2, p_std=1.2, even=False):
        self.args = args
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        if self.even:
            self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
            self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def get_t(self, t, sig_max=None):
        sig_max = (self.args.sigma_max-1e-4) if sig_max is None else sig_max
        
        t = sig_max ** (1 / self.args.rho) + t * (
                self.args.sigma_min ** (1 / self.args.rho) - sig_max ** (1 / self.args.rho)
        )
        t = t ** self.args.rho
        return t

    def sample(self, bs, device, num_heun_step=1):
        log_sigmas = self.p_mean + self.p_std * th.randn(bs//2, device=device)
        sigmas = th.exp(log_sigmas) 
        t = th.rand(bs - bs//2, device=device) * self.args.diffusion_mult # it determines how large sigma to sample from
        sigmas = th.cat((sigmas, self.get_t(t))).view(-1)
        weights = th.ones_like(sigmas)
        return sigmas, weights
