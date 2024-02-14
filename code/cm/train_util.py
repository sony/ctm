import copy
import functools
import os
import cv2
from skimage.metrics import structural_similarity as SSIM_
import shutil
import gc

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
import torch.nn.functional as F

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import nvidia_smi

from .fp16_util import (
    get_param_groups_and_shapes,
    get_target_param_groups_and_shapes,
    make_master_params,
    state_dict_to_master_params,
    master_params_to_model_params,
)
import numpy as np
from cm.sample_util import karras_sample
from cm.random_util import get_generator
from torchvision.utils import make_grid, save_image
import datetime
import dnnlib
import pickle
import glob
import scipy

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        #*,
        model,
        discriminator,
        diffusion,
        data,
        batch_size,
        args=None,
    ):
        self.args = args
        self.model = model
        if self.args.sanity_check:
            for name, param in self.model.named_parameters():
                logger.log("check and understand how consistency-type models override model parameters")
                logger.log("model parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        self.discriminator = discriminator
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = args.microbatch if args.microbatch > 0 else batch_size
        self.lr = args.lr
        self.ema_rate = (
            [args.ema_rate]
            if isinstance(args.ema_rate, float)
            else [float(x) for x in args.ema_rate.split(",")]
        )
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.fids = []
        self.generator = get_generator('determ', self.args.eval_num_samples, self.args.eval_seed)
        self.x_T = self.generator.randn(*(self.args.sampling_batch, self.args.in_channels, self.args.image_size, self.args.image_size),
                                        device='cpu') * self.args.sigma_max #.to(dist_util.dev())
        if self.args.class_cond:
            self.classes = self.generator.randint(0, self.args.num_classes, (self.args.sampling_batch,), device='cpu')
            if self.args.data_name.lower() == 'cifar10':
                self.classes.sort()

        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        if self.args.sanity_check:
            for name, param in self.model.named_parameters():
                logger.log("model parameter after overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        if self.discriminator != None:
            if self.args.sanity_check:
                for name, param in self.discriminator.named_parameters():
                    logger.log("discriminator parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            self._load_and_sync_discriminator_parameters()
            if self.args.sanity_check:
                for name, param in self.discriminator.named_parameters():
                    logger.log("discriminator parameter after overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                    break

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
        )
        if self.args.sanity_check:
            logger.log("mp trainer master parameter (should same to the model parameter if no linear_probing): ", self.mp_trainer.master_params[1].reshape(-1)[:3])

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.args.weight_decay
        )
        if self.args.sanity_check:
            print("opt state dict before overriding: ", self.opt.state_dict())

        if self.discriminator != None:
            self.d_mp_trainer = MixedPrecisionTrainer(
                model=self.discriminator,
                use_fp16=args.use_d_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
            )
            self.d_opt = RAdam(
                self.d_mp_trainer.master_params, lr=args.d_lr, weight_decay=self.args.weight_decay, betas=(0.5, 0.9)
            )
        if self.resume_step:
            self._load_optimizer_state()
            if self.discriminator != None:
                try:
                    if self.args.sanity_check:
                        print("discriminator opt state dict before overriding: ", self.d_opt.state_dict())
                    self._load_d_optimizer_state()
                    if self.args.sanity_check:
                        print("discriminator opt state dict after overriding: ", self.d_opt.state_dict())
                except:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! warning !!!!!!!!!!!!!!!!!!!!!!!!!!!! discriminator optimizer not loaded successfully")
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            self.ddp_discriminator = None
            if self.args.gan_training:
                self.ddp_discriminator = DDP(
                    self.discriminator,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading pretrained model from checkpoint: {resume_checkpoint}...")
                if dist.get_world_size() > 1:
                    state_dict = th.load(resume_checkpoint, map_location=dist_util.dev())#"cpu")
                else:
                    state_dict = dist_util.load_state_dict(
                        resume_checkpoint, map_location='cpu',  # dist_util.dev()
                    )
                self.model.load_state_dict(state_dict, strict=False)
                logger.log(f"end loading pretrained model from checkpoint: {resume_checkpoint}...")

        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.model.buffers())
        logger.log(f"end synchronizing pretrained model from GPU0 to all GPUs")

    def _load_and_sync_discriminator_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            resume_checkpoint = bf.join(bf.dirname(resume_checkpoint), f"d_model{self.resume_step:06}.pt")
            if dist.get_rank() == 0:
                if os.path.exists(resume_checkpoint):
                    logger.log(f"loading discriminator model from checkpoint: {resume_checkpoint}...")
                    #try:
                    if dist.get_world_size() > 1:
                        state_dict = th.load(resume_checkpoint, map_location="cpu")
                    else:
                        state_dict = dist_util.load_state_dict(
                            resume_checkpoint, map_location=dist_util.dev()
                        )
                    self.discriminator.load_state_dict(state_dict)
                    logger.log(f"end loading discriminator model from checkpoint: {resume_checkpoint}...")

        dist_util.sync_params(self.discriminator.parameters())
        dist_util.sync_params(self.discriminator.buffers())
        logger.log(f"end synchronizing discriminator from GPU0 to all GPUs")

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        if self.args.sanity_check:
            logger.log(f"{rate} ema param before overriding: ", ema_params[1].reshape(-1)[:3])
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                if dist.get_world_size() > 1:
                    state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())#"cpu")
                else:
                    state_dict = dist_util.load_state_dict(
                        ema_checkpoint, map_location=dist_util.dev()
                    )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
                logger.log(f"end loading EMA from checkpoint: {ema_checkpoint}...")

        dist_util.sync_params(ema_params)
        logger.log(f"end synchronizing EMA from GPU0 to all GPUs")
        if self.args.sanity_check:
            logger.log(f"{rate} ema param after overriding: ", ema_params[1].reshape(-1)[:3])

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            if dist.get_world_size() > 1:
                state_dict = th.load(opt_checkpoint, map_location="cpu")
            else:
                state_dict = dist_util.load_state_dict(
                    opt_checkpoint, map_location=dist_util.dev()
                )
            self.opt.load_state_dict(state_dict)
            logger.log(f"end loading optimizer state from checkpoint: {opt_checkpoint}")

        if self.args.sanity_check:
            print("opt state dict after overriding: ", self.opt.state_dict()['state'])

    def _load_d_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"d_opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading d_optimizer state from checkpoint: {opt_checkpoint}")
            if os.path.exists(opt_checkpoint):
                if dist.get_world_size() > 1:
                    state_dict = th.load(opt_checkpoint, map_location="cpu")
                else:
                    state_dict = dist_util.load_state_dict(
                        opt_checkpoint, map_location=dist_util.dev()
                    )
                self.d_opt.load_state_dict(state_dict)
            logger.log(f"end loading d_optimizer state from checkpoint: {opt_checkpoint}")

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.args.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.args.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def sampling(self, model, sampler, ctm=None, teacher=False, step=-1, num_samples=-1, batch_size=-1, rate=0.999,
                 png=False, resize=True, generator=None, class_generator=None, sample_dir=''):
        if not teacher:
            model.eval()
        if step == -1:
            step = self.args.sampling_steps
        if batch_size == -1:
            batch_size = self.args.sampling_batch

        number = 0
        while num_samples > number:
            print(f"{number} number samples complete")
            with th.no_grad():
                model_kwargs = {}
                if self.args.class_cond:
                    if self.args.train_classes >= 0:
                        classes = th.ones(size=(batch_size,), device=dist_util.dev(), dtype=int) * self.args.train_classes
                        model_kwargs["y"] = classes
                    elif self.args.train_classes == -2:
                        classes = [0, 1, 9, 11, 29, 31, 33, 55, 76, 89, 90, 130, 207, 250, 279, 281, 291, 323, 386, 387,
                                   388, 417, 562, 614, 759, 789, 800, 812, 848, 933, 973, 980]
                        assert batch_size % len(classes) == 0
                        model_kwargs["y"] = th.tensor([x for x in classes for _ in range(batch_size // len(classes))], device=dist_util.dev())
                    else:
                        if class_generator != None:
                            model_kwargs["y"] = class_generator.randint(0, self.args.num_classes, (batch_size,), device=dist_util.dev())
                        else:
                            if num_samples == -1:
                                model_kwargs["y"] = self.classes.to(dist_util.dev())
                            else:
                                model_kwargs["y"] = th.randint(0, self.args.num_classes, size=(batch_size, ), device=dist_util.dev())
                if generator != None:
                    x_T = generator.randn(*(batch_size, self.args.in_channels, self.args.image_size, self.args.image_size),
                                device=dist_util.dev()) * self.args.sigma_max
                    if self.args.large_log:
                        print("x_T: ", x_T[0][0][0][:3])
                else:
                    x_T = None

                sample = karras_sample(
                    diffusion=self.diffusion,
                    model=model,
                    shape=(batch_size, self.args.in_channels, self.args.image_size, self.args.image_size),
                    steps=step,
                    model_kwargs=model_kwargs,
                    device=dist_util.dev(),
                    clip_denoised=True if teacher else self.args.clip_denoised,
                    sampler=sampler,
                    generator=None,
                    teacher=teacher,
                    ctm=ctm if ctm != None else True if self.args.training_mode.lower() == 'ctm' else False,
                    x_T=x_T if generator != None else self.x_T.to(dist_util.dev()) if num_samples == -1 else None,
                    clip_output=self.args.clip_output,
                    sigma_min=self.args.sigma_min,
                    sigma_max=self.args.sigma_max,
                    train=False,
                )
                if resize:
                    sample = F.interpolate(sample, size=224, mode="bilinear")

                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)
                all_images = [sample.cpu().numpy() for sample in gathered_samples]
                arr = np.concatenate(all_images, axis=0)
                if dist.get_rank() == 0:
                    os.makedirs(get_blob_logdir(), exist_ok=True)
                    if self.args.large_log:
                        print(f"saving to {bf.join(get_blob_logdir(), sample_dir)}")
                    nrow = int(np.sqrt(arr.shape[0]))
                    image_grid = make_grid(th.tensor(arr).permute(0, 3, 1, 2) / 255., nrow, padding=2)
                    if num_samples == -1:
                        with bf.BlobFile(bf.join(get_blob_logdir(), f"{'teacher_' if teacher else ''}sample_{sampler}_sampling_step_{step}_step_{self.step}.png"), "wb") as fout:
                            save_image(image_grid, fout)
                    else:
                        if generator != None:
                            os.makedirs(bf.join(get_blob_logdir(), sample_dir),
                                        exist_ok=True)
                            np.savez(bf.join(get_blob_logdir(), f"{sample_dir}/sample_{number // arr.shape[0]}.npz"),
                                     arr)
                            if png and number <= 3000:
                                with bf.BlobFile(bf.join(get_blob_logdir(),
                                                         f"{sample_dir}/sample_{number // arr.shape[0]}.png"), "wb") as fout:
                                    save_image(image_grid, fout)
                        else:
                            r = np.random.randint(100000000)
                            if self.args.large_log:
                                logger.log(f'{dist.get_rank()} number {number}')
                            os.makedirs(bf.join(get_blob_logdir(), f"{sample_dir}"),
                                        exist_ok=True)
                            np.savez(bf.join(get_blob_logdir(), f"{sample_dir}/sample_{r}.npz"),
                                     arr)
                            if png and number <= 1000:
                                with bf.BlobFile(bf.join(get_blob_logdir(),
                                                         f"{sample_dir}/sample_{r}.png"), "wb") as fout:
                                    save_image(image_grid, fout)
                number += arr.shape[0]
        if not teacher:
            model.train()

    def calculate_similarity_metrics(self, image_path, num_samples=50000, step=1, batch_size=100, rate=0.999, sampler='exact', log=True):
        files = glob.glob(os.path.join(image_path, 'sample*.npz'))
        files.sort()
        count = 0
        psnr = 0
        ssim = 0
        for i, file in enumerate(files):
            images = np.load(file)['arr_0']
            for k in range((images.shape[0] - 1) // batch_size + 1):
                #ref_img = self.ref_images[count + k * batch_size: count + (k + 1) * batch_size]
                if count + batch_size > num_samples:
                    remaining_num_samples = num_samples - count
                else:
                    remaining_num_samples = batch_size
                img = images[k * batch_size: k * batch_size + remaining_num_samples]
                ref_img = self.ref_images[count: count + remaining_num_samples]
                psnr += cv2.PSNR(img, ref_img) * remaining_num_samples
                ssim += SSIM_(img,ref_img,multichannel=True,channel_axis=3,data_range=255) * remaining_num_samples
                count = count + remaining_num_samples
                print(count)
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
        assert count == num_samples
        print(count)
        psnr /= num_samples
        ssim /= num_samples
        assert num_samples % 1000 == 0
        if log:
            logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate} PSNR-{num_samples // 1000}k: {psnr}, SSIM-{num_samples // 1000}k: {ssim}")
        else:
            return psnr, ssim

    def calculate_inception_stats(self, data_name, image_path, num_samples=50000, batch_size=100, device=th.device('cuda')):
        if data_name.lower() == 'cifar10':
            print(f'Loading images from "{image_path}"...')
            mu = th.zeros([self.feature_dim], dtype=th.float64, device=device)
            sigma = th.zeros([self.feature_dim, self.feature_dim], dtype=th.float64, device=device)
            files = glob.glob(os.path.join(image_path, 'sample*.npz'))
            count = 0
            for file in files:
                images = np.load(file)['arr_0']  # [0]#["samples"]
                for k in range((images.shape[0] - 1) // batch_size + 1):
                    mic_img = images[k * batch_size: (k + 1) * batch_size]
                    mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                    features = self.detector_net(mic_img, **self.detector_kwargs).to(th.float64)
                    if count + mic_img.shape[0] > num_samples:
                        remaining_num_samples = num_samples - count
                    else:
                        remaining_num_samples = mic_img.shape[0]
                    mu += features[:remaining_num_samples].sum(0)
                    sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                    count = count + remaining_num_samples
                    print(count)
                    if count >= num_samples:
                        break
                if count >= num_samples:
                    break
            assert count == num_samples
            print(count)
            mu /= num_samples
            sigma -= mu.ger(mu) * num_samples
            sigma /= num_samples - 1
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()
            return mu, sigma
        else:
            filenames = glob.glob(os.path.join(image_path, '*.npz'))
            imgs = []
            for file in filenames:
                try:
                    img = np.load(file)  # ['arr_0']
                    try:
                        img = img['data']
                    except:
                        img = img['arr_0']
                    imgs.append(img)
                except:
                    pass
            imgs = np.concatenate(imgs, axis=0)
            os.makedirs(os.path.join(image_path, 'single_npz'), exist_ok=True)
            np.savez(os.path.join(os.path.join(image_path, 'single_npz'), f'data'),
                     imgs)  # , labels)
            logger.log("computing sample batch activations...")
            sample_acts = self.evaluator.read_activations(
                os.path.join(os.path.join(image_path, 'single_npz'), f'data.npz'))
            logger.log("computing/reading sample batch statistics...")
            sample_stats, sample_stats_spatial = tuple(self.evaluator.compute_statistics(x) for x in sample_acts)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'stats'), 'wb') as f:
                pickle.dump({'stats': sample_stats, 'stats_spatial': sample_stats_spatial}, f)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'acts'), 'wb') as f:
                pickle.dump({'acts': sample_acts[0], 'acts_spatial': sample_acts[1]}, f)
            return sample_acts, sample_stats, sample_stats_spatial

    def compute_fid(self, mu, sigma, ref_mu=None, ref_sigma=None):
        if np.array(ref_mu == None).sum():
            ref_mu = self.mu_ref
            assert ref_sigma == None
            ref_sigma = self.sigma_ref
        m = np.square(mu - ref_mu).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, ref_sigma), disp=False)
        fid = m + np.trace(sigma + ref_sigma - s * 2)
        fid = float(np.real(fid))
        return fid

    def calculate_inception_stats_npz(self, image_path, num_samples=50000, step=1, batch_size=100, device=th.device('cuda'),
                                      rate=0.999):
        print(f'Loading images from "{image_path}"...')
        mu = th.zeros([self.feature_dim], dtype=th.float64, device=device)
        sigma = th.zeros([self.feature_dim, self.feature_dim], dtype=th.float64, device=device)

        files = glob.glob(os.path.join(image_path, 'sample*.npz'))
        count = 0
        for file in files:
            images = np.load(file)['arr_0']  # [0]#["samples"]
            for k in range((images.shape[0] - 1) // batch_size + 1):
                mic_img = images[k * batch_size: (k + 1) * batch_size]
                mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                features = self.detector_net(mic_img, **self.detector_kwargs).to(th.float64)
                if count + mic_img.shape[0] > num_samples:
                    remaining_num_samples = num_samples - count
                else:
                    remaining_num_samples = mic_img.shape[0]
                mu += features[:remaining_num_samples].sum(0)
                sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                count = count + remaining_num_samples
                logger.log(count)
            if count >= num_samples:
                break
        assert count == num_samples
        print(count)
        mu /= num_samples
        sigma -= mu.ger(mu) * num_samples
        sigma /= num_samples - 1
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()

        m = np.square(mu - self.mu_ref).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, self.sigma_ref), disp=False)
        fid = m + np.trace(sigma + self.sigma_ref - s * 2)
        fid = float(np.real(fid))
        assert num_samples % 1000 == 0
        logger.log(f"{self.step}-th step exact sampler (NFE {step}) EMA {rate} FID-{num_samples // 1000}k: {fid}")

class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        teacher_model,
        ema_scale_fn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = self.args.training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.total_training_steps = self.args.total_training_steps

        if target_model:
            if self.args.sanity_check:
                for name, param in self.target_model.named_parameters():
                    logger.log("target model parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            self._load_and_sync_ema_parameters_to_target_parameters()
            if self.args.sanity_check:
                for name, param in self.target_model.named_parameters():
                    logger.log("target model parameter after overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            self.target_model.requires_grad_(False)
            self.target_model.train()

            if self.args.use_fp16:
                self.target_model_param_groups_and_shapes = get_target_param_groups_and_shapes(
                    self.target_model.named_parameters(), self.model.named_parameters()
                )
                self.target_model_master_params = make_master_params(
                    self.target_model_param_groups_and_shapes
                )
            else:
                self.target_model_param_groups_and_shapes = list(self.target_model.named_parameters())
                self.target_model_master_params = list(target_model.parameters())
            for rate, params in zip(self.ema_rate, self.ema_params):
                if rate == 0.999:
                    logger.log(f"loading target model from 0.999 ema...")
                    update_ema(
                        self.target_model_master_params,
                        params,
                        rate=0.0,
                    )
                    if self.args.use_fp16:
                        master_params_to_model_params(
                            self.target_model_param_groups_and_shapes,
                            self.target_model_master_params,
                        )
            if self.args.sanity_check:
                for name, param in self.target_model.named_parameters():
                    logger.log("target model parameter after all: ", param.data.cpu().detach().reshape(-1)[:3])
                    break

        if teacher_model:
            #self._load_and_sync_teacher_parameters()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
        self.diffusion.teacher_model = teacher_model

        self.global_step = self.step
        self.initial_step = copy.deepcopy(self.step)
        if self.args.gpu_usage:
            nvidia_smi.nvmlInit()
            self.deviceCount = nvidia_smi.nvmlDeviceGetCount()
            self.print_gpu_usage('Before everything')

        if self.args.check_dm_performance:
            if not os.path.exists(self.args.dm_sample_path_seed_42):
                self.sampling(model=self.teacher_model, sampler='heun', teacher=True, step=18 if self.args.data_name.lower() == 'cifar10' else 40,
                              num_samples=self.args.eval_num_samples, batch_size=self.args.eval_batch,
                              rate=0.0, ctm=False, png=False, resize=False,
                              generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              sample_dir=self.args.dm_sample_path_seed_42)

        if self.args.data_name.lower() == 'cifar10':
            if dist.get_rank() == 0:
                print('Loading Inception-v3 model...')
                detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
                self.detector_kwargs = dict(return_features=True)
                self.feature_dim = 2048
                with dnnlib.util.open_url(detector_url, verbose=(0 == 0)) as f:
                    self.detector_net = pickle.load(f).to(dist_util.dev())
                with dnnlib.util.open_url(self.args.ref_path) as f:
                    ref = dict(np.load(f))
                self.mu_ref = ref['mu']
                self.sigma_ref = ref['sigma']
                if self.args.check_dm_performance:
                    if self.args.ae_image_path_seed_42 != '':
                        self.ae_mu, self.ae_sigma = self.calculate_inception_stats(self.args.data_name,
                                                                                   self.args.ae_image_path_seed_42,
                                                                                   num_samples=self.args.eval_num_samples)
                    self.dm_mu, self.dm_sigma = self.calculate_inception_stats(self.args.data_name,
                                                                               self.args.dm_sample_path_seed_42,
                                                                               num_samples=self.args.eval_num_samples)
                    logger.log(f"DM FID-50k: {self.compute_fid(self.dm_mu, self.dm_sigma)}")
                    ref_files = glob.glob(os.path.join(self.args.dm_sample_path_seed_42, 'sample*.npz'))
                    ref_files.sort()
                    self.ref_images = []
                    for i, ref_file in enumerate(ref_files):
                        ref_images = np.load(ref_file)['arr_0']
                        self.ref_images.append(ref_images)
                    self.ref_images = np.concatenate(self.ref_images)
                    if self.args.ae_image_path_seed_42 != '':
                        logger.log(f"Regenerated DM Samples (by LSGM AE) FID-50k: {self.compute_fid(self.ae_mu, self.ae_sigma)}")
                        psnr, ssim = self.calculate_similarity_metrics(
                            self.args.ae_image_path_seed_42, num_samples=self.args.eval_num_samples, step=1,
                            rate=0.0, sampler='LSGM Auto-Encoder', log=False)
                        logger.log(f"Regenerated DM Samples (by LSGM AE) PSNR-50k: {psnr}, SSIM-10k: {ssim}")
        else:
            import tensorflow.compat.v1 as tf
            from cm.evaluator import Evaluator
            if dist.get_rank() == 0:
                config = tf.ConfigProto(
                    allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
                )
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.1
                self.evaluator = Evaluator(tf.Session(config=config), batch_size=100)

                self.ref_acts = self.evaluator.read_activations(self.args.ref_path)
                self.ref_stats, self.ref_stats_spatial = self.evaluator.read_statistics(self.args.ref_path, self.ref_acts)
                if self.args.check_dm_performance:
                    if os.path.exists(os.path.join(os.path.join(self.args.dm_sample_path_seed_42, 'single_npz'), f'stats')):
                        with open(os.path.join(os.path.join(self.args.dm_sample_path_seed_42, 'single_npz'), f'acts'), 'rb') as f:
                            sample_acts = pickle.load(f)
                            sample_acts = (sample_acts['acts'], sample_acts['acts_spatial'])
                        with open(os.path.join(os.path.join(self.args.dm_sample_path_seed_42, 'single_npz'), f'stats'), 'rb') as f:
                            sample_stats = pickle.load(f)
                            sample_stats, sample_stats_spatial = (sample_stats['stats'], sample_stats['stats_spatial'])
                    else:
                        sample_acts, sample_stats, sample_stats_spatial = self.calculate_inception_stats(self.args.data_name,
                                                                        self.args.dm_sample_path_seed_42,
                                                                        num_samples=self.args.eval_num_samples)
                    logger.log("Inception Score-50k:", self.evaluator.compute_inception_score(sample_acts[0]))
                    logger.log("FID-50k:", sample_stats.frechet_distance(self.ref_stats))
                    logger.log("sFID-50k:", sample_stats_spatial.frechet_distance(self.ref_stats_spatial))
                    prec, recall = self.evaluator.compute_prec_recall(self.ref_acts[0], sample_acts[0])
                    logger.log("Precision:", prec)
                    logger.log("Recall:", recall)
                    if self.args.gpu_usage:
                        self.print_gpu_usage('After computing DM FIDs')
                    #self.evaluator.sess.close()
            gc.collect()
            th.cuda.empty_cache()
            tf.reset_default_graph()
            if self.args.gpu_usage:
                self.print_gpu_usage('After emptying cache')


    def print_gpu_usage(self, prefix=''):
        for i in range(self.deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            logger.log(
                f"{prefix} |Device {i}| Mem Free: {mem.free / 1024 ** 2:5.2f}MB / {mem.total / 1024 ** 2:5.2f}MB | gpu-util: {util.gpu / 100.0:3.1%} | gpu-mem: {util.memory / 100.0:3.1%} |")


    def _load_and_sync_ema_parameters_to_target_parameters(self):
        if dist.get_rank() == 0:
            for rate, params in zip(self.ema_rate, self.ema_params):
                if rate == self.args.start_ema: # 0.999
                    logger.log(f"loading target model from {self.args.start_ema} ema...")
                    state_dict = self.mp_trainer.master_params_to_state_dict(params)
                    self.target_model.load_state_dict(state_dict)
            logger.log(f"end loading target model from {self.args.start_ema} ema...")

        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())

    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace("model", "target_model")
            resume_target_checkpoint = os.path.join(path, target_name)
            if bf.exists(resume_target_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    f"loading target model from checkpoint: {resume_target_checkpoint}..."
                )
                if dist.get_world_size() > 1:
                    state_dict = th.load(resume_target_checkpoint, map_location="cpu")
                else:
                    state_dict = dist_util.load_state_dict(
                        resume_target_checkpoint, map_location=dist_util.dev()
                    )
                self.target_model.load_state_dict(state_dict, strict=False)

        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())

    def _load_and_sync_teacher_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            teacher_name = name.replace("model", "teacher_model")
            resume_teacher_checkpoint = os.path.join(path, teacher_name)

            if bf.exists(resume_teacher_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    f"loading teacher model from checkpoint: {resume_teacher_checkpoint}..."
                )
                if dist.get_world_size() > 1:
                    state_dict = th.load(resume_teacher_checkpoint, map_location="cpu")
                else:
                    state_dict = dist_util.load_state_dict(
                        resume_teacher_checkpoint, map_location=dist_util.dev()
                    )
                self.teacher_model.load_state_dict(state_dict)#, strict=False)

        dist_util.sync_params(self.teacher_model.parameters())
        dist_util.sync_params(self.teacher_model.buffers())

    def run_loop(self):
        if self.args.gpu_usage:
            self.print_gpu_usage('Before training')
        saved = False
        while (
            self.step < self.args.lr_anneal_steps
            or self.global_step < self.total_training_steps
        ):
            batch, cond = next(self.data)
            if self.args.large_log:
                print("batch size: ", batch.shape)
                print("rank: ", dist.get_rank())
            if self.args.intermediate_samples:
                if dist.get_rank() == 0:
                    if self.step == self.initial_step + 10 or (self.step % self.args.sample_interval == self.args.sample_interval - 1):
                        if self.args.training_mode.lower() == 'ctm':
                            if self.args.consistency_weight > 0.:
                                self.sampling(model=self.ddp_model, sampler='exact')
                                self.sampling(model=self.ddp_model, sampler='exact', step=2)
                                self.sampling(model=self.ddp_model, sampler='exact', step=1)
                            else:
                                self.sampling(model=self.ddp_model, sampler='heun', ctm=True, teacher=True)
                        elif self.args.training_mode.lower() == 'cd':
                            self.sampling(model=self.ddp_model, sampler='onestep', step=1)
                    if self.step == self.initial_step + 10 and self.teacher_model != None:
                        self.sampling(model=self.teacher_model, sampler='heun', ctm=False, teacher=True)
            self.run_step(batch, cond)
            if self.args.gpu_usage:
                self.print_gpu_usage('After one step training')
            if self.args.large_log:
                print("mp trainer master parameter after one step update: ", self.mp_trainer.master_params[1].reshape(-1)[:3])
                for name, param in self.model.named_parameters():
                    print("model parameter after one step update: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                for name, param in self.target_model.named_parameters():
                    print("target model parameter after one step update: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            if self.args.check_ctm_denoising_ability:
                self.eval(step=18, sampler='heun', teacher=True, ctm=True, rate=0.0)
            if (
                self.global_step
                and self.args.eval_interval != -1
                and self.global_step % self.args.eval_interval == self.args.eval_interval - 1
                #and self.step - self.initial_step > 10
                or self.step == self.args.lr_anneal_steps - 1
                or self.global_step == self.total_training_steps - 1
            ):
                if self.args.gpu_usage:
                    self.print_gpu_usage('Before emptying cache in evaluation 1')
                gc.collect()
                th.cuda.empty_cache()
                if self.args.gpu_usage:
                    self.print_gpu_usage('After emptying cache in evaluation 1')
                model_state_dict = self.model.state_dict()
                if self.args.linear_probing:
                    self.eval(step=18, sampler='heun', teacher=True, ctm=True, rate=0.0,
                              generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              delete=True)
                self.evaluation(0.0)
                logger.log('Evaluation with model parameter end')
                for rate, params in zip(self.ema_rate, self.ema_params):
                    if not self.args.compute_ema_fids:
                        if rate != 0.999:
                            continue
                    state_dict = self.mp_trainer.master_params_to_state_dict(params)
                    self.model.load_state_dict(state_dict, strict=False)
                    self.evaluation(rate)
                    logger.log(f'Evaluation with {rate}-EMA model parameter end')
                self.model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict, state_dict
                if self.args.gpu_usage:
                    self.print_gpu_usage('Before emptying cache in evaluation 2')
                gc.collect()
                th.cuda.empty_cache()
                if self.args.gpu_usage:
                    self.print_gpu_usage('After emptying cache in evaluation 2')
            dist.barrier()
            if (
                    self.global_step
                    and self.args.eval_interval != -1
                    and self.global_step % self.args.save_check_period == self.args.save_check_period - 1
                    #and self.step - self.initial_step > 10000
                    or self.step == self.args.lr_anneal_steps - 1
                    or self.global_step == self.total_training_steps - 1
            ):
                gc.collect()
                th.cuda.empty_cache()
                model_state_dict = self.model.state_dict()
                for rate, params in zip(self.ema_rate, self.ema_params):
                    if rate == 0.999:
                        state_dict = self.mp_trainer.master_params_to_state_dict(params)
                        self.model.load_state_dict(state_dict, strict=False)
                        fid = self.save_check(rate)
                        if dist.get_rank() == 0:
                            assert fid != None
                            self.fids.append(fid)
                            save_ckpt = (self.fids[-1] == np.min(self.fids))
                            logger.log("FID by iteration (NFE 1, EMA 0.999): ", self.fids)
                self.model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict, state_dict
                if dist.get_rank() == 0:
                    if save_ckpt:
                        self.save(save_full=False)
                gc.collect()
                th.cuda.empty_cache()
            if self.args.large_log:
                print("mp trainer master parameter after sampling: ",
                      self.mp_trainer.master_params[1].reshape(-1)[:3])
                for name, param in self.model.named_parameters():
                    print("model parameter after sampling: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                for name, param in self.target_model.named_parameters():
                    print("target model parameter after sampling: ", param.data.cpu().detach().reshape(-1)[:3])
                    break

            saved = False
            if (
                self.global_step
                and self.args.save_interval != -1
                and self.global_step % self.args.save_interval == 0
            ):
                self.save()
                if self.discriminator != None:
                    self.d_save()
                saved = True
                gc.collect()
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.global_step % self.args.log_interval == 0:
                logger.dumpkvs()
                logger.log(datetime.datetime.now().strftime("SONY-%Y-%m-%d-%H-%M-%S"))
            if self.args.large_log:
                print("mp trainer master parameter after saving: ",
                      self.mp_trainer.master_params[1].reshape(-1)[:3])
                for name, param in self.model.named_parameters():
                    print("model parameter after saving: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                for name, param in self.target_model.named_parameters():
                    print("target model parameter after saving: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                print(f"0.999 ema param after overriding (should be same to the target parameter): ",
                      self.ema_params[0][1].reshape(-1)[:3])

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()
            if self.discriminator != None:
                self.d_save()

    def save_check(self, rate):
        if self.args.training_mode.lower() == 'ctm':
            assert rate == 0.999
            #fid = self.eval(step=1, rate=rate, ctm=True, delete=True, out=True)
            fid = self.eval(step=1, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                                  class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True, out=True)
            return fid

    def evaluation(self, rate):
        if self.args.training_mode.lower() == 'ctm':
            if self.args.eval_fid:
                self.eval(step=1, rate=rate, ctm=True, delete=True)
            if self.args.eval_similarity:
                self.eval(step=1, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                                  class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True)
            if self.args.eval_fid:
                self.eval(step=2, rate=rate, ctm=True, delete=True)
            if self.args.eval_similarity:
                self.eval(step=2, rate=rate, ctm=True,
                          generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                          class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True)
            if self.args.compute_ema_fids:
                if self.args.eval_fid:
                    self.eval(step=4, rate=rate, ctm=True, delete=True)
                if self.args.eval_similarity:
                    self.eval(step=4, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              metric='similarity', delete=True)
            if self.args.large_nfe_eval:
                step = 18 if self.args.data_name.lower() == 'cifar10' else 40
                if self.args.eval_fid:
                    self.eval(step=step, rate=rate, ctm=True, delete=True)
                if self.args.eval_similarity:
                    self.eval(step=step, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              metric='similarity', delete=True)

        elif self.args.training_mode.lower() == 'cm':
            if self.args.eval_fid:
                self.eval(step=1, sampler='onestep', rate=rate, ctm=False, delete=True)

    def run_step(self, batch, cond):
        if self.args.large_log:
            print("mp trainer master parameter before update: ", self.mp_trainer.master_params[1].reshape(-1)[:3])
            for name, param in self.model.named_parameters():
                print("model parameter before update: ", param.data.cpu().detach().reshape(-1)[:3])
                break
            for name, param in self.target_model.named_parameters():
                print("target model parameter before update: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        self.forward_backward(batch, cond)
        if self.discriminator == None:
            took_step = self.mp_trainer.optimize(self.opt)
            if took_step:
                self._update_ema()
                if self.target_model:
                    self._update_target_ema()
                self.step += 1
                self.global_step += 1
        else:
            if self.step % self.args.g_learning_period == 0:
                took_step = self.mp_trainer.optimize(self.opt)
            else:
                took_step = self.d_mp_trainer.optimize(self.d_opt)
            # print(self.step, took_step)
            if took_step:
                if self.step % self.args.g_learning_period == 0:
                    self._update_ema()
                    if self.target_model:
                        self._update_target_ema()
                self.step += 1
                self.global_step += 1
        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                self.mp_trainer.master_params,
                rate=target_ema,
            )
            if self.args.use_fp16:
                master_params_to_model_params(
                    self.target_model_param_groups_and_shapes,
                    self.target_model_master_params,
                )

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        if self.discriminator != None:
            self.d_mp_trainer.zero_grad()
        num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.args.num_heun_step)]
        if self.args.gan_specific_time:
            gan_num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.args.gan_num_heun_step,
                                                                  heun_step_strategy=self.args.gan_heun_step_strategy)]
        diffusion_training_ = [np.random.rand() < self.args.diffusion_training_frequency]
        gan_training_ = [np.random.rand() < self.args.gan_training_frequency]
        dist.broadcast_object_list(num_heun_step, 0)
        if self.args.gan_specific_time:
            dist.broadcast_object_list(gan_num_heun_step, 0)
        dist.broadcast_object_list(diffusion_training_, 0)
        dist.broadcast_object_list(gan_training_, 0)
        num_heun_step = num_heun_step[0]
        if self.args.gan_specific_time:
            gan_num_heun_step = gan_num_heun_step[0]
        else:
            gan_num_heun_step = -1
        diffusion_training_ = diffusion_training_[0]
        gan_training_ = gan_training_[0]

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            compute_losses = functools.partial(
                self.diffusion.ctm_losses,
                step=self.step,
                model=self.ddp_model,
                x_start=micro,
                model_kwargs=micro_cond,
                target_model=self.target_model,
                discriminator=self.ddp_discriminator,
                init_step=self.initial_step,
                ctm=True if self.training_mode.lower() == 'ctm' else False,
                num_heun_step=num_heun_step,
                gan_num_heun_step=gan_num_heun_step,
                diffusion_training_=diffusion_training_,
                gan_training_=gan_training_,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                if self.step % self.args.g_learning_period == 0:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()
                else:
                    with self.ddp_discriminator.no_sync():
                        losses = compute_losses()

            if 'consistency_loss' in list(losses.keys()):
                # print("Consistency learning")
                loss = self.args.consistency_weight * losses["consistency_loss"].mean()

                if 'd_loss' in list(losses.keys()):
                    if self.args.large_log:
                        print("GAN learning, ", self.args.discriminator_weight, losses['d_loss'].mean())
                    loss = loss + self.args.discriminator_weight * losses['d_loss'].mean()
                if 'denoising_loss' in list(losses.keys()):
                    loss = loss + self.args.denoising_weight * losses['denoising_loss'].mean()
                log_loss_dict({k: v.view(-1) for k, v in losses.items()})
                if self.args.sanity_check:
                    print("rank: ", dist.get_rank())
                    for name, param in self.model.named_parameters():
                        print("model parameter gradient for current microbatch: ",
                              th.autograd.grad(outputs=(2**self.mp_trainer.lg_loss_scale) * loss, inputs=param, retain_graph=True)[0].reshape(-1)[:3])
                        break
                self.mp_trainer.backward(loss)

            elif 'd_loss' in list(losses.keys()):
                assert self.step % self.args.g_learning_period != 0
                loss = (losses["d_loss"]).mean()
                self.d_mp_trainer.backward(loss)
                if self.args.large_log:
                    for param in self.discriminator.parameters():
                        try:
                            print("discriminator param data, grad: ", param.grad.reshape(-1)[:3])
                        except:
                            print("discriminator param grad: ", param.grad)
                        break

            elif 'denoising_loss' in list(losses.keys()):
                loss = losses['denoising_loss'].mean()
                log_loss_dict({k: v.view(-1) for k, v in losses.items()})
                self.mp_trainer.backward(loss)
            if self.args.sanity_check:
                print("rank: ", dist.get_rank())
                for name, param in self.model.named_parameters():
                    print("model parameter gradient across all microbatch: ", param.grad.cpu().detach().reshape(-1)[:3])
                    break

    @th.no_grad()
    def eval(self, step=1, sampler='exact', teacher=False, ctm=False, rate=0.999, generator=None, class_generator=None, metric='fid', delete=False, out=False):
        model = self.model
        sample_dir = f"{self.step}_{sampler}_{step}_{rate}"
        if generator != None:
            sample_dir = sample_dir + "_seed_42"
        self.sampling(model=model, sampler=sampler, teacher=teacher, step=step,
                      num_samples=self.args.eval_num_samples, batch_size=self.args.eval_batch,
                      rate=rate, ctm=ctm, png=True, resize=False, generator=generator,
                      class_generator=class_generator, sample_dir=sample_dir)
        gc.collect()
        th.cuda.empty_cache()
        if dist.get_rank() == 0:
            if self.args.data_name.lower() == 'cifar10':
                if metric == 'fid':
                    mu, sigma = self.calculate_inception_stats(self.args.data_name,
                                                               os.path.join(get_blob_logdir(), sample_dir),
                                                               num_samples=self.args.eval_num_samples)
                    logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
                               f" FID-{self.args.eval_num_samples // 1000}k: {self.compute_fid(mu, sigma)}")
                if metric == 'similarity':
                    mu, sigma = self.calculate_inception_stats(self.args.data_name,
                                                               os.path.join(get_blob_logdir(), sample_dir),
                                                               num_samples=self.args.eval_num_samples)
                    logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) seed 42 EMA {rate}"
                               f" FID-{self.args.eval_num_samples // 1000}k: {self.compute_fid(mu, sigma)}")
                if self.args.check_dm_performance:
                    logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
                               f" FID-{self.args.eval_num_samples // 1000}k compared with DM: {self.compute_fid(mu, sigma, self.dm_mu, self.dm_sigma)}")
                    self.calculate_similarity_metrics(os.path.join(get_blob_logdir(), sample_dir),
                                                      num_samples=self.args.eval_num_samples, step=step, rate=rate)
                if delete:
                    shutil.rmtree(os.path.join(get_blob_logdir(), sample_dir))
                if out:
                    return self.compute_fid(mu, sigma)
            else:
                sample_acts, sample_stats, sample_stats_spatial = self.calculate_inception_stats(self.args.data_name,
                                                                             bf.join(get_blob_logdir(), sample_dir),
                                                                             num_samples=self.args.eval_num_samples)
                logger.log(f"Inception Score-{self.args.eval_num_samples // 1000}k:", self.evaluator.compute_inception_score(sample_acts[0]))
                logger.log(f"FID-{self.args.eval_num_samples // 1000}k:", sample_stats.frechet_distance(self.ref_stats))
                logger.log(f"sFID-{self.args.eval_num_samples // 1000}k:", sample_stats_spatial.frechet_distance(self.ref_stats_spatial))
                prec, recall = self.evaluator.compute_prec_recall(self.ref_acts[0], sample_acts[0])
                logger.log("Precision:", prec)
                logger.log("Recall:", recall)
                #self.evaluator.sess.close()
                #tf.reset_default_graph()

    def save(self, save_full=True):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{self.global_step:06d}.pt"
                else:
                    filename = f"ema_{rate}_{self.global_step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            if not save_full:
                if rate == 0.999:
                    save_checkpoint(rate, params)
            else:
                save_checkpoint(rate, params)
        if save_full:
            logger.log("saving optimizer state...")
            if dist.get_rank() == 0:
                with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"opt{self.global_step:06d}.pt"),
                    "wb",
                ) as f:
                    th.save(self.opt.state_dict(), f)

            if dist.get_rank() == 0:
                if self.target_model:
                    logger.log("saving target model state")
                    filename = f"target_model{self.global_step:06d}.pt"
                    with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                        th.save(self.target_model.state_dict(), f)
                if self.teacher_model and self.training_mode == "progdist":
                    logger.log("saving teacher model state")
                    filename = f"teacher_model{self.global_step:06d}.pt"
                    with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                        th.save(self.teacher_model.state_dict(), f)

            # Save model parameters last to prevent race conditions where a restart
            # loads model at step N, but opt/ema state isn't saved for step N.
            save_checkpoint(0, self.mp_trainer.master_params)
            dist.barrier()

    def d_save(self):
        logger.log("saving d_optimizer state...")
        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"d_opt{self.global_step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.d_opt.state_dict(), f)
            with bf.BlobFile(bf.join(get_blob_logdir(), f"d_model{self.global_step:06d}.pt"), "wb") as f:
                th.save(self.d_mp_trainer.master_params_to_state_dict(self.d_mp_trainer.master_params), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        dist.barrier()

    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(f"{key} mean", values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        logger.logkv_mean(f"{key} std", values.std().item())
        #for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #    quartile = int(4 * sub_t / diffusion.num_timesteps)
        #    logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
