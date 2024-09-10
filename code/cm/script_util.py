import argparse
from uu import Error
from haiku import dropout
import numpy as np
import torch

from .karras_diffusion import KarrasDenoiser
from cm.resample import create_named_schedule_sampler
from cm.classifier import EncoderUNetModel

import blobfile as bf
import os
from torchvision.utils import make_grid, save_image

def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=4,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )

def ctm_data_defaults(data_name):
    return dict(
        train_classes=-1,
        type='png',
        sigma_data=0.5 if data_name == 'cifar10' else 0.5, # assumes 0.5 for other datsets, but preferable if their real sigma_data value is actually calculated.
        deterministic=False,
        num_classes=10,
    )

def ctm_loss_defaults(data_name):
    return dict(
        # CTM hyperparams
        consistency_weight=1.0,
        ctm_estimate_outer_type='target_model_sg',
        ctm_estimate_inner_type='model',
        # ctm_target_inner_type='model_sg',
        ctm_target_inner_type='target_model_sg', # change made on 3rd July (19th aug update: seems to be correct)
        ctm_target_matching=False,
        # sample_s_strategy='uniform',
        # sample_s_strategy='smallest', # 7/28 update. This is in accordance to the improved loss that I devised.
        sample_s_strategy='smallest', # 20 Aug This is the same as 'sigma_s_is_zero'
        heun_step_strategy='weighted',
        heun_step_multiplier=1.0,
        outer_parametrization='euler',
        inner_parametrization='edm' if data_name == 'cifar10' else 'edm',
        time_continuous=False,
        # self_learn=True, # True if no teacher diffusion model; False if there is a teacher diffusion model
        self_learn=False,
        self_learn_iterative=False,
        target_matching=False,

        # DSM hyperparams
        diffusion_training=True,
        apply_adaptive_weight=False,
        denoising_weight=1.,
        bridge_denoising_weight=1.,
        q_part_denoising_weight=1.,
        diffusion_mult = 0.7,
        # diffusion_schedule_sampler='halflognormal',
        diffusion_schedule_sampler='lognormal', # 19th Aug (This is according to EDM as far as I have seen.)
        diffusion_training_frequency=1.,

        # GAN hyperparams
        d_lr=0.002,
        gan_training=False,
        gan_real_free=True,
        discriminator_weight=1.0,
        discriminator_start_itr=0,
        use_d_fp16=False,
        d_architecture='StyleGAN-XL',
        g_learning_period=1,
        gan_fake_outer_type='no',
        gan_fake_inner_type='',
        gan_real_inner_type='',
        gan_target_matching=False,
        data_augment=True,
        d_backbone=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
        d_apply_adaptive_weight=True,
        shift_ratio=0.125,
        cutout_ratio=0.2,
        gan_training_frequency=1.,
        gaussian_filter=False,
        blur_fade_itr=1000,
        blur_init_sigma=2,
        prob_aug=1.0,
        gan_different_augment=False,
        gan_num_heun_step=17 if data_name == 'cifar10' else 39,
        use_x0_as_denoised_in_solver=True, # If True, it follows OpenAI's CM code.
        gan_heun_step_strategy='uniform',
        gan_specific_time=False,
        gan_low_res_train=False,
    )

def ctm_train_defaults(data_name):
    return dict(
        beta_min=0.1,
        beta_max=20.,
        multiplier=1.,
        num_heun_step=17 if data_name == 'cifar10' else 39,
        num_heun_step_random=True,

        # Network architecture
        edm_nn_ncsn=False,
        edm_nn_ddpm=True if data_name == 'cifar10' else False,
        in_channels=3,
        linear_probing=False,
        target_subtract=False,
        
        is_I2I=False, # False -> N2I
    )

def ctm_eval_defaults(data_name):
    return dict(
        intermediate_samples=False,
        sampling_batch=64,
        sample_interval=1000 if data_name == 'cifar10' else 1000,
        sampling_steps=18 if data_name == 'cifar10' else 40,
        eval_interval=1000,
        eval_num_samples=50000,
        # eval_num_samples=5000,
        eval_batch=500,
        #ref_path='/home/dongjun/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz' if data_name == 'cifar10' else "",
        # ref_path='/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz' if data_name == 'cifar10' \
        #     else "/home/fp084243/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz",
        ref_path='/root/code/pretrained/cifar10-32x32.npz' if data_name == 'cifar10' \
                else "/home/fp084243/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz",
        ref_feat_path='',
        large_log=False,
        compute_ema_fids=False,
        #dm_sample_path_seed_42='/data2/dongjun/EighthArticleExperimentalResults/CIFAR10/DM/EDM-VP/fp16-seed-42/edm_heun_sampler_18_steps_ond-vp_itrs_model_ema' if data_name == 'cifar10' else "",
        dm_sample_path_seed_42='/root/code/results/CIFAR10/DM/heun_18_seed_42_ver2' if data_name == 'cifar10' else "/root/code/results/some_dataset/heun_18_seed_42_ver2",
        ae_image_path_seed_42='',
        eval_seed=42,
        eval_fid=False,
        eval_similarity=True,
        save_png=False,
        check_ctm_denoising_ability=False,
        check_dm_performance=True,
        sanity_check=False,
        save_period=1000 if data_name == 'cifar10' else 1000,
        clip_denoised=True,
        clip_output=True,
        gpu_usage=False,
        large_nfe_eval=False,
        eval_sampler='exact' if data_name == 'cifar10' else 'hybrid',
        churn_step_ratio=0. if data_name == 'cifar10' else 0.33,
        # guidance_scale=0.5,
        guidance_scale=0.,
        qpart_loss=False,
        gamma=1.,
        # use_milstein_method=False,
    )

def cm_train_defaults(data_name):
    return dict(
        model_path='',
        #teacher_model_path="/home/dongjun/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl" if data_name == 'cifar10' else "",
        # teacher_model_path="/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl" if data_name == 'cifar10' else "",
        # teacher_model_path="/root/code/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
        teacher_model_path="/root/code/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
        # teacher_dropout=0.0 if data_name == 'cifar10' else 0.1,
        teacher_dropout=0.3 if data_name == 'cifar10' else 0.2 if data_name.startswith('edges2') else 0.3,
        training_mode="ctm",
        target_ema_mode="fixed",
        # scale_mode="fixed",
        scale_mode='ict_exp', # 20th Aug
        # total_training_steps=600000,
        total_training_steps=400000,
        # start_ema=0.999,
        start_ema=0.0, # 20th Aug
        # start_scales=18 if data_name == 'cifar10' else 40,
        # end_scales=18 if data_name == 'cifar10' else 40,
        start_scales=10 if data_name == 'cifar10' else 40, # 20th aug start_scales
        end_scales=1280 if data_name == 'cifar10' else 40, # 20th aug end_scales
        distill_steps_per_iter=50000,
        loss_norm="lpips",
        do_xT_precond=True,
    )

def model_and_diffusion_defaults(data_name, is_I2I: bool):
    """
    Defaults for image training.
    """
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_data = 0.5 if data_name == 'cifar10' else 0.5 # assumes 0.5 for other datsets, but preferable if their real sigma_data value is actually calculated.
    if is_I2I:
        sigma_data_end = sigma_data
        cov = 0.5 * (sigma_data**2)
    else:
        # assume 0.5 for other datasets, but preferable if actual sigma_data_end is calculated.
        sigma_data_end = np.sqrt(sigma_data**2 + sigma_max**2)
        cov = sigma_data**2
    
    res = dict(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=7,
        image_size=32 if data_name == 'cifar10' else 64 if data_name.startswith('edges2') else 256,
        num_channels=192,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        # dropout=0.3 if data_name == 'cifar10' else 0.2 if data_name.startswith('edges2') else 0.3,
        class_cond=False if data_name == 'cifar10' or data_name.startswith('edges2') else True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        attention_type='flash',
        learn_sigma=False,
        weight_schedule="uniform", # CTM Weight schedule is set to "ict" in the experiments. The weight is therefore = 1/(t-s).
        weight_schedule_multiplier=1.,
        diffusion_weight_schedule="karras_weight",
        rescaling=False,
        
        condition_mode=None if data_name == 'cifar10' else 'concat',
        pred_mode=None,
        beta_d=2,
        beta_min=0.1,
        cov_xy=cov,
        sigma_data_end=sigma_data_end,
    )
    return res

def train_defaults(data_name):
    """
        Defaults for model training.
    """
    res = dict(
        out_dir="",
        #data_dir="/home/dongjun/EighthArticleExperimentalResults/CIFAR10/train" if data_name == 'cifar10' else "",
        data_dir="/root/data/cifar10/" if data_name == 'cifar10' else "",
        schedule_sampler="uniform",
        # schedule_sampler="ict",
        lr=0.0004 if data_name == 'cifar10' else 0.000008,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=128 if data_name == 'cifar10' else 2048,
        batch_size=64,
        microbatch=128 if data_name.lower() == 'cifar10' else -1,  # -1 disables microbatches
        # ema_rate="0.999,0.9999" if data_name == 'cifar10' else "0.999,0.9999,0.9999432189950708",
        ema_rate="0.999,0.9999" if data_name == 'cifar10' else
                 "0.9999" if data_name.startswith('edges2') else 
                 "0.999,0.9999,0.9999432189950708",
        # comma-separated list of EMA values
        log_interval=1000,
        save_interval=1000000,
        save_check_period=10000000,
        resume_checkpoint="",
        use_fp16=True,
        use_bf16=False,
        fp16_scale_growth=1e-3,
        bf16_scale_growth=1e-3,
        device_id=4,
        num_workers=2,
        use_MPI=True,
        skip_final_ctm_step=False,
        traditional_ctm=False, # 23rd Aug. Set it to True if one wants to run the original CTM experiments.
    )
    return res

def save(x: torch.Tensor, save_dir, name, npz=False):
    nrow = int(np.sqrt(x.shape[0]))
    image_grid = make_grid((x + 1.) / 2., nrow, padding=2)
    with bf.BlobFile(os.path.join(save_dir, f"{name}.png"), "wb") as fout:
        save_image(image_grid, fout)
    if npz:
        sample = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.detach().cpu()
        os.makedirs(os.path.join(save_dir, 'targets'), exist_ok=True)
        np.savez(os.path.join(save_dir, f"targets/{name}.npz"), sample.numpy())


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

def create_model_and_diffusion(args, feature_extractor=None, discriminator_feature_extractor=None, teacher=False):
    schedule_sampler = create_named_schedule_sampler(args, args.schedule_sampler, args.start_scales)
    diffusion_schedule_sampler = create_named_schedule_sampler(args, args.diffusion_schedule_sampler, args.start_scales)

    # Added on the 13th of May:
    # assert args.data_name.lower() == 'cifar10', f"e2h or e2s need to get songunet as well, but that won't happen in the code currently! It is TODO!!"
    
    if args.data_name.lower() == 'cifar10':
        from cm.networks import EDMPrecond_CTM
        model = EDMPrecond_CTM(img_resolution=args.image_size, img_channels=3,
                               label_dim=1000 if args.data_name.lower() == 'imagenet64' else 10 if args.class_cond else 0, use_fp16=args.use_fp16,
                               use_bf16=args.use_bf16,
                               sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                               sigma_data=args.sigma_data, model_type='SongUNet' if args.data_name.lower() == 'cifar10' else 'DhariwalUNet',
                               teacher=teacher, teacher_model_path=args.teacher_model_path or args.model_path,
                               training_mode=args.training_mode, arch='ddpmpp' if args.data_name.lower() == 'cifar10' else 'adm',
                               linear_probing=args.linear_probing, condition_mode=args.condition_mode,
                               sigma_data_end=args.sigma_data_end, cov_xy=args.cov_xy, inner_parametrization=args.inner_parametrization,
                               dropout=args.dropout,
                               )
        
    elif args.data_name.lower().startswith('edges'):
        # print(f"e2h or e2s need to get songunet as well, but that won't happen in the code currently! It is TODO!!")
        # raise NotImplementedError("not made yet for edges2 datasets!")

        from cm.networks import EDMPrecond_CTM
        model = EDMPrecond_CTM(img_resolution=args.image_size, img_channels=3,
            label_dim=0, use_fp16=args.use_fp16,
            use_bf16=args.use_bf16,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            sigma_data=args.sigma_data, model_type='SongUNet',
            teacher=teacher, teacher_model_path=args.teacher_model_path or args.model_path,
            training_mode=args.training_mode, arch='ddpmpp' if args.data_name.lower().startswith('edges') else 'adm',
            linear_probing=args.linear_probing, condition_mode=args.condition_mode,
            sigma_data_end=args.sigma_data_end, cov_xy=args.cov_xy, inner_parametrization=args.inner_parametrization,
            dropout=args.dropout,
            )
    else:
        model = create_model(
            args,
            args.image_size,
            args.num_channels,
            args.num_res_blocks,
            channel_mult=args.channel_mult,
            learn_sigma=args.learn_sigma,
            class_cond=args.class_cond,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=args.num_heads_upsample,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            resblock_updown=args.resblock_updown,
            use_fp16=args.use_fp16,
            use_bf16=args.use_bf16,
            use_new_attention_order=args.use_new_attention_order,
            training_mode=('teacher' if teacher else args.training_mode),
            attention_type=args.attention_type,
        )
        
    diffusion = KarrasDenoiser(
        args=args, schedule_sampler=schedule_sampler,
        diffusion_schedule_sampler=diffusion_schedule_sampler,
        feature_extractor=feature_extractor,
        discriminator_feature_extractor=discriminator_feature_extractor,
        pred_mode=args.pred_mode, beta_d=args.beta_d, beta_min=args.beta_min,
    )
    return model, diffusion

def create_model(
    args,
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_bf16=False,
    use_new_attention_order=False,
    training_mode='',
    attention_type='flash',
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    from .unet import UNetModel
    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(args.num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        training_mode=training_mode,
        attention_type=attention_type,
    )


def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        # assert target_ema_mode == "fixed" and scale_mode == "ict_exp", "" # <- 19th Aug (scale mode is "fixed" for the org CM exps.)
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "ict_exp":
            target_ema = start_ema
            
            total_training_steps_prime = np.floor(
                total_steps
                / (np.log2(np.floor(end_scales / start_scales)) + 1)
            )
            num_timesteps = start_scales * np.power(
                2, np.floor(step / total_training_steps_prime)
            )
            scales = min(num_timesteps, end_scales) + 1
            
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError
        
        # print('scales', int(scales))
        return float(target_ema), int(scales)

    return ema_and_scales_fn


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
