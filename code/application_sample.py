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
from cm.random_util import get_generator
from cm.sample_util import karras_sample, application_sample
import blobfile as bf
from torchvision.utils import make_grid, save_image
#import classifier_lib


def main():
    args = create_argparser().parse_args()

    if args.use_MPI:
        dist_util.setup_dist(args.device_id)
    else:
        dist_util.setup_dist_without_MPI(args.device_id)

    logger.configure(args, dir=args.out_dir)

    logger.log("creating model and diffusion...")

    if args.pretrained_feature_extractor and args.scale:
        classifier = create_classifier(**args_to_dict(args, list(classifier_defaults().keys()) + ['image_size']))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.to(dist_util.dev())
        if args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
    else:
        classifier = None

    if args.training_mode == 'edm':
        model, diffusion = create_model_and_diffusion(args, teacher=True)
    else:
        model, diffusion = create_model_and_diffusion(args)

    try:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location=dist_util.dev())
        )
    except:
        try:
            model.load_state_dict(
                dist_util.load_state_dict(args.model_path, map_location='cpu')
            )
        except:
            print("model path not loaded")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    elif args.sampler in ["exact", "gamma", "cm_multistep", "gamma_multistep", "stroke"]:
        try:
            ts = tuple(int(x) for x in args.ts.split(","))
        except:
            ts = []
    else:
        ts = None
    print("ts: ", ts)

    if args.stochastic_seed:
        args.eval_seed = np.random.randint(1000000)
    #generator = get_generator(args.generator, args.num_samples, args.seed)
    generator = get_generator(args.generator, args.eval_num_samples, args.eval_seed)

    r = np.random.randint(1000000)
    ref = args.ref_image_path.split('/')[-1].split('.')[0]

    step = args.model_path.split('.')[-2][-6:]
    try:
        ema = float(args.model_path.split('_')[-2])
        assert ema in [0.999, 0.9999, 0.9999432189950708]
    except:
        ema = 'model'
    if args.sampler in ['multistep', 'exact', 'cm_multistep']:
        out_dir = os.path.join(args.out_dir, f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{"".join([str(i) for i in ts])}')
    elif args.sampler in ["gamma"]:
        out_dir = os.path.join(args.out_dir, f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{"".join([str(i) for i in ts])}_ind1_{args.ind_1}_ind2_{args.ind_2}')
    elif args.sampler in ["gamma_multistep"]:
        out_dir = os.path.join(args.out_dir,
                               f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{"".join([str(i) for i in ts])}_gamma_{args.gamma}')

    else:
        out_dir = os.path.join(args.out_dir,
                                f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{args.gamma}_{ts[0]},{ts[1]}_numgrad_{args.num_gradient_descent}_scale_{args.scale}_{ref}_{r}')
    os.makedirs(out_dir, exist_ok=True)
    itr = 0
    eval_num_samples = 0
    while itr * args.batch_size < args.eval_num_samples:
        # org
        x_T = generator.randn(
            *(args.batch_size, args.in_channels, args.image_size, args.image_size),
            device=dist_util.dev()) * args.sigma_max
        # x_T = generator.randn(
        #     *(args.batch_size, args.in_channels, args.image_size, args.image_size),
        #     device=dist_util.dev()) * args.sigma_data_end
        #classes = generator.randint(0, 1000, (args.batch_size,))
        if args.large_log:
            print("x_T: ", x_T[0][0][0][0])
        current = time.time()
        model_kwargs = {}
        if args.class_cond:
            if args.train_classes >= 0:
                classes = th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=int) * int(args.train_classes)
            elif args.train_classes == -2:
                classes = [0, 1, 9, 11, 29, 31, 33, 55, 76, 89, 90, 130, 207, 250, 279, 281, 291, 323, 386, 387,
                           388, 417, 562, 614, 759, 789, 800, 812, 848, 933, 973, 980]
                assert args.batch_size % len(classes) == 0
                #print("!!!!!!!!!!!!!!: ", [x for x in classes for _ in range(args.batch_size // len(classes))])
                #model_kwargs["y"] = th.from_numpy(np.array([[[x] * (args.batch_size // len(classes)) for x in classes]]).reshape(-1)).to(dist_util.dev())
                classes = th.tensor([x for x in classes for _ in range(args.batch_size // len(classes))], device=dist_util.dev())
            else:
                classes = th.randint(
                    low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
                )
            model_kwargs["y"] = classes
            if args.large_log:
                print("classes: ", model_kwargs)
        from PIL import Image
        image = Image.open(args.ref_image_path)
        image = image.resize(
            tuple(args.image_size for x in image.size), resample=Image.BOX
        )
        images = th.from_numpy(np.array(image)).repeat(args.batch_size,1,1,1).to(dist_util.dev()).permute((0,3,1,2))
        images = images / 127.5 - 1.
        print(images.shape, images.min(), images.max())
        #images = th.from_numpy(np.load('/home/dongjun/EighthArticleExperimentalResults/Cat/sample_0-50.npz')['arr_0'][0:1] / 127.5 - 1.).to(dist_util.dev()).permute((0,3,1,2))
        #ts = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
        with th.no_grad():
            x_out, x_in = application_sample(
                images=images,
                diffusion=diffusion,
                model=model,
                shape=(args.batch_size, args.in_channels, args.image_size, args.image_size),
                steps=args.sampling_steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=False if args.data_name in ['church'] else True if args.training_mode=='edm' else args.clip_denoised,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=None,
                ts=ts,
                teacher = True if args.training_mode == 'edm' else False,
                clip_output=args.clip_output,
                ctm=True if args.training_mode.lower() == 'ctm' else False,
                ind_1=args.ind_1,
                ind_2=args.ind_2,
                gamma=args.gamma,
                generator_type=args.generator_type,
                classifier=classifier,
                num_gradient_descent=args.num_gradient_descent,
                scale=args.scale,
                out_dir=out_dir,
            )
            #print(x[0])

        sample = ((x_out + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        if dist.get_rank() == 0:
            sample = sample.detach().cpu()
            if args.large_log:
                print(f"{(itr-1) * args.batch_size} sampling complete...")

            if args.save_format == 'npz':
                if args.class_cond:
                    np.savez(os.path.join(out_dir, f"sample_out_{args.gamma}_{itr}.npz"), sample.numpy(), classes.detach().cpu().numpy())
                else:
                    np.savez(os.path.join(out_dir, f"stroke_sample_out_{args.gamma}_{itr}_{ts[0]},{ts[1]}_numgrad_{args.num_gradient_descent}_scale_{args.scale}.npz"), sample.numpy())
            #if args.save_format == 'png' or itr == 1:
            print("x range: ", x_out.min(), x_out.max())
            nrow = int(np.sqrt(sample.shape[0]))
            image_grid = make_grid((x_out + 1.) / 2., nrow, padding=2)
            if args.class_cond:
                with bf.BlobFile(os.path.join(out_dir, f"class_{args.train_classes}_sample_out_{args.gamma}_{itr}.png"), "wb") as fout:
                    save_image(image_grid, fout)
            else:
                with bf.BlobFile(os.path.join(out_dir, f"stroke_sample_out_{args.gamma}_{itr}_{ts[0]},{ts[1]}_numgrad_{args.num_gradient_descent}_scale_{args.scale}.png"), "wb") as fout:
                    save_image(image_grid, fout)

        eval_num_samples += sample.shape[0]
        if args.large_log:
            print(f"sample {eval_num_samples} time {time.time() - current} sec")
        itr += 1

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
        #data_name='imagenet64',
        data_name='afhq',
        #schedule_sampler="lognormal",
        ind_1=0,
        ind_2=0,
        gamma=0.5,
        ts="",
        ref_image_path="",
        generator_type='dummy',
        num_gradient_descent=1,
        pretrained_feature_extractor=False,
        classifier_path="",
        scale=1.0,
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
