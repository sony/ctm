import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np
from cm import logger
import torch.nn as nn
import cm.dist_util as dist_util
from pg_modules.diffaug import DiffAugment
import cm.script_util as script_util
import torchvision

def load_feature_extractor(args, eval=True):
    feature_extractor = None
    if args.loss_norm == 'lpips':
        from piq import LPIPS
        feature_extractor = LPIPS(replace_pooling=True, reduction="none")
    elif args.loss_norm == 'cnn_vit':
        from pg_modules.projector import F_RandomProj
        backbones = ['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0']
        feature_extractor = []
        backbone_kwargs = {'im_res': args.image_size}
        for i, bb_name in enumerate(backbones):
            feat = F_RandomProj(bb_name, **backbone_kwargs)
            feature_extractor.append([bb_name, feat])
        feature_extractor = nn.ModuleDict(feature_extractor)
        feature_extractor = feature_extractor.train(False).to(dist_util.dev())
        feature_extractor.requires_grad_(False)
    return feature_extractor

def load_discriminator_and_d_feature_extractor(args):
    #assert (args.gan_training == True) == (args.d_architecture == 'StyleGAN-XL')
    if args.gan_training:
        from pg_modules.projector import F_RandomProj
        from pg_modules.discriminator import MultiScaleD
        backbones = ['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0']
        discriminator, discriminator_feature_extractor = [], []
        if args.gan_low_res_train:
            discriminator2 = []
        backbone_kwargs = {'im_res': args.image_size}
        for i, bb_name in enumerate(backbones):
            feat = F_RandomProj(bb_name, **backbone_kwargs)
            disc = MultiScaleD(
                channels=feat.CHANNELS,
                resolutions=feat.RESOLUTIONS,
                **backbone_kwargs,
            )
            discriminator_feature_extractor.append([bb_name, feat])
            discriminator.append([bb_name, disc])
            if args.gan_low_res_train:
                discriminator2.append([bb_name + '_low', disc])
        discriminator_feature_extractor = nn.ModuleDict(discriminator_feature_extractor)
        discriminator_feature_extractor = discriminator_feature_extractor.train(False).to(dist_util.dev())
        discriminator_feature_extractor.requires_grad_(False)
        if args.gan_low_res_train:
            discriminator = discriminator + discriminator2
        discriminator = nn.ModuleDict(discriminator)
        discriminator.to(dist_util.dev())
        discriminator.train()
        if args.use_d_fp16:
            discriminator.convert_to_fp16()
    else:
        discriminator, discriminator_feature_extractor = None, None
    return discriminator, discriminator_feature_extractor

def gaussian_blur(args, img, step):
    blur_sigma = max(1 - step / (args.blur_fade_itr),
                     0) * args.blur_init_sigma if args.blur_fade_itr > 1 else 0
    blur_size = np.floor(blur_sigma * 3) * 2 - 1
    if blur_size > 1:
        img = torchvision.transforms.GaussianBlur(blur_size)(img)
    return img

def get_feature(args, input, feat, brightness, saturation, contrast, translation_x, translation_y,
                               offset_x, offset_y, name, step):
    # augment input
    input_aug_ = input
    if args.data_augment and brightness.shape[0] > 0:
        input_aug_ = DiffAugment(input[:brightness.shape[0]], brightness, saturation, contrast, translation_x, translation_y,
                                   offset_x, offset_y, policy='color,translation,cutout')
        input_aug_ = torch.cat((input_aug_, input[brightness.shape[0]:]))
    # transform to [0,1]
    input_aug = input_aug_.add(1).div(2)
    # apply F-specific normalization
    input_n = Normalize(feat.normstats['mean'], feat.normstats['std'])(input_aug)
    # upsample if smaller, downsample if larger + VIT
    if input.shape[-2] < 256:
        input_n = F.interpolate(input_n, 224, mode='bilinear', align_corners=False)
        if args.save_png and step % args.save_period in [0, 1] and step >= 0:
            input_aug_ = F.interpolate(input_aug_, 224, mode='bilinear', align_corners=False)
            script_util.save(input_aug_, logger.get_dir(), f'{name}_{step}_augment')
    # forward pass
    input_features = feat(input_n)
    return input_features

def get_xl_feature(args, estimate, target=None, feature_extractor=None, discriminator=None, step=-1, **model_kwargs):
    logits_fake, logits_real = [], []
    estimate_features, target_features = [], []
    for bb_name, feat in feature_extractor.items():
        # apply augmentation (x in [-1, 1])
        brightness = (torch.rand(int(estimate.size(0) * args.prob_aug), 1, 1, 1, dtype=estimate.dtype,
                              device=estimate.device) - 0.5)
        # brightness = 0.
        saturation = (torch.rand(int(estimate.size(0) * args.prob_aug), 1, 1, 1, dtype=estimate.dtype,
                              device=estimate.device) * 2)
        # saturation = 0.
        contrast = (torch.rand(int(estimate.size(0) * args.prob_aug), 1, 1, 1, dtype=estimate.dtype,
                            device=estimate.device) + 0.5)
        # contrast = 0.
        shift_x, shift_y = int(estimate.size(2) * args.shift_ratio + 0.5), int(
            estimate.size(3) * args.shift_ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[int(estimate.size(0) * args.prob_aug), 1, 1],
                                   device=estimate.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[int(estimate.size(0) * args.prob_aug), 1, 1],
                                   device=estimate.device)
        cutout_size = int(estimate.size(2) * args.cutout_ratio + 0.5), int(
            estimate.size(3) * args.cutout_ratio + 0.5)
        offset_x = torch.randint(0, estimate.size(2) + (1 - cutout_size[0] % 2),
                              size=[int(estimate.size(0) * args.prob_aug), 1, 1], device=estimate.device)
        offset_y = torch.randint(0, estimate.size(3) + (1 - cutout_size[1] % 2),
                              size=[int(estimate.size(0) * args.prob_aug), 1, 1], device=estimate.device)

        estimate_feature = get_feature(args, estimate, feat, brightness, saturation, contrast,
                                            translation_x, translation_y, offset_x, offset_y, 'estimate',
                                            step)
        estimate_features.append(estimate_feature)
        if discriminator is not None:
            logits_fake += discriminator.module[bb_name](estimate_feature, model_kwargs)
            if args.gan_low_res_train:
                estimate_low_res = F.interpolate(estimate, 16, mode='bilinear', align_corners=False)
                print("estimate, estimate_low_res: ", estimate.shape, estimate_low_res.shape)
                estimate_feature_low_res = get_feature(args, estimate_low_res, feat, brightness, saturation, contrast,
                                               translation_x, translation_y, offset_x, offset_y, 'estimate_low_res',
                                               step)
                logits_fake += discriminator.module[bb_name + '_low'](estimate_feature_low_res, model_kwargs)

        if target != None:
            if args.gan_real_free and args.gan_different_augment:
                # apply augmentation (x in [-1, 1])
                brightness = (torch.rand(int(target.size(0) * args.prob_aug), 1, 1, 1, dtype=target.dtype,
                                         device=target.device) - 0.5)
                # brightness = 0.
                saturation = (torch.rand(int(target.size(0) * args.prob_aug), 1, 1, 1, dtype=target.dtype,
                                         device=target.device) * 2)
                # saturation = 0.
                contrast = (torch.rand(int(target.size(0) * args.prob_aug), 1, 1, 1, dtype=target.dtype,
                                       device=target.device) + 0.5)
                # contrast = 0.
                shift_x, shift_y = int(target.size(2) * args.shift_ratio + 0.5), int(
                    target.size(3) * args.shift_ratio + 0.5)
                translation_x = torch.randint(-shift_x, shift_x + 1, size=[int(target.size(0) * args.prob_aug), 1, 1],
                                              device=target.device)
                translation_y = torch.randint(-shift_y, shift_y + 1, size=[int(target.size(0) * args.prob_aug), 1, 1],
                                              device=target.device)
                cutout_size = int(target.size(2) * args.cutout_ratio + 0.5), int(
                    target.size(3) * args.cutout_ratio + 0.5)
                offset_x = torch.randint(0, target.size(2) + (1 - cutout_size[0] % 2),
                                         size=[int(target.size(0) * args.prob_aug), 1, 1], device=target.device)
                offset_y = torch.randint(0, target.size(3) + (1 - cutout_size[1] % 2),
                                         size=[int(target.size(0) * args.prob_aug), 1, 1], device=target.device)
            with torch.no_grad():
                target_feature = get_feature(args, target, feat, brightness, saturation, contrast,
                                                  translation_x, translation_y, offset_x, offset_y, 'target',
                                                  step)
                target_features.append(target_feature)
            if discriminator is not None:
                logits_real += discriminator.module[bb_name](target_feature, model_kwargs)
                if args.gan_low_res_train:
                    target_low_res = F.interpolate(target, 16, mode='bilinear', align_corners=False)
                    target_feature_low_res = get_feature(args, target_low_res, feat, brightness, saturation, contrast,
                                                           translation_x, translation_y, offset_x, offset_y,
                                                           'target_low_res', step)
                    logits_real += discriminator.module[bb_name + '_low'](target_feature_low_res, model_kwargs)
    if discriminator is not None:
        if target == None:
            return logits_fake
        else:
            return logits_fake, logits_real
    else:
        if target == None:
            return estimate_features
        else:
            return estimate_features, target_features

def get_classifier_guidance(classifier, vpsde, unnormalized_input, std_wve_t, img_resolution, class_labels, log_prob=False):
    mean_vp_tau, std_vp_tau, tau = vpsde.transform_unnormalized_wve_to_normalized_vp(std_wve_t, std_out=True) ## VP pretrained classifier
    input = mean_vp_tau[:,None,None,None] * unnormalized_input
    with torch.enable_grad():
        x_ = input.float().clone().detach().requires_grad_()
        tau = torch.ones(input.shape[0], device=tau.device) * tau.reshape(-1)
        logits = classifier(x_, timesteps=tau)#, condition=class_labels)
        #print("tau: ", tau)
        #print("log probability: ", torch.softmax(logits, 1)[:,281].log())
        #print("log probability mean: ", torch.softmax(logits, 1)[:,281].log().mean())
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), class_labels['y'].view(-1)]
        if log_prob:
            return selected
        classifier_guidance = torch.autograd.grad(outputs=selected.sum(), inputs=x_, retain_graph=False)[0]
        classifier_guidance *= mean_vp_tau[:,None,None,None]
    return classifier_guidance

def get_log_ratio(discriminator, input, time, class_labels):
    if discriminator == None:
        return torch.zeros(input.shape[0], device=input.device)
    else:
        logits = discriminator(input, timesteps=time, condition=class_labels)
        prediction = torch.clip(logits, 1e-5, 1. - 1e-5)
        log_ratio = torch.log(prediction / (1. - prediction))
        return log_ratio

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
                tau = -self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * torch.log(1. + std_wve_t ** 2))
                tau /= self.beta_1 - self.beta_0
            elif self.multiplier == 2.:
                d = - torch.log(1. + std_wve_t ** 2)
                in_ = (2 * (self.b ** 3) - 9 * self.a * self.b * self.c + 27. * (self.a ** 2) * d) ** 2 - 4 * (((self.b ** 2) - 3 * self.a * self.c) ** 3)
                out_ = 2 * (self.b ** 3) - 9 * self.a * self.b * self.c + 27. * (self.a ** 2) * d
                plus = (out_ + in_ ** 0.5)
                minus = (out_ - in_ ** 0.5)
                sign_plus = torch.sign(plus)
                sign_minus = torch.sign(minus)
                tau = - self.b / (3. * self.a) - sign_plus * ((torch.abs(plus) / 2.) ** (1/3.)) / (3. * self.a) - sign_minus * ((torch.abs(minus) / 2.) ** (1/3.)) / (3. * self.a)
        elif multiplier == 1.:
            tau = -self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * torch.log(1. + std_wve_t ** 2))
            tau /= self.beta_1 - self.beta_0
        elif multiplier == 2.:
            d = - torch.log(1. + std_wve_t ** 2)
            in_ = (2 * (self.b ** 3) - 9 * self.a * self.b * self.c + 27. * (self.a ** 2) * d) ** 2 - 4 * (
                        ((self.b ** 2) - 3 * self.a * self.c) ** 3)
            out_ = 2 * (self.b ** 3) - 9 * self.a * self.b * self.c + 27. * (self.a ** 2) * d
            plus = (out_ + in_ ** 0.5)
            minus = (out_ - in_ ** 0.5)
            sign_plus = torch.sign(plus)
            sign_minus = torch.sign(minus)
            tau = - self.b / (3. * self.a) - sign_plus * ((torch.abs(plus) / 2.) ** (1 / 3.)) / (
                        3. * self.a) - sign_minus * ((torch.abs(minus) / 2.) ** (1 / 3.)) / (3. * self.a)
        return tau

    def marginal_prob(self, t, multiplier=-1.):
        log_mean_coeff = - 0.5 * self.integral_beta(t, multiplier)
        #log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def transform_normalized_vp_to_unnormalized_wve(self, t, multiplier=-1.):
        mean, std = self.marginal_prob(t, multiplier=multiplier)
        return std / mean

    def sampling_std(self, num_step):
        #c = 1000 // num_step
        assert 1000 % num_step == 0
        ddim_timesteps = torch.from_numpy(np.array(list(range(0, 1000, 1000 // num_step)))[::-1].copy())
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
        return torch.exp(rescaled_t / 250.) - 1e-44

    def compute_t_cos_from_t_lin(self, t_lin):
        sqrt_alpha_t_bar = torch.exp(-0.25 * t_lin ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t_lin * self.beta_0)
        time = torch.arccos(np.sqrt(self.f_0) * sqrt_alpha_t_bar)
        t_cos = self.T * ((1. + self.s) * 2. / np.pi * time - self.s)
        return t_cos

    def get_diffusion_time(self, batch_size, batch_device, t_min=1e-5, importance_sampling=True):
        if importance_sampling:
            Z = self.normalizing_constant(t_min)
            u = torch.rand(batch_size, device=batch_device)
            return (-self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2 * (self.beta_1 - self.beta_0) *
                    torch.log(1. + torch.exp(Z * u + self.antiderivative(t_min))))) / (self.beta_1 - self.beta_0), Z.detach()
        else:
            return torch.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

    def antiderivative(self, t, stabilizing_constant=0.):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor(t).float()
        return torch.log(1. - torch.exp(- self.integral_beta(t)) + stabilizing_constant) + self.integral_beta(t)

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