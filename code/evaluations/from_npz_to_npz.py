import numpy as np
import glob
import os

sampler = 'ddpm'
#save_dir = '/home/aailab/dongjoun57/SeventhArticleExperimentalResults/ImageNet64/Diffusion/ADM/ddpm_samples/'
dir = '221003_original_npz'
#save_dir = f'/home/aailab/dongjoun57/SeventhArticleExperimentalResults/ImageNet64/Diffusion/ADM/{dir}'
save_dir = '/dataset/LSUN/church/fliped_data_npz/'
save_dir_ = '/dataset/LSUN/church/fliped_data_npz/'
save_dir = '/home/dongjun/EighthArticleExperimentalResults/Church/check_for_min_sigma/samples_0.0'
save_dir_ = '/home/dongjun/EighthArticleExperimentalResults/Church/check_for_min_sigma/samples_0.0_'
save_dir = '/home/dongjun/EighthArticleExperimentalResults/Church/exp/mixed_k_unet_step_18_heun_17_random_DSM_10.0_GAN_adaptive_0.1_unet_hinge_0.1_g_period_5_d_period_6_d_out_32_sigma_min_0.04_ema/consistency_training_topic2_candidate2_exact_sampler_4_steps_140000_itrs_0.9999_ema_'
save_dir_ = '/home/dongjun/EighthArticleExperimentalResults/Church/exp/mixed_k_unet_step_18_heun_17_random_DSM_10.0_GAN_adaptive_0.1_unet_hinge_0.1_g_period_5_d_period_6_d_out_32_sigma_min_0.04_ema/consistency_training_topic2_candidate2_exact_sampler_4_steps_140000_itrs_0.9999_ema__'
save_dir = '/home/dongjun/EighthArticleExperimentalResults/Church/CD/edm_heun_sampler_18_steps_/model_itrs_model_ema'
save_dir_ = '/home/dongjun/EighthArticleExperimentalResults/Church/CD/edm_heun_sampler_18_steps_/model_itrs_model_ema_'
save_dir = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/training_no_flip'
save_dir_ = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/training_no_flip'
save_dir = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/CTM/CTM_from_cd_random_M_3_lpips/ctm_exact_sampler_1_steps_070000_itrs_0.9999_ema_'
save_dir_ = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/CTM/CTM_from_cd_random_M_3_lpips/ctm_exact_sampler_1_steps_070000_itrs_0.9999_ema__'
save_dir = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/CTM/CTM_from_cd_random_M_3_lpips/edm_heun_sampler_40_steps_64_ema_itrs_model_ema'
save_dir_ = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/CTM/CTM_from_cd_random_M_3_lpips/edm_heun_sampler_40_steps_64_ema_itrs_model_ema_'
save_dir = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/CTM/CTM_from_cd_random_M_3_lpips/cd_onestep_sampler_1_steps__lpips_itrs_model_ema'
save_dir_ = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/CTM/CTM_from_cd_random_M_3_lpips/cd_onestep_sampler_1_steps__lpips_itrs_model_ema_'
save_dir = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/CD/CD_0.00008_from_cd/cd_onestep_sampler_1_steps_030000_itrs_0.9999_ema'
save_dir_ = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/CD/CD_0.00008_from_cd/cd_onestep_sampler_1_steps_030000_itrs_0.9999_ema_'
#save_dir = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/DM/EDM_samples/edm_heun_sampler_40_steps_64_ema_itrs_model_ema'
#save_dir_ = '/home/dongjun/EighthArticleExperimentalResults/ImageNet64/DM/EDM_samples/edm_heun_sampler_40_steps_64_ema_itrs_model_ema_'
#save_dir = '/hdd/dongjun/EighthArticleExperimentalResults/Church/pretrained/edm_heun_sampler_100_steps_/model_itrs_model_ema/'
#save_dir_ = '/hdd/dongjun/EighthArticleExperimentalResults/Church/pretrained/edm_heun_sampler_100_steps_/'
ext = 'npz'
#ext = 'npy'
#filenames = glob.glob(os.path.join(save_dir, '*.npy'))
filenames = glob.glob(os.path.join(save_dir, '*.'+ext))
print(save_dir)
print(filenames)
imgs = []
labels = []
for file in filenames:
    try:
        img = np.load(file)#['arr_0']
        if ext == 'npz':
            try:
                img = img['data']
            except:
                img = img['arr_0']
        #label = np.load(file)['arr_1']
        print(img.shape)
        imgs.append(img)
        #labels.append(label)
    except:
        pass
imgs = np.concatenate(imgs, axis=0)
#labels = np.concatenate(labels, axis=0)
imgs = imgs[:41487]
#imgs = imgs[:10000]
print(imgs.shape)#, labels.shape)
os.makedirs(save_dir_, exist_ok=True)
np.savez(os.path.join(save_dir_, f'data'), imgs)#, labels)