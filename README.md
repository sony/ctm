# [ICLR'24] Consistency Trajectory Model (CTM)
<p align="center">
<img src="/assets/icon.png" alt="ctm" width="60%"/>
</p>
This repository houses the official PyTorch implementation of the paper titled "Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion" on ImageNet 64x64, which is presented at ICLR 2024.

* [arXiv](https://arxiv.org/abs/2310.02279)
* [Project Page](https://consistencytrajectorymodel.github.io/CTM/) 
* [OpenReview](https://openreview.net/forum?id=ymjI8feDTD)
* [Codes of other datasets (to be released...)](https://github.com/Kim-Dongjun)

Contacts:
* Dongjun KIM: <a href="dongjun@stanford.edu">dongjun@stanford.edu</a>
* Chieh-Hsin (Jesse) LAI: <a href="chieh-hsin.lai@sony.com">chieh-hsin.lai@sony.com</a>

## TL;DR
For single-step diffusion model sampling, our new model, Consistency Trajectory Model (CTM), achieves SOTA on CIFAR-10 (FID 1.73) and ImageNet 64x64 (FID 1.92). CTM offers diverse sampling options and balances computational budget with sample fidelity effectively.

## Checkpoints
- Download and put the [checkpoints](https://drive.google.com/drive/folders/1KPF3tWLRad3n18XJ1TD7J04XtoMIQ8QV?usp=sharing) in the file of author_ckpt
- [CTM checkpoint](https://drive.google.com/file/d/17XHwI5-IDpATRnBsxjOi6YCg1oD3MGC6/view?usp=sharing) on ImageNet64 (ema=0.999) 



## Prereqruisites

1. Download (or obtain) the following files

    - Pretrained diffusion model: Please locate it in `args.teacher_model_path`
    - Data: Please locate it in `args.data_dir`
    (Note that the data we use is NOT the downsampled image data. It is ILSVRC2012 data. There are huge performance gap between those two datasets.)
    - Reference statistics: statistics for computing FID, sFID, IS, precision, recall. Please locate them in `args.ref_path`

2. Install docker to your own server

    2-1.  Type `docker pull dongjun57/ctm-docker:latest` to download docker image in docker hub.
    
    2-2.  Create a container by typing in the command: 
        ```
        docker run --gpus=all -itd -v /etc/localtime:/etc/localtime:ro -v /dev/shm:/dev/shm -v [specified directory]:[specified directory] -v /hdd/imagenet/imagenet_dir/train:/hdd/imagenet/imagenet_dir/train -v [specified data directory]:[specified data directory] --name ctm-docker 8caa2682d007
        ```
        The commands could vary by your server environment.

    2-3. Go to the container by `docker exec -it ctm-docker bash`.
    
    2-4. Go to the virtual environment by `conda activate ctm`.

3. Make sure the dependencies consistent with the following.


    ```
    apt install git
    apt install libopenmpi-dev
    python -m pip install tensorflow[and-cuda]
    python -m pip install torch torchvision torchaudio
    python -m pip install blobfile tqdm numpy scipy pandas Cython piq==0.7.0
    python -m pip install joblib==0.14.0 albumentations==0.4.3 lmdb clip@git+https://github.com/openai/CLIP.git pillow
    python -m pip install flash-attn --no-build-isolation
    python -m pip install xformers
    python -m pip install mpi4py
    python -m pip install nvidia-ml-py3 timm==0.4.12 legacy dill nvidia-ml-py3
    ```
    
## Training
- For CTM+DSM training, run `bash commands/CTM+DSM_command.sh` 

    Recommendation: at least run CTM+DSM for 10~50k iterations

- For CTM+DSM+GAN training, run `bash commands/CTM+DSM+GAN_command.sh`
 
    Recommendation: at least run CTM+DSM+GAN for >=30k iterations



## Sampling

Please see `commands/sampling_commands.sh` for detailed sampling commands.


## Evaluating

Run `python3.8 evaluations/evaluator.py [location_of_statistics] [location_of_samples]` 

The first argument is the reference path and the second argument is the folder of your samples (>=50k samples for correct evaluation).


Please refer to the statistics of [ADM (Prafulla Dhariwal, Alex Nichol)](https://github.com/openai/guided-diffusion).


## Customized dataset
Users need to manually replace the data_name with your data name: manually modify the data_name in `cm_train.py` or `image_sample.py`



## Citations


```
@article{kim2023consistency,
  title={Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion},
  author={Kim, Dongjun and Lai, Chieh-Hsin and Liao, Wei-Hsiang and Murata, Naoki and Takida, Yuhta and Uesaka, Toshimitsu and He, Yutong and Mitsufuji, Yuki and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.02279},
  year={2023}
```
