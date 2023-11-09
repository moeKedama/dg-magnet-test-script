## Prerequisites

- Windows11/Ubuntu 22.04 LTS
- Python 3.10.11
- Pytorch 2.0.1
- NVIDIA GPU
- Anaconda 
- CUDA 11.7 /CUDA11.8 /CUDA12.1
- Recent GPU driver 

## Preparation

```
conda create -n torch2 python=3.10
conda activate torch2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install scikit-learn
pip install matplotlib 
pip install pandas 
pip install seaborn
```

## Test only

- Download the [dg-magnet-test-script-model](https://drive.google.com/file/d/1wPTIV1UM8M_bGaygE-NBeeGJ71x6DaA2/view?usp=sharing), put the model into `pth_file` directory. 

```
├── pth_file/10domain
│   ├── ViT_DANN_model_for_valid_7601.pth
├── pth_file/1domain
│   ├── 3C90
│   ├── ├── ViT_3C90_4000.pth
│   ├── 3C94
│   ├── 3E6
│   ├── 3F4
│   ├── 77
│   ├── 78
│   ├── N27
│   ├── N30
│   ├── N49
│   ├── N87
├── pth_file/subdomain
│   ├── 3
│   ├── ├── ViT_sub_3_11401.pth
│   ├── 7
│   ├── N
```

- Download the [MagNet Challenge Validation Data](https://www.dropbox.com/sh/4ppuzu7z4ky3m6l/AAApqXcxr_Fnr5x9f5qDr8j8a?dl=0), put test datasets into `valid_data` directory.

```
├── valid_data
│   ├── 3C90
│   ├── ├── B_waveform.csv
│   ├── ├── Frequency.csv
│   ├── ├── H_Waveform.csv
│   ├── ├── Temperature.csv
│   ├── ├── Volumetric_Loss.csv
│   ├── 3C94
│   ├── 3E6
│   ├── 3F4
│   ├── 77
│   ├── 78
│   ├── N27
│   ├── N30
│   ├── N49
│   ├── N87
```

- Modify the function `get_args()` .
  - modify the `--dataset` in the `get_args() ` (see the commented code).
- Modify the `norm_dict` and `pth_PATH` in `main` .
  - `norm_dict` load the normalization coefficients as a *dictionary* , the saved coefficients are in `var_file` directory.
  - `pth_PATH` is the path of saved model.
- Run , result will be saved at `pred_file/`.

```
├── pred_file/10domain
│   ├── pred_3C90.csv
│   ├── pred_3C94.csv
│   ├── pred_3E6.csv
│   ├── pred_3F4.csv
│   ├── pred_77.csv
│   ├── pred_78.csv
│   ├── pred_N27.csv
│   ├── pred_N30.csv
│   ├── pred_N49.csv
│   ├── pred_N87.csv
```

## Train

*Coming soon*.



## FULL VERSION PRETEST RESULTS

[Google Drive Link](https://drive.google.com/file/d/1k8Odom3b7HC1hGeDV3Rz9DBGP7Rxk2Zn/view?usp=sharing)

## Acknowledgments

- [minjiechen/magnetchallenge](https://github.com/minjiechen/magnetchallenge)
- [jindongwang/transferlearning](https://github.com/jindongwang/transferlearning)

