# IACC: Cross Illumination-Aware and Colour Correction for Underwater Images Under Mixed Natural and Artificial Lighting

## 1. Environmental requirements
```shell
python=3.9
pytorch=1.12.1
timm
ruamel.yaml
kornia
```

## 2. Train
- Download the UIEB dataset
- Divide the dataset into training set, validation set and test set
- edit **train_path** and **valid_path** in config.yaml
- activate your python environment
- python run.py


## 3. test
Test metrics from [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)
