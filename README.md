# Gradient-Aware Loss Function for Improved Learning in Speech Enhancement

This repository provides the official implementation for the paper\
**"Gradient-Aware Loss Function for Improved Learning in Speech
Enhancement"**.\
It includes training scripts, evaluation code, and configuration files
used in the experiments.\
[\[Paper\]](https://arxiv.org/) *(link to be updated)*

------------------------------------------------------------------------

## 📦 Requirements

    librosa==0.11.0
    matplotlib==3.9.4
    numpy==2.0.2
    pesq==0.0.4
    soundfile==0.13.1
    torch==2.7.1+cu128
    torchaudio==2.7.1+cu128
    torchmetrics==1.7.2

------------------------------------------------------------------------

## 💪️ Training (train.py)

1.  Configure model and training parameters in **config.py**.

2.  If you want to resume training from a specific epoch,\
    set `load_checkpoint_name` to:

        load_checkpoint_name = <save_checkpoint_name>_<epoch>
    Example:\
    `Proposed_TD_FT-MSE_lr-3_100`

3.  To start training from scratch, leave `load_checkpoint_name` empty.

------------------------------------------------------------------------

## 🧠 Inference (test.py)

1.  Configure model and evaluation parameters in **config.py**.

2.  To evaluate a specific checkpoint, set `load_checkpoint_name` to:

        load_checkpoint_name = <save_checkpoint_name>_<epoch>

    Example:\
    `Proposed_TD_FT-MSE_lr-3_456`

3.  **Inference requires `load_checkpoint_name` to be set.**

------------------------------------------------------------------------

## 📄 Citation

If you find this work helpful, please consider citing:

    (To be added once paper is available)
