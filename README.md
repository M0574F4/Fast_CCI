# High-Throughput Blind Co-Channel Interference Cancellation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Paper](https://img.shields.io/badge/paper-arXiv%202424.12541-blue.svg)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)

## Introduction

This repository implements the methods presented in our paper:

**"High-Throughput Blind Co-Channel Interference Cancellation for Edge Devices Using Depthwise Separable Convolutions, Quantization, and Pruning"**

[![Paper](https://img.shields.io/badge/paper-arXiv%202424.12541-blue.svg)](https://arxiv.org/pdf/2411.12541)

Our work focuses on enhancing interference cancellation in edge devices by leveraging depthwise separable convolutions, quantization, and pruning techniques to achieve high throughput and efficiency.

## Features

- **Depthwise Separable Convolutions:** Reduces computational complexity without compromising performance.
- **Quantization:** Lowers model size and inference latency.
- **Pruning:** Eliminates redundant parameters for optimized performance.
- **Scalable Training:** Supports distributed training across multiple devices.
- **Custom DataLoader:** Tailored for the ICASSP 2024 SP Grand Challenge dataset with unknown interference types.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/M0574F4/Fast_CCI.git
   cd Fast_CCI
   
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt

## Usage
## Training
To start training the model, use the train.py script with the desired parameters:

   ```bash
  python train.py \
    dataloader.batch_size=<batch_size> \
    model.model_type=<model_type> \
    model.enc_block_type=<enc_block_type> \
    model.bottleneck_type=lstm \
    model.encoder_filters=<encoder_filters> \
    model.hidden_lstm_size=<hidden_lstm_size> \
    model.conv_dim=<conv_dim> \
    trainer.max_steps=<max_steps> \
    trainer.backward_option=<backward_option> \
    dataloader.sig_len=<sig_len> \
    distributed.n_devices=<n_devices> \
    distributed.strategy=<strategy>
```

## Dataset

We utilize the [ICASSP 2024 SP Grand Challenge: Data-Driven Signal Separation in Radio Spectrum](https://rfchallenge.mit.edu/icassp24-single-channel/) dataset. This dataset provides a challenging single-channel signal separation problem designed to push the boundaries of data-driven techniques in radio spectrum processing.

### Customization

Unlike the original challenge, our approach assumes **unknown interference types**, making the task more generalized and applicable to real-world scenarios. A custom `DataLoader` is included in this repository to handle this specific scenario, providing efficient data preprocessing and augmentation.

For more information about the dataset and its usage, visit the [ICASSP Challenge page](https://rfchallenge.mit.edu/icassp24-single-channel/).

## Results

Our model demonstrates significant performance improvements across multiple metrics and showcases scalability for edge-device deployments. Below are key results from our research:

### Key Findings

- **Figure 4:** Scatter plot of MSE score versus MACs for different models. The size of the circles is proportional to the number of parameters of each model.

  ![Figure 4](fig4.png)

- **Figure 7:** This graph illustrates the increasing inference throughput as the batch size is expanded, highlighting the scalability and efficiency gains achieved during batch processing.

  ![Figure 7](fig7.png)

### Summary

- **Efficiency Gains:** Depthwise separable convolutions significantly reduce computational costs while maintaining high accuracy.
- **Scalability:** The model demonstrates linear throughput scaling with batch size, making it highly suitable for real-time and batch inference scenarios.

_For a comprehensive analysis and additional results, please refer to the [paper](https://arxiv.org/pdf/2411.12541)._ 
