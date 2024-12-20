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
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
