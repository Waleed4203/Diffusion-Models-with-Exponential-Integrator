# AO-DEIS: Adaptive-Order Discrete Diffusion Sampler (PyTorch)

This directory contains the PyTorch implementation of the **Adaptive-Order Diffusion Exponential Integrator Sampler (AO-DEIS)**, applied to discrete diffusion models (DDIM/DDPM). 

While standard fast-samplers rely on fixed-order polynomial extrapolation, they often diverge in ultra-low step regimes (10-20 NFE) or under high Classifier-Free Guidance. AO-DEIS introduces a mathematically rigorous, dynamic order-switching mechanism that guarantees stability with **zero extra neural network evaluations**.

## Features & Contributions
* **Dynamic Order Selection:** Calculates an embedded error at each step using historical $\epsilon$-buffers, dynamically falling back to lower-order solvers in stiff ODE regions to prevent truncation spikes.
* **Proper Bootstrapping:** A strict `max_safe_order` cap ensures the solver never uses dummy noise vectors during the critical early timesteps.
* **Native Evaluation Pipeline:** Bypasses complex TF-GAN dependencies. Uses a streamlined `deis_celeba.ipynb` notebook to visually compare NFE convergence and calculate `pytorch-fid` directly against the raw CIFAR-10 dataset.

## Installation & Setup

1. **Environment Setup:** Ensure you have the `deis39` conda environment active with PyTorch installed.
2. **Install Evaluation Metrics:**
   ```bash
   pip install pytorch-fid
   pip install ipykernel
   ```

## Usage

### 1. Extract Real CIFAR-10 Images (For FID Calculation)
To compute an accurate FID score without relying on pre-computed `.npz` statistics, we extract the raw CIFAR-10 dataset to a local folder:
```bash
python extract_cifar.py
```
This will download and save 50,000 real CIFAR-10 images to `temp/cifar10_real`.

### 2. Generate Samples via Terminal
To sample using the trained checkpoint (`temp/train/ema.ckpt`):
```bash
python main.py --runner sample --config ddim_cifar10.yml --model_path temp/train/ema.ckpt --device cuda
```
*Note: Generated samples are saved to `temp/sample/`.*

### 3. End-to-End Jupyter Evaluation (Recommended)
The most comprehensive way to evaluate the model is via the custom evaluation notebook.
Open **`deis_celeba.ipynb`** in VS Code (ensure the `deis39` kernel is selected) and run all cells. The notebook will automatically:
1. Load the PyTorch EMA weights.
2. Wrap the U-Net in the `AODEISSampler`.
3. Generate side-by-side visual grids comparing convergence at **10 NFE** vs. **20 NFE**.
4. Natively execute `pytorch-fid` to compute the final metric.

## Current Benchmarks
* **Model:** Discrete DDIM/DDPM (PyTorch)
* **Dataset:** CIFAR-10
* **Training Steps:** 75,000 (Early Checkpoint)
* **Sampling Steps:** 10 NFE
* **FID Score:** 20.34

*Note: As training continues toward standard benchmarks (500k+ steps), the FID score will scale accordingly, maintaining stability at ultra-low step counts thanks to the AO-DEIS logic.*
