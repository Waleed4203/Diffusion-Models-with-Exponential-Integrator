# DEIS Project - Quick Start Guide

## What is DEIS?

DEIS (Diffusion Exponential Integrator Sampler) is a fast sampling method for diffusion models. This project implements score-based generative models with the DEIS sampler for efficient image generation.

## Quick Setup (5 Minutes)

### 1. Create Environment
```powershell
conda create -n deis39 python=3.9 -y
conda activate deis39
```

### 2. Install Dependencies
```powershell
# Navigate to project directory
cd path\to\deis-main\deis-main

# Install from requirements file
pip install -r requirements_waleed.txt

# Install JAX from conda (better Windows support)
conda install -n deis39 -c pkgs/main jaxlib=0.4.23 jax -y
```

### 3. Verify Installation
```powershell
python -c "import jax, flax, tensorflow; print('All imports successful!')"
```

---

## Running the Project

### Option 1: Train a New Model

Train a score-based diffusion model on CIFAR-10:

```powershell
python demo/continuous_cifar/main.py `
    --config demo/continuous_cifar/configs/vp/cifar10_ddpmpp_continuous.py `
    --workdir ./my_training `
    --mode train
```

**What this does:**
- Downloads CIFAR-10 dataset automatically
- Trains a diffusion model from scratch
- Saves checkpoints every 50,000 steps
- Logs training progress every 50 steps
- Takes ~7-10 days on CPU (or 3-4 hours on GPU)

**Output files:**
- `./my_training/checkpoints/` - Model checkpoints
- `./my_training/samples/` - Generated image samples
- `./my_training/stdout.txt` - Training logs
- `./my_training/tensorboard/` - TensorBoard logs

### Option 2: Evaluate a Trained Model

Evaluate a trained checkpoint:

```powershell
python demo/continuous_cifar/main.py `
    --config demo/continuous_cifar/configs/vp/cifar10_ddpmpp_continuous.py `
    --workdir ./my_training `
    --mode eval `
    --eval_folder evaluation_results
```

**What this does:**
- Loads trained model from checkpoints
- Generates sample images
- Computes quality metrics (FID, IS, KID)
- Saves results to `./my_training/evaluation_results/`

### Option 3: Test DEIS Sampler

Run unit tests to verify the DEIS implementation:

```powershell
# Test the tabulated DEIS solver
pytest tests/test_tab_deis.py -v

# Test reverse-time integration
pytest tests/test_rev_ts.py -v

# Run all tests
pytest tests/ -v
```

### Option 4: Interactive Sampling (Jupyter Notebook)

```powershell
# Install Jupyter
pip install jupyter

# Start notebook server
jupyter notebook demo/continuous_cifar/Score_SDE_demo.ipynb
```

---

## Configuration Options

### Adjusting Training Settings

Edit `demo/continuous_cifar/configs/default_cifar10_configs.py`:

```python
# Batch size (reduce if out of memory)
config.training.batch_size = 16  # default: 16, original: 128

# Number of training iterations
training.n_iters = 1300001  # ~1.3M steps

# Logging frequency
training.log_freq = 50  # log every 50 steps

# Checkpoint frequency
training.snapshot_freq = 50000  # save every 50k steps

# JIT compilation steps (reduce if out of memory)
training.n_jitted_steps = 1  # default: 1, original: 5
```

### Adjusting Model Size

Edit `demo/continuous_cifar/configs/vp/cifar10_ddpmpp_continuous.py`:

```python
# Model capacity
model.nf = 64  # base channels (default: 64, original: 128)

# Model depth
model.ch_mult = (1, 2, 2)  # channel multipliers (default: (1,2,2), original: (1,2,2,2))
model.num_res_blocks = 2  # residual blocks per level (default: 2, original: 4)
```

**Memory vs Quality Trade-off:**
| Config | Memory | Quality | Training Time |
|--------|--------|---------|---------------|
| Small (nf=32, 1 block) | ~500MB | Low | Fast |
| Medium (nf=64, 2 blocks) | ~2GB | Good | Medium |
| Large (nf=128, 4 blocks) | ~13GB | Best | Slow |

---

## Common Use Cases

### Use Case 1: Quick Test Run

Test if everything works (5 minutes):

```powershell
# Modify config for quick test
# In default_cifar10_configs.py:
# - batch_size = 4
# - n_iters = 100
# - log_freq = 10

python demo/continuous_cifar/main.py `
    --config demo/continuous_cifar/configs/vp/cifar10_ddpmpp_continuous.py `
    --workdir ./test_run `
    --mode train
```

### Use Case 2: Resume Training

Resume from a checkpoint:

```powershell
# Just run the same command - it auto-resumes
python demo/continuous_cifar/main.py `
    --config demo/continuous_cifar/configs/vp/cifar10_ddpmpp_continuous.py `
    --workdir ./my_training `
    --mode train
```

The code automatically detects existing checkpoints in `./my_training/checkpoints-meta/` and resumes from the latest one.

### Use Case 3: Generate Samples Only

Generate samples without full evaluation:

```python
# In config, set:
config.eval.enable_sampling = True
config.eval.enable_loss = False
config.eval.enable_bpd = False
config.eval.num_samples = 1000  # number of samples to generate
```

Then run:
```powershell
python demo/continuous_cifar/main.py `
    --config demo/continuous_cifar/configs/vp/cifar10_ddpmpp_continuous.py `
    --workdir ./my_training `
    --mode eval
```

### Use Case 4: Different Datasets

The code supports multiple datasets. To use a different one:

```python
# In config:
config.data.dataset = 'CELEBA'  # or 'LSUN', 'ImageNet', etc.
config.data.image_size = 64  # adjust as needed
```

---

## Monitoring Training

### Method 1: Check Logs

```powershell
# View training logs in real-time
Get-Content ./my_training/stdout.txt -Wait

# Or on Linux/Mac:
tail -f ./my_training/stdout.txt
```

### Method 2: TensorBoard

```powershell
# Install tensorboard (if not already installed)
pip install tensorboard

# Start TensorBoard
tensorboard --logdir ./my_training/tensorboard

# Open browser to http://localhost:6006
```

### Method 3: Check Samples

Generated samples are saved as PNG images in `./my_training/samples/iter_XXXXX_host_0/sample.png`

---

## Performance Optimization

### For CPU Training

1. **Reduce batch size:** `batch_size = 8` or `4`
2. **Reduce model size:** `nf = 32`, `num_res_blocks = 1`
3. **Disable sampling:** `training.snapshot_sampling = False`
4. **Use single jitted step:** `n_jitted_steps = 1`

### For GPU Training (Python 3.10+ required)

```powershell
# Create Python 3.10 environment
conda create -n deis310 python=3.10 -y
conda activate deis310

# Install CUDA-enabled JAX
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install -r requirements_waleed.txt

# Verify GPU is detected
python -c "import jax; print(jax.devices())"
# Should show: [cuda(id=0)]
```

Then you can use larger configs:
```python
config.training.batch_size = 128
training.n_jitted_steps = 5
model.nf = 128
model.num_res_blocks = 4
```

---

## Troubleshooting

### Problem: Out of Memory

**Error:**
```
RESOURCE_EXHAUSTED: Out of memory allocating X bytes
```

**Solutions:**
1. Reduce `batch_size` to 8 or 4
2. Reduce `n_jitted_steps` to 1
3. Reduce model size (`nf=32`)
4. Close other applications
5. Use GPU if available

### Problem: Slow Training

**Symptom:** Each step takes 5-10 seconds

**Solutions:**
1. First run is always slow (JIT compilation)
2. Subsequent runs are faster (cached)
3. Use GPU for 50-100x speedup
4. Reduce model size for faster iterations

### Problem: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'X'
```

**Solution:**
```powershell
# Reinstall all dependencies
pip install -r requirements_waleed.txt --force-reinstall
```

### Problem: CUDA Not Available

**Error:**
```
Unable to initialize backend 'cuda'
```

**Solutions:**
1. This is normal if you don't have an NVIDIA GPU
2. Training will use CPU automatically
3. For GPU support, see "For GPU Training" section above

---

## Project Structure

```
deis-main/
├── demo/
│   └── continuous_cifar/          # Main training code
│       ├── main.py                # Entry point
│       ├── run_lib.py             # Training/eval logic
│       ├── losses.py              # Loss functions (MODIFIED)
│       ├── models/
│       │   └── utils.py           # Model utilities (MODIFIED)
│       ├── configs/
│       │   ├── default_cifar10_configs.py  # Default config (MODIFIED)
│       │   └── vp/
│       │       └── cifar10_ddpmpp_continuous.py  # VP-SDE config (MODIFIED)
│       └── ...
├── jax_deis/                      # JAX DEIS implementation
│   ├── sampler.py                 # Main sampler
│   ├── vpsde.py                   # VP-SDE (MODIFIED for compatibility)
│   └── ...
├── th_deis/                       # PyTorch DEIS implementation
│   ├── sampler.py
│   ├── vpsde.py                   # (MODIFIED for compatibility)
│   └── ...
├── tests/                         # Unit tests
│   ├── test_tab_deis.py
│   └── test_rev_ts.py
├── requirements_waleed.txt        # Dependencies (NEW)
├── SETUP_GUIDE_WALEED.md         # Detailed setup guide (NEW)
├── WALEED_README.md              # This file (NEW)
└── README.md                      # Original README
```

---

## FAQ

### Q: How long does training take?

**A:** On CPU: ~7-10 days for full training (1.3M iterations). On GPU: ~3-4 hours.

### Q: Can I stop and resume training?

**A:** Yes! Just run the same training command. It automatically resumes from the latest checkpoint.

### Q: How much disk space do I need?

**A:** ~5GB for dataset + checkpoints + samples.

### Q: Can I use my own dataset?

**A:** Yes, but you'll need to modify the data loading code in `datasets.py`.

### Q: What GPU do I need?

**A:** Any NVIDIA GPU with 8GB+ VRAM (RTX 3060, 3070, 4060, etc.). CPU training also works but is slower.

### Q: How do I know if training is working?

**A:** Check that `training_loss` decreases over time. It should start around 1.0 and decrease to ~0.01-0.001.

### Q: Can I use this for other image sizes?

**A:** Yes, modify `config.data.image_size`. Note: larger images need more memory and training time.

---

## Getting Help

1. **Check logs:** `./my_training/stdout.txt`
2. **Read setup guide:** `SETUP_GUIDE_WALEED.md`
3. **Check original README:** `README.md`
4. **Run tests:** `pytest tests/ -v`

---

## Citation

If you use this code, please cite the original DEIS paper:

```bibtex
@article{zhang2022deis,
  title={Fast Sampling of Diffusion Models with Exponential Integrator},
  author={Zhang, Qinsheng and Chen, Yongxin},
  journal={arXiv preprint arXiv:2204.13902},
  year={2022}
}
```

---

## License

This project follows the original DEIS license. See `LICENSE` file for details.




and

& "$env:USERPROFILE\miniconda3\envs\deis39\Scripts\pip.exe" install "jax[cuda12_pip]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 2>&1 | Select-Object -Last 10