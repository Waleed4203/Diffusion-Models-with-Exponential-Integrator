# DEIS AO-DEIS Environment Setup — GPU (CUDA 12.x) Edition
# Run each block in order in an Anaconda Prompt after Miniconda installs.

# ── STEP 1: Create conda env ──────────────────────────────────────────────
conda create -n deis39 python=3.10 -y

# ── STEP 2: Install JAX with CUDA 12 GPU support ──────────────────────────
# RTX 5060 + CUDA 12.8 → use jax[cuda12] pip wheels
conda activate deis39
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ── STEP 3: Install PyTorch with CUDA 12.4 ────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ── STEP 4: Install TF + rest of requirements ─────────────────────────────
# (We skip tensorflow-intel / tensorflow-estimator on non-Intel; use plain tf)
pip install tensorflow==2.13.0 tensorflow-gan==2.0.0 tensorflow-datasets==4.8.3 tensorflow-hub==0.16.1 tensorflow-probability==0.21.0 tensorboard==2.13.0 keras==2.13.1

# ── STEP 5: Compatibility pins ────────────────────────────────────────────
pip install protobuf==3.20.3 setuptools==69.5.1 typing-extensions==4.5.0

# ── STEP 6: Python ecosystem ─────────────────────────────────────────────
pip install flax==0.8.5 optax==0.2.4 chex==0.1.90 orbax-checkpoint==0.6.4 ml-collections==0.1.1 absl-py==2.3.1 numpy==1.24.3 scipy==1.13.1 opt-einsum==3.4.0 six==1.17.0 wrapt==2.1.2 msgpack==1.1.2

# ── STEP 7: Verify GPU ────────────────────────────────────────────────────
python -c "import jax; print('JAX devices:', jax.devices())"
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
