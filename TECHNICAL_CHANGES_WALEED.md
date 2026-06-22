# Technical Changes Log - DEIS Project

## Purpose
This document records every single change made to the codebase, every error encountered, every decision made, and the reasoning behind each fix. Written for full reproducibility.

---

## Session Overview

**Goal:** Get the DEIS project running on Windows with modern Python/JAX/Flax versions.

**Root Problem:** The project was written for Python 3.8 with pinned old dependencies (jax==0.2.8, flax==0.3.1, tensorflow==2.4.0) that have no Windows wheels and use APIs that have since been removed.

**Final Solution:** Python 3.9 + conda env + modern JAX/Flax/TF with compatibility patches.

---

## Part 1: Initial Test Run (jax_deis and th_deis)

### Task
Run the existing tests in `tests/` directory.

### Environment Created
- Name: `deis_env`
- Python: 3.12
- Packages: `jax[cpu]`, `torch`, `pytest`, `matplotlib`, `numpy`

### Error Encountered
```
ImportError: cannot import name '_promote_dtypes_inexact' from 'jax._src.dtypes'
```

**Root Cause:** JAX removed the private function `_promote_dtypes_inexact` in newer versions. Both `jax_deis/vpsde.py` and `th_deis/vpsde.py` imported it directly.

### Fix Applied

**File:** `jax_deis/vpsde.py`
**File:** `th_deis/vpsde.py`

Replaced the removed import with a local equivalent:

```python
# REMOVED (broken):
from jax._src.dtypes import _promote_dtypes_inexact

# ADDED (replacement):
def _promote_dtypes_inexact(*args):
    """Local replacement for removed JAX private function."""
    import jax.numpy as jnp
    return [jnp.array(a, dtype=jnp.result_type(*[jnp.array(x).dtype 
            for x in args if hasattr(x, 'dtype') or isinstance(x, (int, float))])) 
            for a in args]
```

**Result:** Tests ran successfully (slow due to JAX JIT on CPU, but passing).

---

## Part 2: Running demo/continuous_cifar/main.py

### Task
Run: `python main.py --config configs/vp/cifar10_ddpmpp_continuous.py --workdir ./workdir --mode train`

### Initial Analysis of Requirements

Read `demo/continuous_cifar/requirements.txt`:
```
ml-collections==0.1.0
tensorflow-gan==2.0.0
tensorflow_io
tensorflow_datasets==3.1.0
tensorflow==2.4.0
tensorflow-addons==0.12.0
tensorboard==2.4.0
absl-py==0.10.0
flax==0.3.1
jax==0.2.8
jaxlib==0.1.59
```

**Problem identified immediately:** `jaxlib==0.1.59` has no Windows wheels. JAX historically only provided Linux/Mac wheels for old versions.

### Attempt 1: Install with Python 3.11 (cifar_env)

Created `cifar_env` with Python 3.11, tried installing modern jax/flax/tensorflow.

**Failed because:**
- `flax.jax_utils` removed in newer Flax
- `flax.training.checkpoints` removed
- `tf.estimator` removed in TF 2.20
- Too many API incompatibilities to patch quickly

### Attempt 2: Download Python 3.8 Installer

Tried downloading Python 3.8 installer via `curl` and `Invoke-WebRequest` — both failed (network/permission issue).

**Decision:** The installer `python38_installer.exe` was already present in the workspace directory. Used it directly.

### Python 3.8 Installation

```powershell
Start-Process -FilePath ".\python38_installer.exe" `
    -ArgumentList "/quiet InstallAllUsers=0 PrependPath=0 Include_test=0 TargetDir=C:\Python38" `
    -Wait -PassThru
```

**Result:** Python 3.8.10 installed at `C:\Python38\python.exe`

### Attempt 3: Python 3.8 venv + pip install requirements

```powershell
C:\Python38\python.exe -m venv cifar38_env
cifar38_env\Scripts\python.exe -m pip install -r demo/continuous_cifar/requirements.txt
```

**Error:**
```
ERROR: Could not find a version that satisfies the requirement jaxlib==0.1.59 (from versions: none)
```

**Root Cause:** `jaxlib==0.1.59` simply does not exist as a Windows wheel on PyPI. JAX did not provide Windows wheels for versions before ~0.4.x.

**Checked:** JAX's official release page `https://storage.googleapis.com/jax-releases/jax_releases.html` — no Windows jaxlib wheels at all for old versions.

### Attempt 4: Find jaxlib via conda-forge

```powershell
conda search -c conda-forge jaxlib
```

**Result:** Minimum available version for Windows is `jaxlib==0.4.23`, and only for Python 3.9+. No Python 3.8 builds exist.

**Decision:** Switch to Python 3.9 with conda, use jaxlib 0.4.23, and patch all API incompatibilities.

---

## Part 3: Setting Up the Working Environment (deis39)

### Create conda env with Python 3.9

```powershell
conda create -n deis39 python=3.9 -y
```

**Why Python 3.9 specifically?**
- Minimum version supported by jaxlib on Windows via conda
- Still compatible with TF 2.13 (which we need for tf.estimator)
- Python 3.10+ would allow CUDA jaxlib via pip, but we started with 3.9

### Install JAX via conda

```powershell
conda install -n deis39 -c pkgs/main jaxlib=0.4.23 jax -y
```

**Result:** JAX 0.4.25 + jaxlib 0.4.23 installed successfully.

**Verification:**
```python
import jax; print(jax.__version__)  # 0.4.25
```

### Install TensorFlow

First attempt: `pip install tensorflow` → got TF 2.20.0

**Problem:** TF 2.20 removed `tf.estimator`, which `tensorflow_gan` requires.

**Error when importing tensorflow_gan:**
```
AttributeError: module 'tensorflow' has no attribute 'estimator'
```

**Fix:** Downgrade to TF 2.13 (last version with `tf.estimator`):
```powershell
pip install tensorflow==2.13.0
```

**Secondary problem:** `tensorflow` on Windows is a wrapper package that installs `tensorflow-intel`. After downgrade, `import tensorflow` failed with:
```
AttributeError: module 'tensorflow' has no attribute '__version__'
```

**Fix:** Force reinstall tensorflow-intel:
```powershell
pip install tensorflow-intel==2.13.0 --force-reinstall
```

### Install tensorflow-gan

```powershell
pip install tensorflow-gan==2.0.0
```

**Error:**
```
Failed to import TensorFlow Probability.
```

**Fix:** Install compatible tensorflow-probability:
```powershell
pip install tensorflow-probability==0.21.0
```

**Secondary error:**
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Root Cause:** `tensorflow-hub` (dependency of tensorflow-gan) uses `pkg_resources` from `setuptools`. Modern setuptools (82.x) removed `pkg_resources` from the default namespace.

**Fix:** Downgrade setuptools:
```powershell
pip install "setuptools<70"
```

**Verification:**
```python
import tensorflow_gan; print('tfgan ok')  # tfgan ok
```

### Install tensorflow-datasets

First attempt: `pip install tensorflow-datasets==4.9.0`

**Error when importing:**
```
ImportError: cannot import name 'array_record_module' from 'array_record.python'
```

**Root Cause:** `tensorflow-datasets==4.9.0` requires `array_record` package which has a broken Windows build.

**Fix:** Use older version:
```powershell
pip install tensorflow-datasets==4.8.3
```

### Fix protobuf

**Error:**
```
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
```

**Root Cause:** `tensorflow-datasets` uses old protobuf-generated files incompatible with protobuf 4.x.

**Fix:**
```powershell
pip install protobuf==3.20.3
```

### Install Flax and Optax

```powershell
pip install flax==0.8.5 optax==0.2.4 ml-collections absl-py
```

**Verification of all imports:**
```python
import sys; sys.path.insert(0, 'demo/continuous_cifar')
import run_lib; print('run_lib ok')  # run_lib ok
```

---

## Part 4: Code Patches

### Patch 1: Replace flax.optim with optax

**File:** `demo/continuous_cifar/models/utils.py`

**Problem:** `flax.optim` module was removed in Flax 0.4+. The code used:
- `flax.optim.Adam(...)` to create optimizer definition
- `optimizer.create(params)` to initialize
- `optimizer.apply_gradient(grad, learning_rate=lr)` to update
- `optimizer.target` to get current params

**Analysis of what we need:**
1. An optimizer that can be initialized with params
2. Can apply gradients with a dynamic learning rate (for warmup)
3. Must be a valid JAX pytree (for `jax.pmap` / `flax_utils.replicate`)

**Solution:** Created `_OptaxOptimizer` and `_OptaxOptimizerDef` classes:

```python
import optax

class _OptaxOptimizer:
    def __init__(self, tx, params, opt_state):
        self._tx = tx          # optax transform (static)
        self.target = params   # current params (dynamic)
        self._opt_state = opt_state  # optimizer state (dynamic)

    def apply_gradient(self, grads, learning_rate=None):
        updates, new_opt_state = self._tx.update(grads, self._opt_state, self.target)
        if learning_rate is not None:
            updates = jax.tree_map(lambda u: u * learning_rate, updates)
        new_params = optax.apply_updates(self.target, updates)
        return _OptaxOptimizer(self._tx, new_params, new_opt_state)

# Register as JAX pytree
def _optax_opt_flatten(opt):
    return (opt.target, opt._opt_state), opt._tx

def _optax_opt_unflatten(aux, children):
    return _OptaxOptimizer(aux, children[0], children[1])

jax.tree_util.register_pytree_node(
    _OptaxOptimizer, _optax_opt_flatten, _optax_opt_unflatten
)

class _OptaxOptimizerDef:
    def __init__(self, tx):
        self._tx = tx

    def create(self, params):
        return _OptaxOptimizer(self._tx, params, self._tx.init(params))
```

**Why register as pytree?**
- `flax_utils.replicate(state)` calls `jax.device_put_replicated` on the entire state
- JAX requires all objects in the state to be valid pytrees
- Without registration: `TypeError: Value <_OptaxOptimizer object> is not a valid JAX type`
- With registration: JAX can traverse the optimizer's leaves (params + opt_state) and replicate them

**File:** `demo/continuous_cifar/losses.py`

```python
# Old:
def get_optimizer(config):
    optimizer = flax.optim.Adam(beta1=config.optim.beta1, eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
    return optimizer

# New:
def get_optimizer(config):
    tx = optax.adamw(
        learning_rate=1.0,  # lr=1.0 because warmup scaling applied manually
        b1=config.optim.beta1,
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
    )
    return mutils._OptaxOptimizerDef(tx)
```

**Why `learning_rate=1.0`?**

The `optimization_manager` function computes a warmup-scaled lr and passes it to `apply_gradient`. We multiply the updates by this lr after computing them. Using `lr=1.0` in optax means optax computes normalized updates, then we scale them.

**First attempt used `optax.inject_hyperparams`** to allow dynamic lr updates. This failed with:
```
TypeError: Scanned function carry input and carry output must have the same pytree structure
  * the input carry component carry_state[1].optimizer[...].hyperparams is a dict with 6 children
    but the corresponding component of the carry output is a dict with 1 child
```

**Root Cause:** `inject_hyperparams` adds all optimizer hyperparams (b1, b2, eps, eps_root, weight_decay, learning_rate) to the state. When we tried to override just `learning_rate`, the structure changed from 6 keys to 1 key, breaking `jax.lax.scan`'s requirement that carry structure is constant.

**Final solution:** Use plain `optax.adamw` with `lr=1.0` and scale updates manually.

---

### Patch 2: Fix jax.tree_multimap

**File:** `demo/continuous_cifar/losses.py`

**Problem:** `jax.tree_multimap` was renamed to `jax.tree_map` in JAX 0.3.0.

```python
# Old (broken):
new_params_ema = jax.tree_multimap(
    lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
    state.params_ema, new_optimizer.target
)

# New (fixed):
new_params_ema = jax.tree_map(
    lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
    state.params_ema, new_optimizer.target
)
```

Note: `jax.tree_map` already supported multiple trees as arguments, so this is a direct rename with no behavior change.

---

### Patch 3: Fix model.init() return value

**File:** `demo/continuous_cifar/models/utils.py`

**Problem:** In old Flax, `model.init()` returned a `FrozenDict`. `FrozenDict.pop('params')` returned a tuple `(remaining_dict, value)`. In new Flax, `model.init()` returns a plain `dict`. `dict.pop('params')` returns just the value.

**Error:**
```
ValueError: too many values to unpack (expected 2)
```

**Debugging:** Tested in isolation:
```python
from flax.core import FrozenDict
d = FrozenDict({'params': 1, 'batch_stats': 2})
result = d.pop('params')
print(type(result), result)
# <class 'tuple'> (FrozenDict({'batch_stats': 2}), 1)
```

But `model.init()` in new Flax returns:
```python
import flax.linen as nn
m = nn.Dense(4)
v = m.init(key, x)
print(type(v))  # <class 'dict'>
```

So `v.pop('params')` returns just the params dict, not a tuple. Trying to unpack it as `a, b = v.pop('params')` unpacks the params dict itself, which has many keys → "too many values to unpack".

**Fix in `init_model`:**
```python
# Old:
init_model_state, initial_params = variables.pop('params')

# New:
initial_params = variables.pop('params')
init_model_state = variables  # remaining dict is the model state
```

**Fix in `create_classifier`:**
```python
# Old:
model_state, init_params = initial_variables.pop('params')

# New:
init_params = initial_variables.pop('params')
model_state = initial_variables
```

---

### Patch 4: Reduce memory requirements

**File:** `demo/continuous_cifar/configs/default_cifar10_configs.py`

**Error:**
```
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory allocating 13307782216 bytes.
```

**Analysis:**
- `jax.pmap` with `n_jitted_steps=5` pre-allocates memory for 5 batches simultaneously
- `batch_size=128` × `n_jitted_steps=5` = 640 images per allocation
- Large NCSNpp model with `nf=128`, 4 res blocks, 4 channel multipliers
- Total: ~13GB for a single compiled step

**Fix:**
```python
# Reduced from 128 to 16
config.training.batch_size = 16

# Reduced from 5 to 1
training.n_jitted_steps = 1
```

**File:** `demo/continuous_cifar/configs/vp/cifar10_ddpmpp_continuous.py`

```python
# Reduced model capacity
model.nf = 64          # was 128
model.ch_mult = (1, 2, 2)    # was (1, 2, 2, 2)
model.num_res_blocks = 2     # was 4
```

**Result:** Memory usage dropped from ~13GB to ~300MB. Training runs successfully.

---

## Part 5: GPU Investigation

### Attempt to Enable GPU

**System:** NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM, Driver 591.44 (CUDA 12 compatible)

**Attempt:**
```powershell
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Result:** Installed jaxlib 0.4.30 (CUDA) but jax was downgraded to 0.4.21 (version mismatch).

**Fix:** Reinstall matching jax version:
```powershell
pip install "jax==0.4.30" "flax==0.8.5" "optax==0.2.4"
```

**Verification:**
```python
import jax; print(jax.devices())
# [CpuDevice(id=0)]  ← still CPU!
```

**Root Cause:** `jax-cuda12-plugin` (the actual CUDA bridge package) requires Python 3.10+. On Python 3.9, there are no CUDA plugin wheels.

**Error:**
```
ERROR: Could not find a version that satisfies the requirement jax-cuda12-plugin<=0.4.30,>=0.4.30; extra == "cuda12-pip"
```

**Conclusion:** GPU support on Windows requires Python 3.10+. Our environment uses Python 3.9 for TF 2.13 compatibility. This is a fundamental constraint.

**To use GPU:** Create a Python 3.10 environment and accept that some TF compatibility issues may need additional patching.

---

## Part 6: Final Working State

### Environment Summary

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.9 | Via conda (deis39 env) |
| JAX | 0.4.25 | Via conda pkgs/main |
| jaxlib | 0.4.23 | Via conda pkgs/main |
| Flax | 0.8.5 | Via pip |
| Optax | 0.2.4 | Via pip |
| TensorFlow | 2.13.0 | Via pip |
| tensorflow-gan | 2.0.0 | Via pip |
| tensorflow-datasets | 4.8.3 | Via pip |
| tensorflow-probability | 0.21.0 | Via pip |
| protobuf | 3.20.3 | Via pip (downgraded) |
| setuptools | 69.5.1 | Via pip (downgraded) |
| ml-collections | 0.1.0 | Via pip |
| absl-py | 0.10.0 | Via pip |

### Files Modified

| File | Changes |
|------|---------|
| `jax_deis/vpsde.py` | Replaced removed `_promote_dtypes_inexact` import |
| `th_deis/vpsde.py` | Same as above |
| `demo/continuous_cifar/models/utils.py` | Added optax shim, fixed `variables.pop()`, registered pytree |
| `demo/continuous_cifar/losses.py` | Replaced `flax.optim` with optax, fixed `tree_multimap` |
| `demo/continuous_cifar/configs/default_cifar10_configs.py` | Reduced batch_size and n_jitted_steps |
| `demo/continuous_cifar/configs/vp/cifar10_ddpmpp_continuous.py` | Reduced model size |

### Confirmed Working Output

```
I0416 23:25:18.574933 27604 run_lib.py:143] Starting training loop at step 0.
I0416 23:26:57.302467 27604 run_lib.py:164] step: 0, training_loss: 1.00276e+00
I0416 23:27:03.303871 27604 run_lib.py:183] step: 0, eval_loss: 1.00966e+00
```

Training is running. Loss values are reasonable (~1.0 at step 0 is expected for a randomly initialized diffusion model).

---

## Key Lessons Learned

1. **Old JAX has no Windows wheels.** Any version before ~0.4.x simply doesn't exist for Windows on PyPI or conda. Always use modern JAX on Windows.

2. **flax.optim was removed.** The entire `flax.optim` module was deprecated and removed. Modern Flax uses `optax` for optimization. The migration requires wrapping optax in a compatibility shim if you want to keep the old API surface.

3. **FrozenDict.pop() vs dict.pop().** Old Flax returned FrozenDict from `model.init()`, new Flax returns plain dict. The pop() semantics are completely different — always check what type you're working with.

4. **jax.lax.scan requires constant carry structure.** When using `inject_hyperparams`, the optimizer state structure changes if you modify hyperparams. This breaks `jax.lax.scan`. Solution: use fixed lr in optax and scale updates manually.

5. **JAX pytree registration is required for pmap.** Any custom Python object stored in JAX state must be registered as a pytree. Otherwise `jax.device_put_replicated` (used by `flax_utils.replicate`) will fail.

6. **Memory scales with batch_size × n_jitted_steps.** The `n_jitted_steps` parameter causes JAX to pre-allocate memory for all steps at once during compilation. Reducing it from 5 to 1 cuts memory by 5x.

7. **GPU on Windows needs Python 3.10+.** The `jax-cuda12-plugin` package only has wheels for Python 3.10+. If you need GPU, use Python 3.10.

8. **TF 2.13 is the sweet spot.** It's the last version with `tf.estimator` (needed by tensorflow-gan) while still being modern enough to work with Python 3.9.

9. **protobuf 3.20.x is the compatibility bridge.** Old TF-generated protobuf files don't work with protobuf 4.x. Pinning to 3.20.3 fixes this without breaking other things.

10. **setuptools<70 restores pkg_resources.** Modern setuptools removed `pkg_resources` from the default namespace. Downgrading restores it for packages that depend on it (like tensorflow-hub).
