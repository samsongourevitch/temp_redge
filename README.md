# ReDGE

## Implemented samplers

- `redge`
- `redge_cov`
- `reindge`
- `st`
- `gumbel`
- `reinmax`

## Installation

This repository is managed with `uv`:

```bash
uv sync
source .venv/bin/activate
uv pip install -e .
```

## Configure local paths

Before running demos, update `local_paths.py`:

```python
REPO_DIR = Path(__file__).resolve().parents[0]
MODELS_DIR = REPO_DIR / "path_to_models"
DATA_DIR = REPO_DIR / "path_to_data"
```

All scripts resolve checkpoints/datasets through `MODELS_DIR` and `DATA_DIR`.

## Required assets

Download the shared assets first:

- Checkpoints: https://drive.google.com/file/d/1GrZujukxyxijgcXN-0zdYU52dc1YzURs/view?usp=sharing
- Dataset: https://drive.google.com/file/d/19p1_Bha8ngKvNvfz-KEOQUA11-fZeIOE/view?usp=sharing

Then place files under your configured directories:

- Model files under `MODELS_DIR`
- Dataset files under `DATA_DIR` (including `sudoku_data/sudoku_test.csv` for Sudoku MDM eval)

Optional MaskGIT helper:

```bash
python assets/dl_maskgit.py
```

This downloads MaskGIT weights into `MODELS_DIR / "maskgit"`.

## Transformers version note

There is a known split:

- `demo/sudoku_mdm.py` requires `transformers==4.37.2` (hard check in script)
- `demo/maskgit_guidance.py` works with recent `transformers` versions

Switch versions depending on the demo you run:

```bash
uv pip install transformers==4.37.2
```

## Running demos

These scripts use Hydra overrides (`key=value`).

### 1) Sudoku guidance (no MDM prior)

```bash
python demo/sudoku.py \
  sampler=redge \
  sampler.n_steps=3 \
  sampler.t_1=0.9 \
  n_opt_steps=2000 \
  lr=0.05 \
  device=cuda:0
```

### 2) Guided MDM Sudoku generation

```bash
python demo/sudoku_mdm.py \
  sampler=reindge \
  sampler.n_steps=3 \
  sampler.t_1=0.5 \
  demo.guidance.n_opt_steps=5000 \
  demo.guidance.lr=0.1 \
  demo.max_samples=1000 \
  demo.batch_size=64 \
  device=cuda:0
```

### 3) MaskGIT reward guidance

```bash
python demo/maskgit_guidance.py \
  demo.prompt="orange sportscar" \
  demo.label=817 \
  demo.reward_scaling=100 \
  sampler=redge \
  sampler.n_steps=3 \
  sampler.t_1=0.7 \
  device=cuda:0
```

## Hyperparameters and tuning

ReDGE-style samplers expose two main knobs:

- `n_steps` (int): number of DDIM reverse steps
- `t_1` (float): diffusion endpoint / relaxation strength (eqiuvalent to the temperature in Gumbel-based samplers)

Practical defaults:

- `n_steps`: `3-9` (start with `5`)
- `t_1`: `0.3-0.9` (start with `0.5`)

Guidelines:

- If optimization is unstable/noisy, increase `t_1`
- If samples are too soft/biased, decrease `t_1` or increase `n_steps` slightly

## Credits

- Sudoku model scripts were adapted from: https://github.com/HKUNLP/diffusion-vs-ar
- MaskGIT integration was adapted from: https://github.com/valeoai/Halton-MaskGIT
