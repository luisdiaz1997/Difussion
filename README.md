# Realistica

A diffusion model package for realistic image generation.

## Installation

```bash
pip install -e .
```

For GPU support on Linux with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

## Package Structure

```
realistica/
├── __init__.py          # Package initialization
├── models.py            # Diffusion model architectures
├── training_utils.py    # Training utilities and helpers
└── data_utils.py        # Data loading and preprocessing

notebooks/               # Jupyter notebooks for examples and experiments
```

## Development

Install in development mode:
```bash
pip install -e .
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- See requirements.txt for full dependencies
