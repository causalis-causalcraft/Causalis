# Installation

This guide shows the quickest way to install Causalis directly from GitHub.

## Prerequisites
  - python>=3.7
  - numpy>=1.19.0
  - pandas>=1.0.0
  - plotly>=5.0.0
  - scikit-learn
  - pip
  - pip:
    - doubleml
    - -e .

## Install from pip

```bash
pip install causalis
```

## Install from GitHub
```bash
pip install --upgrade pip
pip install "git+https://github.com/causalis-causalcraft/Causalis.git"
```

### Verify the installation
```bash
python -c "import causalis; print(causalis.__version__)"
```

## Optional: conda environment after cloning the repo
You can also create a conda env using the provided file (useful for local development):
```bash
conda env create -f environment.yml
conda activate causalis
```

You're all set! Import the library to get started:

```python
import causalis
```
