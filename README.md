# dimsim

[![CI](https://github.com/openforcefield/dimsim/actions/workflows/gh-ci.yaml/badge.svg)](https://github.com/openforcefield/dimsim/actions/workflows/gh-ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/dimsim/badge/?version=latest)](https://dimsim.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/openforcefield/dimsim/branch/main/graph/badge.svg)](https://codecov.io/gh/openforcefield/dimsim)

Distributed simulation package

## Overview

`dimsim` is a Python package for distributed simulation.

## Installation

### From source

```bash
git clone https://github.com/openforcefield/dimsim.git
cd dimsim
pip install -e .
```

### Development installation

For development, use conda/mamba to create an environment from the provided file:

```bash
mamba env create -f devtools/conda-envs/dev.yaml
mamba activate dimsim-dev
pip install -e .
```

Alternatively, install with pip using the dev extras:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import dimsim

print(f"dimsim version: {dimsim.__version__}")
```

## Documentation

Full documentation is available at [dimsim.readthedocs.io](https://dimsim.readthedocs.io).

## Testing

Run tests with pytest:

```bash
pytest
```

Or with coverage:

```bash
pytest --cov=dimsim --cov-report=html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Lily Wang

## Acknowledgments

This package structure follows best practices from the [MolSSI Cookiecutter](https://github.com/molssi/cookiecutter-cms).

