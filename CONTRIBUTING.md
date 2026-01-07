# Contributing to dimsim

Thank you for your interest in contributing to dimsim! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dimsim.git
   cd dimsim
   ```
3. Install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```

2. Make your changes and ensure:
   - Code follows the project's style guidelines
   - All tests pass
   - New features include tests
   - Documentation is updated if needed

3. Run tests:
   ```bash
   pytest
   ```

4. Check code style:
   ```bash
   black dimsim tests
   isort dimsim tests
   flake8 dimsim tests
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

6. Push to your fork:
   ```bash
   git push origin feature-name
   ```

7. Submit a pull request through GitHub

## Code Style

This project uses:
- **black** for code formatting (line length: 88)
- **isort** for import sorting
- **flake8** for linting
- Type hints are encouraged but not required

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Use pytest fixtures for common test setup

## Documentation

- Update documentation for new features
- Use NumPy-style docstrings
- Build documentation locally to verify:
  ```bash
  cd docs
  make html
  ```

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include a clear description of the changes
- Reference any related issues
- Ensure CI passes before requesting review
- Be responsive to feedback

## Questions?

Feel free to open an issue for questions or discussion!
