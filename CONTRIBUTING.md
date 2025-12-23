# Contributing to Arcane

Thank you for your interest in contributing to Arcane! We welcome contributions from the community to help improve this educational GPT-style transformer model.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

Before contributing, please:
1. Read the [README.md](README.md) to understand the project
2. Review the [Code of Conduct](CODE_OF_CONDUCT.md)
3. Check existing [issues](https://github.com/AlexLee1109/Arcane/issues) and [pull requests](https://github.com/AlexLee1109/Arcane/pulls)

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Arcane.git
   cd Arcane
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a new branch for your feature**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

- **Bug Fixes**: Found a bug? Submit a fix!
- **Features**: New features that align with the project goals
- **Documentation**: Improvements to README, code comments, or tutorials
- **Performance**: Optimizations and efficiency improvements
- **Examples**: New examples or use cases

### Before You Start

1. **Check for existing issues**: Search the [issue tracker](https://github.com/AlexLee1109/Arcane/issues) to see if your idea or bug has been discussed
2. **Create an issue**: For significant changes, open an issue first to discuss your proposal
3. **Keep changes focused**: Submit one feature or fix per pull request

## Code Style

We follow Python best practices and PEP 8 guidelines:

- Use **4 spaces** for indentation (no tabs)
- Maximum line length of **88 characters** (Black formatter standard)
- Use meaningful variable and function names
- Add docstrings to classes and functions
- Type hints are encouraged

### Formatting

We use **Black** for code formatting:
```bash
black .
```

We use **flake8** for linting:
```bash
flake8 .
```

## Testing

While the project currently doesn't have a comprehensive test suite, we encourage:
- Manual testing of your changes
- Verifying that existing functionality still works
- Adding tests for new features when possible

To test the chatbot:
```bash
python main.py
```

To test training (requires data):
```bash
python train.py
```

## Submitting Changes

1. **Commit your changes**
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```

   Write clear, concise commit messages:
   - Use present tense ("Add feature" not "Added feature")
   - Be specific about what changed
   - Reference issue numbers if applicable (#123)

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**
   - Go to the [Arcane repository](https://github.com/AlexLee1109/Arcane)
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template with:
     - Description of changes
     - Related issue numbers
     - Testing performed
     - Screenshots (if UI changes)

4. **Respond to feedback**
   - Be open to suggestions
   - Make requested changes promptly
   - Ask questions if anything is unclear

## Reporting Issues

When reporting bugs or requesting features:

### Bug Reports Should Include:
- Clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, PyTorch version)
- Error messages or logs
- Screenshots (if applicable)

### Feature Requests Should Include:
- Clear description of the feature
- Use cases and benefits
- Possible implementation approach
- Any alternative solutions considered

## Code Review Process

- All submissions require review before merging
- Maintainers will review your code and may request changes
- Once approved, a maintainer will merge your PR
- Your contribution will be acknowledged in the project

## Questions?

If you have questions about contributing:
- Open an issue with the "question" label
- Reach out to the maintainers

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

Thank you for contributing to Arcane! 🚀
