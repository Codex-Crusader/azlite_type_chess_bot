# Contributing to AZ-Lite (AlphaZero-Inspired Chess Engine)

Thank you for your interest in contributing!
This project is still growing, and contributions of all kinds are welcome — code, documentation, testing, or even ideas for improvement.

---

## Getting Started

### Fork the Repository
Click the Fork button at the top of this repo and clone your fork locally:
```bash
git clone https://github.com/<your-username>/azlite_type_chess_bot.git
cd azlite_type_chess_bot
```

### Set Up Your Environment
Create a Python environment (3.9+ recommended) and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run the Tests
Make sure everything works before making changes:
```bash
pytest tests/
```

---

## How to Contribute

### Bug Reports & Feature Requests - Open a GitHub Issue. Please include details and steps to reproduce if it’s a bug.

Code Contributions - Create a new branch and open a Pull Request (PR).
Documentation - Improvements to docs and examples are always welcome.
Tests - Adding or improving test coverage helps everyone.

### Code Style

1. Follow PEP8 for Python code.
2. Keep function and variable names clear and descriptive.
3. Avoid over-engineering — simple and readable code is preferred.
4. Each new feature or bug fix should include/update tests in the tests/ folder.

### Submitting Changes

Create a feature branch:
```bash
git checkout -b feature/my-new-feature
```

Commit your changes with a clear message:
```bash
git commit -m "Add feature: self-play logging"
```

Push your branch and open a Pull Request:
```bash
git push origin feature/my-new-feature
```
---

Please go through CODE_OF_CONDUCT.md.
Thank You
