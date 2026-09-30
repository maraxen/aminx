# Contributing to Aminx

We welcome contributions from the community! To ensure a smooth development process, please follow these guidelines.

## Setting Up Your Development Environment

To get started, you'll need `uv` and `pre-commit` installed.

1.  **Install `uv`**:
    Follow the official installation instructions for `uv` from [astral.sh](https://astral.sh/uv).

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/maraxen/Aminx.git
    cd Aminx
    ```

3.  **Install dependencies**:
    Install all project dependencies (including dev tools) using `uv`:
    ```bash
    uv sync
    ```

4.  **Set up pre-commit hooks**:
    Install the pre-commit hooks to ensure your contributions adhere to our code quality standards.
    ```bash
    pre-commit install
    ```

## Running Pre-commit Hooks

The pre-commit hooks will run automatically when you commit your changes. To run them manually on all files, use the following command:

```bash
pre-commit run --all-files
```

This will run `ruff` for linting and formatting. Please ensure that all hooks pass before submitting a pull request.

## Type Checking

We use `astral ty` for static type checking. It is **advisory**: it is not enforced via pre-commit hooks and CI does not run it (the suite only runs `pytest`), and a full run currently reports on the order of a thousand diagnostics, so there is no clean baseline to regress against. `ty` is declared in the `dev` extra, so run it manually with:

```bash
uv run --extra dev ty check
```

We welcome contributions to improve type compliance across the codebase, including the `tests/` directory!

## Submitting a Pull Request

1.  Create a new branch for your feature or bug fix.
2.  Make your changes and commit them.
3.  Ensure all pre-commit hooks pass.
4.  Push your branch to your fork.
5.  Open a pull request with a clear description of your changes.

Thank you for contributing to Aminx!