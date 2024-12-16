## Installation
The acoustic tomography Python package can be installed by downloading the source code. The following sections detail how to download and install the package for each use case.

### Requirements
This package is a Python project designed to work with all actively supported versions of Python. It is highly recommended that users work within a virtual environment to maintain a clean and sandboxed environment. The simplest way to get started with virtual environments is through [conda](https://docs.conda.io/projects/conda/en/latest/) or [venv](https://docs.python.org/3/library/venv.html).

> **Warning**
> Installing into a Python environment that contains a previous version of the package may cause conflicts. It is recommended to install the package into a new virtual environment to avoid compatibility issues.

> **Note**
> This project uses [Poetry](https://python-poetry.org/) to manage dependencies and build the `pyproject.toml` file. However, the package has not been published to PyPI or Conda. Installation is only possible via the source code.

---

### Source Code Installation
Developers and users intending to inspect the source code or run the provided examples can install the package directly from the GitHub repository. The following steps outline the process:

#### Step 1: Download the Source Code
```bash
# Clone the repository from GitHub
# Replace <repo-url> with the actual repository URL for the acoustic tomography project
git clone -b main <repo-url>
```

#### Step 2: Create and Activate a Virtual Environment
```bash
# Using conda
conda create -n atom python=3.12
conda activate atom
pip install -e <path to setup.py>

# OR using venv
python -m venv atom
source atom/bin/activate  # On Windows, use: atom\Scripts\activate
pip install -e <path to setup.py>
```

#### Step 3: Install Dependencies with Poetry
```bash
# Navigate to the project directory
cd <repository-name>

# Install dependencies and set up the environment
poetry install
```

#### Step 4: Verify the Installation
Open a Python interpreter and import the package to ensure it has been installed correctly:
```python
import atom
help(atom)
```
---

### Developer Installation
For users who wish to contribute to the codebase, the process is similar to the source code installation but includes additional steps for setting up development tools:

#### Step 1: Install Development Dependencies
After cloning the repository and creating a virtual environment, install the development dependencies:
```bash
poetry install --with dev
```

#### Step 2: Pre-Commit Hooks
Set up pre-commit hooks to maintain code quality:
```bash
poetry run pre-commit install
```

---

### Updating the Package
To update your local copy of the repository, use the following commands:

#### From Source
```bash
# Save any changes and ensure you're on the main branch
git checkout main

# Pull the latest changes from GitHub
git pull origin main

# Reinstall dependencies if necessary
poetry install
```

---

### Additional Notes
- The repository includes a set of Jupyter notebooks with examples demonstrating the theory, workflows, and applications of acoustic tomography. These notebooks are located in the `examples/notebooks/` directory.
- A comprehensive Python API reference is included in the documentation to guide users in leveraging the library’s functionality for custom workflows.

By following these steps, you can start exploring and contributing to the exciting field of acoustic travel time tomography!

