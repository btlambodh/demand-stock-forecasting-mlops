# Set up the conda environment
How to set up the conda environment, including all the important steps like sourcing `.bashrc`, creating/activating the environment, and exporting/importing with `environment.yml` and/or `requirements.txt`.

---

## 🚀 **How to Set Up Conda Environment**

### 1. **Initialize Conda for Your Shell**

If you haven’t used conda in this terminal before, initialize it:

```bash
conda init
source ~/.bashrc  # Apply the initialization in your current shell
```

*If you see a message about closing and reopening the shell, you can also just run `source ~/.bashrc` instead.*

---

### 2. **Create a New Conda Environment**

**Option A: Using a YAML file** (recommended if you have `environment.yml`):

```bash
conda env create -f environment.yml
```

**Option B: Manually, then pip install from requirements.txt:**

```bash
conda create -n dsfenv python=3.10  # or your preferred version
conda activate dsfenv
pip install -r requirements.txt
```

---

### 3. **Activate the Environment**

```bash
conda activate dsfenv
```

---

### 4. **Export the Environment** (for reproducibility)

To save your environment (including pip dependencies) for others to use:

```bash
conda env export -n dsfenv > environment.yml
```

---

### 5. **(Optional) Using requirements.txt from environment.yml**

If you want pip to install dependencies from `requirements.txt` during environment creation, in your `environment.yml` add:

```yaml
dependencies:
  - python=3.10
  - pip
  - pip:
      - -r requirements.txt
```

---

### 6. **Deactivate the Environment**

When you are done:

```bash
conda deactivate
```

---

## 📝 **Summary of Useful Commands**

| Task                        | Command                                        |
| --------------------------- | ---------------------------------------------- |
| Initialize conda in shell   | `conda init`<br>`source ~/.bashrc`             |
| Create environment (YAML)   | `conda env create -f environment.yml`          |
| Create environment (manual) | `conda create -n dsfenv python=3.10`           |
| Activate environment        | `conda activate dsfenv`                        |
| Install pip deps            | `pip install -r requirements.txt`              |
| Export environment          | `conda env export -n dsfenv > environment.yml` |
| Deactivate environment      | `conda deactivate`                             |

---

**That’s it! You’re ready to go!**

Let me know if you want this as a code-ready markdown snippet or want it tailored for a specific stack.
