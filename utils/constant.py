import os

# Repository root (parent of utils/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Default paths. Override via environment variables before launching the script,
# or by editing the values below for your setup.
DEFAULTS = {
    "base_dir": REPO_ROOT,
    "data_dir": os.path.join(REPO_ROOT, "data"),
    "HF_HOME": os.path.join(REPO_ROOT, "hf_cache"),
    "TRANSFORMERS_CACHE": os.path.join(REPO_ROOT, "hf_cache", "transformers"),
    "HF_DATASETS_CACHE": os.path.join(REPO_ROOT, "hf_cache", "datasets"),
}


def get_constant():
    """Populate os.environ with the paths above, preferring pre-set env vars."""
    for var_name, default in DEFAULTS.items():
        os.environ.setdefault(var_name, default)
