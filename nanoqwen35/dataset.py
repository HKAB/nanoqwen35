"""
Dataset utilities: list parquet files from a data directory.
"""

import os
from nanoqwen35.common import get_base_dir

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

def list_parquet_files(data_dir=None):
    """Returns sorted full paths to all parquet files in data_dir."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet') and not f.endswith('.tmp'))
    return [os.path.join(data_dir, f) for f in files]

def list_parquet_files_by_domain(root_dir):
    """Returns {domain_name: [sorted parquet paths]} for each non-empty subdirectory of root_dir."""
    domains = {}
    for entry in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, entry)
        if os.path.isdir(subdir):
            files = sorted(f for f in os.listdir(subdir) if f.endswith('.parquet') and not f.endswith('.tmp'))
            if files:
                domains[entry] = [os.path.join(subdir, f) for f in files]
    return domains


def get_pretokenize_metadata(dataset_root: str):
    """Returns parsed pretokenize_metadata.json from dataset_root, or None if not found."""
    import json
    meta_path = os.path.join(dataset_root, "pretokenize_metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)