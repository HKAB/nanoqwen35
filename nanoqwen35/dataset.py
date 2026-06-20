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


def list_all_parquet_files(root_dir: str) -> list:
    """Returns a sorted flat list of all .parquet files under root_dir (recursive, no .tmp files)."""
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.parquet') and not f.endswith('.tmp'):
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def get_pretokenize_metadata(dataset_root: str):
    """Returns parsed pretokenize_metadata.json from dataset_root, or None if not found."""
    import json
    meta_path = os.path.join(dataset_root, "pretokenize_metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def get_merged_metadata(dataset_root: str):
    """Returns parsed merged_metadata.json from dataset_root, or None if not found."""
    import json
    meta_path = os.path.join(dataset_root, "merged_metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)