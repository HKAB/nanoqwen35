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