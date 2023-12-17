import glob
import os
from itertools import groupby
from typing import Dict, List


def get_filter_raw_image_paths(dir_path: str) -> Dict[str, List[str]]:
    files = [os.path.split(file_path)[-1] for file_path in glob.glob(os.path.join(dir_path, "*.*"))]
    grouped_files = {key: list(values) for key, values in groupby(files, lambda x: ''.join(x.split('_')[:2]))}
    filtered_files = {key: values for key, values in grouped_files.items() if len(values) == 13}
    return filtered_files
