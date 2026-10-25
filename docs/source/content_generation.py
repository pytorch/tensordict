import os
import shutil
from pathlib import Path

FILE_DIR = os.path.dirname(__file__)
TUTORIALS_GEN_DIR = "reference/generated/tutorials"
TUTORIALS_MEDIA_GEN_DIR = "reference/generated/tutorials/media"


def generate_tutorial_references(tutorial_path: str, file_type: str) -> None:
    """Creates a python file per tutorial script.

    Sphinx natively doesn't support adding files from outside its root directory. To include the tutorials in
    our docs (https://pytorch.org/rl/) each entry is locally copied.

    Args:
        tutorial_path (str): path to the tutorial scripts.
    """
    # Create target dir
    if file_type == "tutorial":
        target_path = os.path.join(FILE_DIR, TUTORIALS_GEN_DIR)
    else:
        target_path = os.path.join(FILE_DIR, TUTORIALS_MEDIA_GEN_DIR)
    Path(target_path).mkdir(parents=True, exist_ok=True)

    # Iterate tutorial files and copy
    file_paths = [
        os.path.join(tutorial_path, f)
        for f in os.listdir(tutorial_path)
        if f.endswith((".py", ".rst", ".png"))
    ]

    for file_path in file_paths:
        shutil.copyfile(file_path, os.path.join(target_path, Path(file_path).name))
