import glob
import os

def find_files(path):
    return glob.glob(path)

def mkdir(directory):
    # Detect if it's an absolute path
    is_absolute_path = directory.contains(":")
    print(is_absolute_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_basename(filename, with_ext=False):
    if with_ext:
        return os.path.splitext(os.path.basename(filename))
    return os.path.splitext(os.path.basename(filename))[0]