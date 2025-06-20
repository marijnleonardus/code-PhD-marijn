import os
import sys

def add_local_paths(script_file, relative_paths):
    script_dir = os.path.dirname(os.path.abspath(script_file))
    for rel_path in relative_paths:
        abs_path = os.path.abspath(os.path.join(script_dir, rel_path))
        if abs_path not in sys.path:
            sys.path.append(abs_path)
            