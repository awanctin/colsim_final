import glob
import os


def drop_if_stale(module_file, dependency_files):
    """Delete the numba disk cache of a module if a dependency is newer.

    Args:
        module_file (str): Path of the module that owns the cache
        dependency_files (list): Paths of the files the kernels call

    Returns:
        int: Number of cache files deleted
    """
    stem = os.path.splitext(os.path.basename(module_file))[0]
    cache_dir = os.path.join(os.path.dirname(module_file), "__pycache__")
    entries = [p for ext in ("nbi", "nbc") for p in glob.glob(os.path.join(cache_dir, f"{stem}.*.{ext}"))]
    if not entries:
        return 0

    newest_dependency = max(os.path.getmtime(f) for f in dependency_files)
    try:
        oldest_entry = min(os.path.getmtime(p) for p in entries)
    except FileNotFoundError:
        return 0
    if newest_dependency <= oldest_entry:
        return 0

    removed = 0
    for p in entries:
        try:
            os.remove(p)
            removed += 1
        except FileNotFoundError:
            continue
    return removed
