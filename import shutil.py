import argparse
import os
import shutil
import sys
from pathlib import Path

# --- CONFIGURATION ---
# This is the folder the script will flatten by default.
# I have set it to the folder name usually created when you unzip the file normally.
DEFAULT_TARGET_DIR = r"C:\Users\mjhen\Downloads\orignal data excel sheets"

def unique_destination(root: Path, folder_name: str) -> Path:
    """
    Pick a destination path in `root` that won't overwrite an existing path.
    If `root/folder_name` exists, use `root/folder_name__1`, `__2`, ...
    """
    dest = root / folder_name
    if not dest.exists():
        return dest

    i = 1
    while True:
        candidate = root / f"{folder_name}__{i}"
        if not candidate.exists():
            return candidate
        i += 1

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten folders by moving every nested directory under ROOT into ROOT.\n"
            "Folders already directly under ROOT are left in place.\n"
            "Name collisions are resolved by renaming with __1, __2, ..."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_TARGET_DIR,  # Updated default to your specific path
        help=f"Main directory to flatten (default: {DEFAULT_TARGET_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves, but do not change anything.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Folder name to skip anywhere in the tree (repeatable). "
            "Example: --exclude .git --exclude node_modules"
        ),
    )
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    
    # Safety check
    if not root.exists():
        print(f"ERROR: The directory does not exist: {root}", file=sys.stderr)
        print("Did you unzip the file yet?", file=sys.stderr)
        return 2
    
    if not root.is_dir():
        print(f"ERROR: Not a directory: {root}", file=sys.stderr)
        return 2

    print(f"Flattening directory: {root}")
    print("-" * 40)

    exclude_names = set(args.exclude)

    # Collect directories first (so moving them does not break traversal).
    dirs: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root, topdown=True, followlinks=False):
        dirpath_p = Path(dirpath)

        # Prune excluded and symlink directories so we don't descend into them.
        pruned: list[str] = []
        for d in dirnames:
            if d in exclude_names:
                continue
            p = dirpath_p / d
            if p.is_symlink():
                continue
            pruned.append(d)
            dirs.append(p)
        dirnames[:] = pruned

    # Deepest-first prevents moving a parent before its children.
    dirs.sort(key=lambda p: len(p.relative_to(root).parts), reverse=True)

    moved = 0
    for src in dirs:
        # Already at top level => nothing to do.
        if src.parent == root:
            continue

        dest = unique_destination(root, src.name)

        if args.dry_run:
            print(f"Would move: {src.relative_to(root)} -> {dest.name}")
            continue

        try:
            shutil.move(str(src), str(dest))
            print(f"Moved: {src.relative_to(root)} -> {dest.name}")
            moved += 1
        except Exception as e:
            print(f"ERROR moving {src} -> {dest}: {e}", file=sys.stderr)

    print("-" * 40)
    if args.dry_run:
        print("Dry-run complete (no changes made).")
    else:
        print(f"Done. Moved {moved} folder(s) to the root level.")
        # Optional: remove empty directories left behind
        # for d in dirs:
        #     try: d.rmdir() 
        #     except: pass

    return 0

if __name__ == "__main__":
    # We pass sys.argv[1:] to allow you to override arguments if needed,
    # otherwise it uses the default path defined at the top.
    raise SystemExit(main(sys.argv[1:]))