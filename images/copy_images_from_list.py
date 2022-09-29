from pathlib import Path
from tqdm import tqdm
import argparse
import shutil


def copy_from_list(source_path: Path, dest_path: Path, list_file: Path, 
    is_absolute: bool, move: bool = False):

    dest_path.mkdir(exist_ok=True, parents=True)
    
    files_to_copy = list_file.read_text().splitlines()
    pbar = tqdm(files_to_copy)
    for f in pbar:
        if f:
            src_path = str(source_path / f) if not is_absolute else f
            pbar.set_description(src_path)
            if move:
                shutil.move(src_path, dest_path)
            else:
                shutil.copy(src_path, dest_path)
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="""Copies the files
    on a list to another directory""")
    
    argparser.add_argument(
        "-s",
        "--source",
        help="Source path",
        type=Path,
        required=False
    )
    argparser.add_argument(
        "-d",
        "--dest",
        help="Destination path",
        type=Path,
        required=True
    )
    argparser.add_argument(
        "-l",
        "--list",
        help="File with the list of filenames to copy",
        type=Path,
        required=True
    )
    argparser.add_argument(
        "--is-absolute",
        help="Paths on the list are absolute",
        action="store_true"
    )
    argparser.add_argument(
        "--move",
        help="Move the files instead of copy them",
        action="store_true"
    )

    args = argparser.parse_args()

    copy_from_list(args.source, args.dest, args.list,
        args.is_absolute, args.move)
