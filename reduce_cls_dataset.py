from pathlib import Path
from tqdm import tqdm
import argparse
import random
import shutil



def reduce_dataset(in_path: Path, out_path: Path, max_samples: int = None,
    percent: float = None, shuffle: bool = False):
    
    assert (max_samples is not None) ^ (percent is not None)
    assert in_path != out_path

    out_path.mkdir(parents=True)

    for cl in tqdm(list(in_path.iterdir())):
        cl_out = out_path.joinpath(cl.name)
        cl_out.mkdir()
        
        files = list(cl.iterdir())
        
        if shuffle:
            random.shuffle(files)
        
        if max_samples is not None:
            files = files[:min(len(files), max_samples)]
        elif percent is not None:
            files = files[:min(len(files), int(len(files)*percent))]
        
        for f in files:
            shutil.copy(f, cl_out.joinpath(f.name))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""
    Create a copy of a dataset with fewer samples.
    The root path of the dataset should have the following structure:
      root/
        |- Class_1/
        |    |- sample_1
        |    |- sample_2
        |    |- ...
        |- Class_2/
        |- ...
    """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "input_path",
        type=Path,
        help="Root path of the dataset, containing one folder for each class"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to save the reduced dataset"
    )
    parser.add_argument(
        "--max-samples",
        required=False,
        type=int,
        help="Maximum number of samples per each class"
        )
    parser.add_argument(
        "--percent",
        required=False,
        type=float,
        help="Percentage of samples to take from each class"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffles the samples"
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        help="Random seed"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    
    reduce_dataset(args.input_path, args.output_path, args.max_samples,
        args.percent, args.shuffle)
