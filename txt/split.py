import argparse
from pathlib import Path
import random
from typing import List, Tuple, Optional


def split_data(
    filenames: List[str],
    val_ratio: float,
    test_ratio: Optional[float] = None
) -> Tuple[List[str], List[str], List[str]]:
    """Splits a list of filenames into train, validation, and test sets.

    Args:
        filenames (List[str]): List of filenames to split.
        val_ratio (float): Ratio of names to allocate to the validation set.
        test_ratio (Optional[float]): Ratio of names to allocate to the test
            set (if any). Defaults to None

    Returns:
        Tuple[List[str], List[str], List[str]]: A tuple containing train,
            validation, and test lists.
    """
    random.shuffle(filenames)
    total = len(filenames)

    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio) if test_ratio else 0

    val_set = filenames[:val_size]
    test_set = filenames[val_size : val_size + test_size]
    train_set = filenames[val_size + test_size :]

    return train_set, val_set, test_set


def main(
    input_file: Path,
    output_dir: Path,
    val_ratio: float,
    test_ratio: Optional[float],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = input_file.read_text().strip().splitlines()
    train_set, val_set, test_set = split_data(filenames, val_ratio, test_ratio)

    (output_dir / "train.txt").write_text("\n".join(train_set))
    (output_dir / "val.txt").write_text("\n".join(val_set))
    if test_set:
        (output_dir / "test.txt").write_text("\n".join(test_set))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shuffle and split a list of filenames into train, "
                    "validation, and test sets."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Path to the input file containing names.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Directory to save the output files.",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--val",
        type=float,
        default=0.2,
        help="Proportion for validation (default: %(default)s).",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=float,
        default=None,
        help="Proportion of names for test (default: %(default)s).",
    )
    args = parser.parse_args()
    main(
        input_file=args.input,
        output_dir=args.output,
        val_ratio=args.val,
        test_ratio=args.test,
    )
