from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2


IMG_EXT = (".png", ".jpg", ".jpeg", ".tiff")

# TODO: total count
# TODO: plot appearance
# TODO: output argument
# TODO: auto train val test

def print_stats(results: dict):
    print(f"Num classes: {len(results['classes'])}")
    print(f"[{', '.join(results['classes'])}]")
    print("Class count:")
    print("\n".join([f"{k}: {v}" for k,v in results["class_count"].items()]))


def plot_stats(results: dict):
    # class_count
    plt.bar(
        results["class_count"].keys(),
        results["class_count"].values()
    )
    plt.title(results["dataset_name"])
    plt.show()

    # # size_per_class
    # plt.hist(
    #     [i[0] for s in results["size_per_class"].values() for i in s]
    # )
    # plt.show()


def dataset_stats(in_path: Path):
    
    classes = sorted(list(in_path.iterdir()))

    images_per_class = {
        c: [i for i in c.iterdir() if i.name.lower().endswith(IMG_EXT)]
        for c in classes
    }

    class_count = {k.name: len(v) for k, v in images_per_class.items()}

    # size_per_class = {}
    # for c, imgs_list in images_per_class.items():
    #     size_per_class[c] = []
    #     for img_path in imgs_list:
    #         img = cv2.imread(str(img_path))
    #         h, w, _ = img.shape
    #         size_per_class[c].append([w, h])

    return {
        "dataset_name": in_path.name,
        "classes": [c.name for c in classes],
        "class_count": class_count,
        # "size_per_class": size_per_class
    }


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""
    Get some stats about a dataset.
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
    args = parser.parse_args()

    results = dataset_stats(args.input_path.resolve())
    print_stats(results)
    plot_stats(results)
