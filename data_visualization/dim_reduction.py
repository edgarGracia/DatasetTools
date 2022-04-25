from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import argparse
import random

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

sns.set(rc={'figure.figsize':(11.7,8.27)})



def load_data(root_path: Path, subsample: int = None, classes: list = None):
    x = []
    y = []

    for class_path in root_path.iterdir():
        if classes is not None and class_path.name not in classes:
            continue
        
        samples_paths = list(class_path.iterdir())

        if subsample is not None:
            random.shuffle(samples_paths)
            samples_paths = samples_paths[:min(len(samples_paths), subsample)]
            
        for sample in samples_paths:
            x.append(np.load(sample).flatten())
            y.append(class_path.name)

    x = np.array(x)
    y = np.array(y, dtype="str")

    print(f"Features dim: {x.shape}, classes: {set(y)}")
    return x, y


def plot_tsne(x: np.ndarray, y: np.ndarray, out_path: Path = None,
    scale: bool = True, title: str = None):
    
    palette = sns.color_palette("colorblind", len(set(y)))

    if scale:
        x = StandardScaler().fit_transform(x)

    tsne = TSNE()
    X_embedded = tsne.fit_transform(x)
    
    plot = sns.scatterplot(
        X_embedded[:,0],
        X_embedded[:,1],
        hue=y,
        legend='full',
        palette=palette
    )

    if title is not None:
        plot.set_title(title)

    if out_path is not None:
        plt.savefig(str(out_path))
    else:
        plt.show()
    plt.close()
    

def plot_umap(x: np.ndarray, y: np.ndarray, out_path: Path = None,
    scale: bool = True, title: str = None):
    
    import umap # pip install umap-learn
    
    palette = sns.color_palette("colorblind", len(set(y)))
    
    if scale:
        x = StandardScaler().fit_transform(x)
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(x)
    
    plot = sns.scatterplot(
        embedding[:,0],
        embedding[:,1],
        hue=y,
        legend='full',
        palette=palette
    )
    
    if title is not None:
        plot.set_title(title)

    if out_path is not None:
        plt.savefig(str(out_path))
    else:
        plt.show()
    plt.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
    Apply a tsne or a umap over high dimensionality data to visualize it on a 2D plot.
    Data must be stored on individual .npy files on different folders for each class:
      root/
        |- Class_1/
        |    |- sample_1.npy
        |    |- sample_2.npy
        |    |- ...
        |- Class_2/
        |- ...
    """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "features",
        help="Root path to the features",
        type=Path
    )
    parser.add_argument(
        "--scale",
        help="Automatic scaling of the features",
        action="store_true"
    )
    parser.add_argument(
        "--tsne",
        action="store_true"
    )
    parser.add_argument(
        "--umap",
        action="store_true"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image",
        type=Path,
        required=False
    )
    parser.add_argument(
        "--title",
        required=False,
        help="Plot title"
    )
    parser.add_argument(
        "--subsample",
        help="Take only n random samples from each class",
        type=int,
        required=False
    )
    parser.add_argument(
        "--classes",
        help="List of classes to process separated by ','",
        required=False
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

    classes = args.classes.split(",") if args.classes is not None else None
    
    x, y = load_data(root_path=args.features, subsample=args.subsample,
        classes=classes)

    if args.tsne:
        out = (args.output.parent.joinpath("tsne_" + args.output.name)
            if args.umap and args.output is not None else args.output)
        plot_tsne(x, y, out_path=out, scale=args.scale, title=args.title)
    if args.umap:
        out = (args.output.parent.joinpath("umap_" + args.output.name)
            if args.tsne and args.output is not None else args.output)
        plot_umap(x, y, out_path=out, scale=args.scale, title=args.title)
