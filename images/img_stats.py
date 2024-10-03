import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from tqdm import tqdm

sns.set()


class Mean:
    def __init__(self):
        self.count_pix = 0
        self.sum_ch = np.zeros((4))

    def update(self, img: np.ndarray):
        self.count_pix += img.shape[0] * img.shape[1]
        self.sum_ch += img.sum((0, 1))

    def compute(self) -> np.ndarray:
        mean = self.sum_ch / self.count_pix
        return mean

    def print(self):
        mean = self.compute()
        mean = mean[[2, 1, 0, 3]]
        print(
            f"Mean (RGBA): [{', '.join([str(i) for i in mean.round(3)])}]"
            + f" ([{', '.join([str(round(i/255., 3)) for i in mean])}])"
        )


class Sizes:
    def __init__(self, plot: bool = False, output_path: Path = None):
        self.plot = plot
        self.output_path = output_path
        self.sizes = []

    def update(self, img: np.ndarray):
        self.sizes.append(np.array(img.shape[:2]))

    def compute(self) -> np.ndarray:
        sizes = np.array(self.sizes)
        h_max = sizes[:, 0].max()
        h_min = sizes[:, 0].min()
        w_max = sizes[:, 1].max()
        w_min = sizes[:, 1].min()
        h_mean = sizes[:, 0].mean()
        w_mean = sizes[:, 1].mean()
        aspect_ratios = sizes[:, 1] / sizes[:, 0]
        ar_mean = aspect_ratios.mean()
        ar_max = aspect_ratios.max()
        ar_min = aspect_ratios.min()

        if self.plot:
            # Sizes histogram
            fig, axs = plt.subplots(2, 1)
            fig.suptitle("Image size histogram")
            sns.histplot(sizes[:, 1], ax=axs[0])
            sns.histplot(sizes[:, 0], ax=axs[1])
            axs[0].set_ylabel("Width")
            axs[1].set_ylabel("Height")

            if self.output_path is None:
                plt.show()
            else:
                plt.savefig(str(self.output_path / "sizes_hist.png"))
            plt.close(fig)

            # Sizes plot
            ax = plt.gca()
            for s in sizes:
                y = s[0] / -2
                x = s[1] / -2
                ax.add_patch(
                    Rectangle((x, y), s[1], s[0], fill=None, alpha=0.3, color="black")
                )
            scale = max(h_max / 1.8, w_max / 1.8)
            plt.xlim([-scale, scale])
            plt.ylim([-scale, scale])
            ax.set_title("Image sizes")
            ax.grid(False)
            ax.set_aspect("equal", adjustable="box")
            if self.output_path is None:
                plt.show()
            else:
                plt.savefig(str(self.output_path / "sizes.png"))

            # Aspect ratio histogram
            plt.gca()
            sns.histplot(aspect_ratios)
            plt.title("Aspect ratio histogram (w/h)")
            plt.xticks(
                [p.get_x() + p.get_width() / 2 for p in plt.gca().patches], rotation=-45
            )
            if self.output_path is None:
                plt.show()
            else:
                plt.savefig(str(self.output_path / "aspect_ratio_hist.png"))

        return {
            "h_max": h_max,
            "h_min": h_min,
            "w_max": w_max,
            "w_min": w_min,
            "h_mean": h_mean,
            "w_mean": w_mean,
            "ar_max": ar_max,
            "ar_min": ar_min,
            "ar_mean": ar_mean,
        }

    def print(self):
        s = self.compute()
        d = 2
        print(f"Width - Mean: {round(s['w_mean'], d)}, Max: {round(s['w_max'], d)}, Min: {round(s['w_min'], d)}")
        print(f"Height - Mean: {round(s['h_mean'], d)}, Max: {round(s['h_max'], d)}, Min: {round(s['h_min'], d)}")
        print(f"Aspect Ratio (w/h) - Mean: {round(s['ar_mean'], d)}, Max: {round(s['ar_max'], d)}, Min: {round(s['ar_min'], d)}")


class STD:
    def __init__(self, mean_channels: np.ndarray):
        self.mean = mean_channels
        self.count_pix = 0
        self.sum_std = np.zeros((4))

    def update(self, img: np.ndarray):
        self.count_pix += img.shape[0] * img.shape[1]
        self.sum_std += ((img - self.mean) ** 2).sum((0, 1))

    def compute(self) -> np.ndarray:
        std = (self.sum_std / self.count_pix) ** 0.5
        return std

    def print(self):
        std = self.compute()
        std = std[[2, 1, 0, 3]]
        print(
            f"STD (RGBA): [{', '.join([str(i) for i in std.round(3)])}]"
            + f" ([{', '.join([str(round(i/255., 3)) for i in std])}])"
        )


class Histogram:
    def __init__(self, output_path: Path = None):
        self.output_path = output_path
        self.counts = np.zeros((4, 256), dtype="int32")

    def update(self, img: np.ndarray):
        hs = [np.histogram(img[:, :, i], bins=256, range=(0, 256))[0] for i in range(4)]
        self.counts += np.array(hs)

    def compute(self) -> np.ndarray:
        # plt.plot(self.counts[3,:], color="black")
        plt.plot(self.counts[0, :], color="b")
        plt.plot(self.counts[1, :], color="g")
        plt.plot(self.counts[2, :], color="r")
        plt.title("Histogram")

        if self.output_path is None:
            plt.show()
        else:
            plt.savefig(str(self.output_path / "histogram.png"))

    def print(self):
        self.compute()


def compute_img_stats(
    img_root_path: Path, stats_todo: dict, recursive: bool = False
) -> dict:

    if img_root_path.is_file():
        img_paths = [img_root_path]
    elif recursive:
        img_paths = [i for i in img_root_path.rglob("*") if i.is_file()]
    else:
        img_paths = [i for i in img_root_path.iterdir() if i.is_file()]

    try:
        for img_path in tqdm(img_paths):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Unable to read {img_path}")
                continue

            img = np.stack((img,) * 4, axis=-1) if img.ndim == 2 else img
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA) if img.shape[-1] == 3 else img

            for s in stats_todo.values():
                s.update(img)
    except KeyboardInterrupt:
        pass

    return stats_todo


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Calculate and plot some
        stats about an image or all the images on a directory"""
    )

    parser.add_argument(
        "input_path", type=Path, help="Path to the images folder or image file"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Traverse all the folder on the input path",
    )
    parser.add_argument(
        "--mean", action="store_true", help="Computes the per channel mean"
    )
    parser.add_argument("--std", action="store_true", help="Computes the std")
    parser.add_argument(
        "--histogram", action="store_true", help="Computes the histogram of the images"
    )
    parser.add_argument(
        "--size",
        action="store_true",
        help="Computes the min, max and mean size of the images",
    )
    parser.add_argument(
        "--size-plot",
        action="store_true",
        help="Plots an histogram and the shape of the images",
    )
    parser.add_argument(
        "--output-plots",
        type=Path,
        default=None,
        help="Path to save the plots. If not set the plots will only be shown.",
    )

    args = parser.parse_args()

    stats = {}

    if args.mean or args.std:
        stats["Mean"] = Mean()
    if args.histogram:
        stats["Histogram"] = Histogram(args.output_plots)
    if args.size or args.size_plot:
        stats["Sizes"] = Sizes(args.size_plot, args.output_plots)

    compute_img_stats(args.input_path, stats, args.recursive)
    for s in stats.values():
        s.print()

    # STD needs to know the mean
    if args.std:
        std = STD(stats["Mean"].compute())
        std_res = compute_img_stats(args.input_path, {"STD": std}, args.recursive)
        std_res["STD"].print()
