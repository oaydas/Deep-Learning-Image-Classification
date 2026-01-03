# Use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import resize, ImageStandardizer, DogsDataset
from utils import set_random_seed


def main():
    set_random_seed()
    
    training_set = DogsDataset("train")
    training_set.X = resize(training_set.X)

    positive_idx = np.where(training_set.y == 1)[0]
    negative_idx = np.where(training_set.y == 0)[0]

    standardizer = ImageStandardizer()
    standardizer.fit(training_set.X)

    N = 5

    fig, axes = plt.subplots(nrows=2, ncols=N, figsize=(2 * N, 4))

    pad = 3
    axes[0, 0].annotate(
        "Positive",
        xy=(0, 0.5),
        xytext=(-axes[0, 0].yaxis.labelpad - pad, 0),
        xycoords=axes[0, 0].yaxis.label,
        textcoords="offset points",
        size="large",
        ha="right",
        va="center",
        rotation="vertical",
    )
    axes[1, 0].annotate(
        "Negative",
        xy=(0, 0.5),
        xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
        xycoords=axes[1, 0].yaxis.label,
        textcoords="offset points",
        size="large",
        ha="right",
        va="center",
        rotation="vertical",
    )

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    rand_pos_idx = np.random.choice(positive_idx, size=N, replace=False)
    rand_neg_idx = np.random.choice(negative_idx, size=N, replace=False)

    for i, idx in enumerate(rand_pos_idx):
        axes[0, i].imshow(training_set.X[idx])

    for i, idx in enumerate(rand_neg_idx):
        axes[1, i].imshow(training_set.X[idx])

    plt.tight_layout()
    plt.savefig("visualize_labels.png")
    print("Saved figure to visualize_labels.png.")


if __name__ == "__main__":
    main()

