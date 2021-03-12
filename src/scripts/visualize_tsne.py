import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import seaborn as sns

import argparse
import os
from multiprocessing import cpu_count
import time
from pprint import pformat
import logging
import sys
from typing import Dict, Any, List, Tuple

from src.models import Resnet50
from src.dataset import Dataset
from src.utils import get_embeddings_from_dataloader, set_random_seed


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

plt.rcParams['figure.figsize'] = (32, 32)
plt.rcParams['figure.dpi'] = 150


def scale_to_01_range(x: np.ndarray) -> float:
    """
    Scale and move the coordinates so they fit [0; 1] range
    """
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def resize_image(image: np.ndarray, max_image_size: int) -> np.ndarray:
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_border(image: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    height, width, _ = image.shape
    # get the color corresponding to image class
    image = cv2.rectangle(image, (0, 0), (width - 1, height - 1), color, thickness=6)
    return image


def compute_plot_coordinates(image: np.ndarray,
                             x: int,
                             y: int,
                             image_centers_area_size: int,
                             offset: int
                             ) -> Tuple[int, int, int, int]:

    image_height, image_width, _ = image.shape
    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    xmin = center_x - int(image_width / 2)
    ymin = center_y - int(image_height / 2)
    xmax = xmin + image_width
    ymax = ymin + image_height

    return xmin, ymin, xmax, ymax


def plot_tsne_images(X: np.ndarray,
                     image_paths: List[str],
                     labels: List[int],
                     plot_size=2000,
                     max_image_size=224
                     ) -> np.ndarray:

    n_classes: int = np.unique(labels).max() + 1
    palette: np.ndarray = (np.array(sns.color_palette("hls", n_classes)) * 255).astype(np.int)

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    x_scaled: np.ndarray = X[:, 0]
    y_scaled: np.ndarray = X[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    x_scaled = scale_to_01_range(x_scaled)
    y_scaled = scale_to_01_range(y_scaled)

    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset: int = max_image_size // 2
    image_centers_area_size: int = plot_size - 2 * offset

    plot: np.ndarray = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in zip(image_paths, labels, x_scaled, y_scaled):
        image: np.ndarray = cv2.imread(image_path)
        # scale the image to put it to the plot
        image = resize_image(image, max_image_size)
        # draw a rectangle with a color corresponding to the image class
        color = palette[label].tolist()
        image = draw_border(image, color)
        # compute the coordinates of the image on the scaled plot visualization
        xmin, ymin, xmax, ymax = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)
        # put the image to its TSNE coordinates using numpy subarray indices
        plot[ymin:ymax, xmin:xmax, :] = image

    return plot


def plot_scatter(X: np.ndarray, y: np.ndarray, idx_to_class: Dict[int, str]):
    """
    Render a scatter plot with as many as unique colors as the number of classes.
    Source: https://www.datacamp.com/community/tutorials/introduction-t-sne

    Args:
        X: 2-D array output of t-sne algorithm
        y: 1-D array containing the labels of the dataset
        idx_to_class: dictionary mapping from class's index to class's name
    Return:
        Tuple
    """
    # Choose a color palette with seaborn
    n_classes: int = len(idx_to_class)
    palette: np.ndarray = np.array(sns.color_palette("hls", n_classes))

    # Create a scatter plot
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    scatter = ax.scatter(
        X[:, 0], X[:, 1], lw=0, s=40, c=palette[y.astype(np.int)]
    )
    ax.axis("tight")
    ax.axis("off")

    # Add the labels for each sample corresponding to the label
    for i in range(n_classes):
        # Position of each label at median of data points
        x_text, y_text = np.median(X[y == i, :], axis=0)
        text = ax.text(x_text, y_text, str(i), fontsize=20)
        text.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])

    # Add legends for each class
    annotations: List = []
    for i in range(n_classes):
        circle = Line2D([0], [0], marker='o', color=palette[i], label=f"{i}: {idx_to_class[i]}")
        annotations.append(circle)
    plt.legend(handles=annotations, loc="best")

    return figure, ax, scatter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

    parser.add_argument(
        "-i", "--images_dir",
        type=str,
        required=True,
        help="Directory to images"
    )
    parser.add_argument(
        "-c", "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "-n", "--n_images",
        type=int,
        default=300,
        help="Number of random images to visualize"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=12345,
        help="Random seed"
    )
    args: Dict[str, Any] = vars(parser.parse_args())

    set_random_seed(args["random_seed"])

    # Create output directory if not exists
    if not os.path.isdir(args["output_dir"]):
        os.makedirs(args["output_dir"])
        logging.info(f"Created output directory {args['output_dir']}")

    # Initialize device
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initialized device {device}")

    # Load model's checkpoint
    checkpoint_path: str = args["checkpoint_path"]
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path']}")

    # Load config
    config: Dict[str, Any] = checkpoint["config"]
    logging.info(f"Loaded config: {pformat(config)}")

    # Intialize model
    model = Resnet50(config["embedding_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    logging.info(f"Initialized model: {model}")

    # Initialize batch size and n_workers
    batch_size: int = config.get("batch_size", None)
    if not batch_size:
        batch_size = config["classes_per_batch"] * config["samples_per_class"]
    n_cpus: int = cpu_count()
    if n_cpus >= batch_size:
        n_workers: int = batch_size
    else:
        n_workers: int = n_cpus
    logging.info(f"Found {n_cpus} cpus. "
                 f"Use {n_workers} threads for data loading "
                 f"with batch_size={batch_size}")

    # Initialize transform
    transform = T.Compose([
        T.Resize((config["image_size"], config["image_size"])),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    logging.info(f"Initialized transforms: {transform}")

    # Initialize dataset and dataloader
    dataset = Dataset(args["images_dir"], transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    logging.info(f"Initialized loader: {loader.dataset}")

    # Calculate embeddings from images in reference set
    start = time.time()
    embeddings, labels, image_paths = get_embeddings_from_dataloader(
        loader, model, device, return_numpy_array=True, return_image_paths=True
    )
    end = time.time()
    logging.info(f"Calculated {len(embeddings)} embeddings: {end - start} second")

    # Init T-SNE
    tsne = TSNE(n_components=2, perplexity=35, learning_rate=200, n_iter=2000, n_jobs=-1)
    X_transformed = tsne.fit_transform(embeddings)

    # Visualize T-SNE in points
    idx_to_class: Dict[int, str] = {
        idx: class_name.split("-")[-1]
        for idx, class_name in dataset.idx_to_class.items()
    }
    plot_scatter(X_transformed, labels, idx_to_class)
    tnse_points_path: str = os.path.join(args["output_dir"], "tsne_points.jpg")
    plt.savefig(tnse_points_path, dpi=150)
    logging.info(f"Plot is saved at: {tnse_points_path}")

    if args["n_images"] > 0:
        # Shuffle all images and get random n_images to visualize T-SNE
        indices: List[int] = np.random.permutation(len(image_paths))[:args["n_images"]]
        embeddings: np.ndarray = np.asarray([embeddings[i] for i in indices])
        labels = [labels[i] for i in indices]
        image_paths = [image_paths[i] for i in indices]
        X_transformed = np.asarray([X_transformed[i] for i in indices])

    # Visualize T-SNE in images
    plot: np.ndarray = plot_tsne_images(X_transformed, image_paths, labels, plot_size=5000)
    tsne_images_path: str = os.path.join(args["output_dir"], "tsne_images.jpg")
    cv2.imwrite(tsne_images_path, plot)
    logging.info(f"Plot is saved at: {tsne_images_path}")
