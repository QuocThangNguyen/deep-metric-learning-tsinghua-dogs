import enum
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
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
from src.utils import get_embeddings_from_dataloader
from src.scripts.visualize_tsne import draw_border


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def _plot_similarity_grid(distances_matrix: np.ndarray, image_size: Tuple[int, int]):
    n_images: int = len(distances_matrix)
    max_distance: float = np.max(distances_matrix) * 2

    rows: List[np.ndarray] = []
    for i in range(n_images):
        row: List[np.ndarray] = []
        for j in range(n_images):
            distance: float = distances_matrix[i][j]
            # As distances may larger than 1, so we scale them to range [0-1] to apply color map
            scaled_distance: float = distance / max_distance
            cell: np.ndarray = np.empty(image_size)
            cell.fill(scaled_distance)
            cell = (cell * 255).astype(np.uint8)

            # Color depend on value: blue is closer to 0, the larger the greeener
            cell: np.ndarray = cv2.applyColorMap(cell, cv2.COLORMAP_JET)

            # Add distance value as text centered on image
            text: str = f"{distance:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=4
            )
            x: int = (cell.shape[0] - text_width) // 2
            y: int = (cell.shape[1] + text_height) // 2
            cv2.putText(cell, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 0, 255))

            row.append(cell)

        rows.append(np.concatenate(row, axis=1))

    grid: np.ndarray = np.concatenate(rows, axis=0)
    return grid


def _get_random_images(embeddings: np.ndarray,
                       labels: np.ndarray,
                       image_paths: List[str],
                       n_classes_to_visualize=2,
                       n_images_per_class=5
                       ) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    total_classes: np.ndarray = np.unique(labels)
    chosen_classes: np.ndarray = np.random.choice(total_classes, size=n_classes_to_visualize, replace=False)

    # Indices of chosen images
    indices: np.ndarray = []
    for i in chosen_classes:
        class_indices: np.ndarray = np.where(labels == i)[0]
        class_indices = np.random.permutation(class_indices)[:n_images_per_class]
        indices.append(class_indices)
    indices = np.hstack(indices)

    indices = np.random.permutation(indices)
    embeddings: np.ndarray = np.asarray([embeddings[i] for i in indices])
    labels: List[int] = [labels[i] for i in indices]
    image_paths: List[str] = [image_paths[i] for i in indices]

    return embeddings, labels, image_paths


def visualize_similarity(embeddings: np.ndarray,
                         labels: List[int],
                         image_paths: List[str],
                         image_size=(224, 224)
                         ) -> np.ndarray:

    distances_matrix: np.ndarray = cdist(embeddings, embeddings)
    similarity_grid: np.ndarray = _plot_similarity_grid(distances_matrix, image_size)

    unique_labels: np.ndarray = np.unique(labels)
    n_classes: int = len(unique_labels)
    palette: np.ndarray = (np.array(sns.color_palette("hls", n_classes)) * 255).astype(np.int)
    class_to_color: Dict[int, Tuple] = {
        class_: palette[i].tolist()
        for i, class_ in enumerate(unique_labels)
    }

    images: List[np.ndarray] = [cv2.imread(path) for path in image_paths]
    images = [cv2.resize(image, image_size) for image in images]
    images = [
        draw_border(image, color=class_to_color[i])
        for image, i in zip(images, labels)
    ]

    horizontal_grid = np.hstack(images)
    vertical_grid = np.vstack(images)
    zeros = np.zeros((*image_size, 3))
    vertical_grid = np.vstack((zeros, vertical_grid))

    grid = np.vstack((horizontal_grid, similarity_grid))
    grid = np.hstack((vertical_grid, grid))
    return grid.astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing dogs similarity in a grid")

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
        "--n_classes",
        type=str,
        default=2,
        help="Number of random classes to visualize"
    )
    parser.add_argument(
        "--n_images_per_class",
        type=str,
        default=5,
        help="Number of images per class to visualize"
    )
    args: Dict[str, Any] = vars(parser.parse_args())

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
    if not batch_size:  # For triplet loss: batch_size = classes_per_batch x samples_per_class
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

    # Get random images to visualize
    embeddings, labels, image_paths = _get_random_images(
        embeddings, labels, image_paths, args["n_classes"], args["n_images_per_class"]
    )

    # Visualize similarity grid
    grid: np.ndarray = visualize_similarity(embeddings, labels, image_paths)
    output_path: str = os.path.join(args["output_dir"], "similarity.jpg")
    cv2.imwrite(output_path, grid)
    logging.info(f"Plot is saved at: {output_path}")
