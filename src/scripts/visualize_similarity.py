import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
import seaborn as sns

import argparse
import os
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


def visualize_similarity(embeddings: np.ndarray,
                         image_paths: List[str],
                         labels: List[int],
                         image_size=(224, 224)
                         ) -> np.ndarray:

    distances_matrix: np.ndarray = cdist(embeddings, embeddings)
    # distances_matrix = np.clip(distances_matrix, 0, 1)
    similarity_grid: np.ndarray = plot_similarity_grid(distances_matrix, image_size)

    n_classes: int = np.unique(labels).max() + 1
    palette: np.ndarray = (np.array(sns.color_palette("hls", n_classes)) * 255).astype(np.int)
    images: List[np.ndarray] = [cv2.imread(path) for path in image_paths]
    images = [cv2.resize(image, image_size) for image in images]
    images = [
        draw_border(image, color=palette[i].tolist())
        for image, i in zip(images, labels)
    ]

    horizontal_grid = np.hstack(images)
    vertical_grid = np.vstack(images)
    zeros = np.zeros((*image_size, 3))
    vertical_grid = np.vstack((zeros, vertical_grid))

    grid = np.vstack((horizontal_grid, similarity_grid))
    grid = np.hstack((vertical_grid, grid))
    return grid.astype(np.uint8)


def plot_similarity_grid(distances_matrix: np.ndarray, image_size: Tuple[int, int]):
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
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=4)
            x: int = (cell.shape[0] - text_width) // 2
            y: int = (cell.shape[1] + text_height) // 2
            cv2.putText(cell, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 0, 255))

            row.append(cell)

        rows.append(np.concatenate(row, axis=1))

    grid: np.ndarray = np.concatenate(rows, axis=0)
    return grid


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
        "-n", "--n_images",
        type=str,
        default=10,
        help="Number of random images to visualize"
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
    model = torch.nn.DataParallel(Resnet50(config["embedding_size"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Initialized model: {model}")

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
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    logging.info(f"Initialized loader: {loader.dataset}")

    # Calculate embeddings from images in reference set
    start = time.time()
    embeddings, labels, image_paths = get_embeddings_from_dataloader(
        loader, model, device, return_numpy_array=True, return_image_paths=True
    )
    end = time.time()
    logging.info(f"Calculated {len(embeddings)} embeddings: {end - start} second")

    # Get n random images to visualize
    indices: List[int] = np.random.permutation(len(image_paths))[:args["n_images"]]
    embeddings: np.ndarray = np.asarray([embeddings[i] for i in indices])
    image_paths: List[str] = [image_paths[i] for i in indices]
    labels: List[int] = [labels[i] for i in indices]

    # Visualize similarity grid
    grid: np.ndarray = visualize_similarity(embeddings, image_paths, labels)
    output_path: str = os.path.join(args["output_dir"], "similarity.jpg")
    cv2.imwrite(output_path, grid)
    logging.info(f"Plot is saved at: {output_path}")
