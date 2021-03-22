import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import faiss
from natsort import natsorted

import argparse
import os
from pprint import pformat
import logging
import time
from multiprocessing import cpu_count
import sys
from typing import List, Dict, Any, Optional, Tuple, Callable

from src.models import Resnet50


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)


plt.rcParams['figure.figsize'] = (30, 30)
plt.rcParams['figure.dpi'] = 150


ALLOWED_EXTENSIONS: Tuple[str] = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, transform: Callable, labeled_folders: bool = False):
        self.images_dir: str = images_dir
        self.transform = transform
        self.labeled_folders: bool = labeled_folders
        self.samples: List[Tuple[str, Optional[str]]] = self.__get_all_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, Optional[str]]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label

    def __get_all_samples(self) -> List[Tuple[str, Optional[str]]]:
        samples: List[Tuple[str, Optional[str]]] = []
        for root, _, image_names in natsorted(os.walk(self.images_dir)):
            for image_name in natsorted(image_names):
                if Dataset.has_allowed_extension(image_name):
                    image_path: str = os.path.join(root, image_name)
                    label: str = ""
                    if self.labeled_folders:
                        label = os.path.basename(root)
                    samples.append((image_path, label))
        return samples

    @classmethod
    def has_allowed_extension(cls, image_name: str) -> bool:
        return image_name.lower().endswith(ALLOWED_EXTENSIONS)


def get_embedding(model: torch.nn.Module,
                  image_path: str,
                  transform: Callable,
                  device: torch.device
                  ) -> np.ndarray:
    image: Image.Image = Image.open(image_path).convert("RGB")
    input_tensor: torch.Tensor = transform(image).unsqueeze(dim=0).to(device)
    embedding: torch.Tensor = model(input_tensor)
    return embedding.detach().cpu().numpy()


@torch.no_grad()
def get_embeddings_from_dataloader(loader: DataLoader,
                                   model: torch.nn.Module,
                                   device: torch.device,
                                   ) -> Tuple[np.ndarray, Optional[List[str]], List[str]]:
    model.eval()

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[str] = []
    for images_, labels_ in loader:
        images: torch.Tensor = images_.to(device, non_blocking=True)
        embeddings: torch.Tensor = model(images)
        embeddings_ls.append(embeddings)
        labels_ls.extend(labels_)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]

    images_paths: List[str] = []
    for image_path, _ in loader.dataset.samples:
        images_paths.append(image_path)
    return (embeddings.cpu().numpy(), labels_ls, images_paths)


def query(index: faiss.IndexFlatL2,
          query_embedding: np.ndarray,
          k_queries: int,
          ref_image_paths: List[str],
          ) -> Tuple[List[str], List[int], List[float]]:

    # Searching using nearest neighbors in reference set
    # indices: shape [N_embeddings x k_queries]
    # distances: shape [N_embeddings x k_queries]
    distances, indices = index.search(query_embedding, k_queries)
    indices: List[int] = indices.ravel().tolist()
    distances: List[float] = distances.ravel().tolist()
    image_paths: List[str] = [ref_image_paths[i] for i in indices]
    return image_paths, indices, distances


def visualize_query(query_image_path: str,
                    retrieved_image_paths: List[str],
                    retrieved_distances: List[float],
                    retrieved_labels: Optional[List[str]] = None,
                    query_name: Optional[str] = "",
                    image_size=(224, 224)):

    n_retrieved_images: int = len(retrieved_image_paths)
    nrows: int = 2 + (n_retrieved_images - 1) // 3

    _, axs = plt.subplots(nrows=nrows, ncols=3)

    # Plot query image
    query_image: np.ndarray = cv2.imread(query_image_path, cv2.IMREAD_COLOR)
    query_image = cv2.resize(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB), image_size)
    axs[0, 1].imshow(query_image)
    title: str = "Query"
    if query_name:
        title: str = f"Query: {query_name}"
    axs[0, 1].set_title(title, fontsize=30)

    # Plot retrieved images
    for i in range(n_retrieved_images):
        row: int = i // 3 + 1
        col: int = i % 3

        retrieved_image: np.ndarray = cv2.imread(retrieved_image_paths[i], cv2.IMREAD_COLOR)
        retrieved_image = cv2.resize(cv2.cvtColor(retrieved_image, cv2.COLOR_BGR2RGB), image_size)
        distance: float = round(retrieved_distances[i], 4)
        axs[row, col].imshow(retrieved_image)

        title: str = f"Top {i + 1}\nDistance: {distance}"
        if retrieved_labels:
            label: str = retrieved_labels[i]
            title = f"Top {i + 1}: {label}\nDistance: {distance}"
        axs[row, col].set_title(title, fontsize=30)

    # Turn off axis for all plots
    for ax in axs.ravel():
        ax.axis("off")

    return axs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

    parser.add_argument(
        "-i", "--image_path",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "-r", "--reference_images_dir",
        type=str,
        required=True,
        help="Directory to reference images"
    )
    parser.add_argument(
        "-c", "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-k", "--k_queries",
        type=int,
        default=3,
        help="Number of queries"
    )
    parser.add_argument(
        "--labeled_folders",
        action="store_true",
        default=False,
        help="Whether each folder is a label or not. Default False"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
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

    # Initialize reference set and reference loader
    reference_set = Dataset(args["reference_images_dir"], transform=transform, labeled_folders=args["labeled_folders"])
    reference_loader = DataLoader(reference_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    logging.info(f"Initialized reference loader: {reference_loader.dataset}")

    # Calculate embeddings from images in reference set
    start = time.time()
    ref_embeddings, ref_labels, ref_image_paths = get_embeddings_from_dataloader(
        reference_loader, model, device
    )
    end = time.time()
    logging.info(f"Calculated {len(ref_embeddings)} embeddings in reference set: {end - start} second")

    # Indexing database to search
    index = faiss.IndexFlatL2(config["embedding_size"])
    start = time.time()
    index.add(ref_embeddings)
    end = time.time()
    logging.info(f"Indexed {index.ntotal} embeddings in reference set: {end - start} seconds")

    # Retrive k most similar images in reference set
    start = time.time()
    embedding: np.ndarray = get_embedding(model, args["image_path"], transform, device)
    retrieved_image_paths, retrieved_indices, retrieved_distances = query(
        index, embedding, args["k_queries"], ref_image_paths
    )
    if ref_labels:
        retrieved_labels: List[str] = [ref_labels[i] for i in retrieved_indices]
    end = time.time()
    logging.info(f"Done querying {args['k_queries']} queries: {end - start} seconds")
    logging.info(f"Top {args['k_queries']}: {pformat(retrieved_image_paths)}")

    # Visualize
    visualize_query(
        args["image_path"], retrieved_image_paths, retrieved_distances, retrieved_labels
    )
    output_path: str = os.path.join(args["output_dir"], "infer.jpg")
    plt.savefig(output_path, dpi=150)
    logging.info(f"Plot is saved at: {output_path}")
