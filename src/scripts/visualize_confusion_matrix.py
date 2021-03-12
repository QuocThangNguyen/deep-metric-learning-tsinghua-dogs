import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import faiss
from sklearn.metrics._classification import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

import argparse
import os
from pprint import pformat
import logging
import time
from multiprocessing import cpu_count
import sys
from typing import List, Dict, Any

from src.models import Resnet50
from src.dataset import Dataset
from src.utils import get_embeddings_from_dataloader


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)


plt.rcParams['figure.figsize'] = (30, 30)
plt.rcParams['figure.dpi'] = 150


def plot_confusion_matrix(y_pred: np.ndarray,
                          y_true: np.ndarray,
                          display_labels: List[str],
                          title: str,
                          normalize=None,
                          fontsize=30
                          ) -> ConfusionMatrixDisplay:

    """Plot Confusion Matrix.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    y_true : array-like of shape (n_samples,)
        Target values.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    display_labels : array-like of shape (n_classes,)
        Target names used for plotting.
        Rotation of xtick labels.

    Returns
    -------
    display: `sklearn.metrics.ConfusionMatrixDisplay`
    """

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)

    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=None, xticks_rotation="vertical",
                     values_format=None)

    disp.ax_.set_title(title, fontsize=fontsize)
    return disp


def calculate_precision_at_k(index: faiss.IndexFlatL2,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             y_train: np.ndarray,
                             k: int
                             ) -> float:
    # Searching using nearest neighbors in train set
    _, indices = index.search(X_test, k)  # indices: shape [N_embeddings x k_neighbors]

    y_pred = []
    for i in range(k):
        indices_at_k: np.ndarray = indices[:, i]  # [M]
        y_pred_at_k: np.ndarray = y_train[indices_at_k][:, None]  # [M x 1]
        y_pred.append(y_pred_at_k)

    y_pred: np.ndarray = np.hstack(y_pred)  # [M x k]
    y_test = np.hstack((y_test[:, None],) * k)  # [M x k]

    precision_at_k: float = ((y_pred == y_test).sum(axis=1) / k).mean().item() * 100
    return precision_at_k


def calculate_topk_accuracy(index: faiss.IndexFlatL2,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            y_train: np.ndarray,
                            top_k: int
                            ) -> float:

    # Search using nearest neighbors in train set
    _, indices = index.search(X_test, top_k)  # indices: shape [N_embeddings x k_neighbors]

    y_pred = []
    for i in range(top_k):
        indices_at_k: np.ndarray = indices[:, i]  # [M]
        y_pred_at_k: np.ndarray = y_train[indices_at_k][:, None]  # [M x 1]
        y_pred.append(y_pred_at_k)

    y_pred: np.ndarray = np.hstack(y_pred)  # [M x k]
    y_test = np.hstack((y_test[:, None],) * top_k)  # [M x k]

    n_predictions: int = y_pred.shape[0]
    n_true_predictions: int = ((y_pred == y_test).sum(axis=1) > 0).sum().item()
    topk_accuracy: float = n_true_predictions / n_predictions * 100
    return topk_accuracy


def calculate_all_metrics(index: faiss.IndexFlatL2,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          y_train: np.ndarray,
                          k=(1, 5)
                          ) -> Dict[str, float]:

    metrics: Dict[str, float] = {}
    # Calculate top-1 and top-5 accuracy
    for i in k:
        metrics[f"Top-{i} accuracy"] = calculate_topk_accuracy(index, X_test, y_test, y_train, top_k=i)
    return metrics


def predict(index: faiss.IndexFlatL2,
            embeddings: np.ndarray,
            y_train: np.ndarray,
            k_neighbors: int
            ) -> np.ndarray:

    # Searching using nearest neighbors in reference set
    _, indices = index.search(embeddings, k_neighbors)  # indices: shape [N_embeddings x k_neighbors]
    # Get labels of found indices
    y_pred: np.ndarray = y_train[indices]  # shape: [N_embeddings x k_neighbors]
    # Labels that appears most are predicted labels
    # https://en.wikipedia.org/wiki/Mode_(statistics)
    y_pred = mode(y_pred, axis=1)[0].ravel()  # shape: [N_embedding]
    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

    parser.add_argument(
        "-t", "--test_images_dir",
        type=str,
        required=True,
        help="Directory to test images"
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
        "-k", "--k_neighbors",
        type=int,
        default=1,
        help="Number of neighbors to use for kneighbors queries"
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
    reference_set = Dataset(args["reference_images_dir"], transform=transform)
    reference_loader = DataLoader(reference_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    logging.info(f"Initialized reference loader: {reference_loader.dataset}")

    # Calculate embeddings from images in reference set
    start = time.time()
    ref_embeddings, ref_labels = get_embeddings_from_dataloader(
        reference_loader, model, device, return_numpy_array=True
    )
    end = time.time()
    logging.info(f"Calculated {len(ref_embeddings)} embeddings in reference set: {end - start} second")

    # Indexing database to search
    index = faiss.IndexFlatL2(config["embedding_size"])
    start = time.time()
    index.add(ref_embeddings)
    end = time.time()
    logging.info(f"Indexed {index.ntotal} embeddings in reference set: {end - start} seconds")

    # Initialize test set and test loader
    test_set = Dataset(args["test_images_dir"], transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    logging.info(f"Initialized test loader: {test_loader.dataset}")

    # Calculate embeddings from images in reference set
    start = time.time()
    test_embeddings, test_labels = get_embeddings_from_dataloader(
        test_loader, model, device, return_numpy_array=True
    )
    end = time.time()
    logging.info(f"Calculated {len(test_embeddings)} embeddings in test set: {end - start} second")

    # Get predictions
    start = time.time()
    predicted_labels: np.ndarray = predict(
        index,
        test_embeddings,
        y_train=ref_labels,
        k_neighbors=args["k_neighbors"]
    )
    end = time.time()
    logging.info(f"Done predicting for {len(test_embeddings)} queries: {end - start} seconds")

    # Calculate metrics
    metrics: Dict = calculate_all_metrics(index, X_test=test_embeddings, y_test=test_labels, y_train=ref_labels)
    logging.info(f"Done calculating {len(metrics)} metrics: {pformat(metrics.items())}")

    # Plot confusion matrix
    title: str = f"{len(test_embeddings)} images ({len(test_set.class_to_idx)} classes)\n\n"
    for metric_name, value in metrics.items():
        title += f"{metric_name}: {value:.2f}%\n"

    class_names: List[str] = [
        test_set.idx_to_class[i].split("-")[-1]
        for i in range(len(test_set.idx_to_class))
    ]

    plot_confusion_matrix(
        y_pred=predicted_labels,
        y_true=test_labels,
        display_labels=class_names,
        title=title
    )
    output_path: str = os.path.join(args["output_dir"], "confusion_matrix.jpg")
    plt.savefig(output_path, dpi=150)
    logging.info(f"Plot is saved at: {output_path}")
