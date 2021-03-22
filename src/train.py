import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import RAdam

import argparse
import logging
import json
import time
import sys
import os
import yaml
from pprint import pformat
from typing import Dict, Any

from src.trainer import train_one_epoch
from src.models import Resnet50
from src.losses import TripletMarginLoss, ProxyNCALoss, ProxyAnchorLoss, SoftTripleLoss
from src.samplers import PKSampler
from src.dataset import Dataset, get_subset_from_dataset
from src.utils import set_random_seed, get_current_time, log_embeddings_to_tensorboard


CURRENT_TIME: str = get_current_time()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"src/logs/{CURRENT_TIME}.txt", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def main(args: Dict[str, Any]):
    start = time.time()

    # Intialize config
    config_path: str = args["config"]
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    logger.info(f"Loaded config at: {config_path}")
    logger.info(f"{pformat(config)}")


    # Initialize device
    if args["use_gpu"] and torch.cuda.is_available():
        device: torch.device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    # Intialize model
    model = nn.DataParallel(Resnet50(
        embedding_size=config["embedding_size"],
        pretrained=config["pretrained"]
    ))
    model = model.to(device)
    logger.info(f"Initialized model: {model}")


    # Initialize optimizer
    optimizer = RAdam(model.parameters(), lr=config["lr"])
    logger.info(f"Initialized optimizer: {optimizer}")


    # Initialize train transforms
    transform_train = T.Compose([
        T.Resize((config["image_size"], config["image_size"])),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=5, scale=(0.8, 1.2), translate=(0.2, 0.2)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    logger.info(f"Initialized training transforms: {transform_train}")


    # Initialize training set
    train_set = Dataset(args["train_dir"], transform=transform_train)

    if args["loss"] == "tripletloss":
        # Initialize train loader for triplet loss
        batch_size: int = config["classes_per_batch"] * config["samples_per_class"]
        train_loader = DataLoader(
            train_set,
            batch_size,
            sampler=PKSampler(
                train_set.targets,
                config["classes_per_batch"],
                config["samples_per_class"]
            ),
            shuffle=False,
            num_workers=args["n_workers"],
            pin_memory=True,
        )
        logger.info(f"Initialized train_loader: {train_loader.dataset}")

        # Intialize loss function
        loss_function = TripletMarginLoss(
            margin=config["margin"],
            sampling_type=config["sampling_type"]
        )
        logger.info(f"Initialized training loss: {loss_function}")

    elif args["loss"] == "proxy_nca":
        # Initialize train loader for proxy-nca loss
        batch_size: int = config["batch_size"]
        train_loader = DataLoader(
            train_set,
            config["batch_size"],
            shuffle=True,
            num_workers=args["n_workers"],
            pin_memory=True,
        )
        logger.info(f"Initialized train_loader: {train_loader.dataset}")

        loss_function = ProxyNCALoss(
            n_classes=len(train_set.classes),
            embedding_size=config["embedding_size"],
            embedding_scale=config["embedding_scale"],
            proxy_scale=config["proxy_scale"],
            smoothing_factor=config["smoothing_factor"],
            device=device
        )

    elif args["loss"] == "proxy_anchor":
        # Intialize train loader for proxy-anchor loss
        batch_size: int = config["batch_size"]
        train_loader = DataLoader(
            train_set,
            config["batch_size"],
            shuffle=True,
            num_workers=args["n_workers"],
            pin_memory=True,
        )
        logger.info(f"Initialized train_loader: {train_loader.dataset}")

        loss_function = ProxyAnchorLoss(
            n_classes=len(train_set.classes),
            embedding_size=config["embedding_size"],
            margin=config["margin"],
            alpha=config["alpha"],
            device=device
        )

    elif args["loss"] == "soft_triple":
        # Intialize train loader for proxy-anchor loss
        batch_size: int = config["batch_size"]
        train_loader = DataLoader(
            train_set,
            config["batch_size"],
            shuffle=True,
            num_workers=args["n_workers"],
            pin_memory=True,
        )
        logger.info(f"Initialized train_loader: {train_loader.dataset}")

        loss_function = SoftTripleLoss(
            n_classes=len(train_set.classes),
            embedding_size=config["embedding_size"],
            n_centers_per_class=config["n_centers_per_class"],
            lambda_=config["lambda"],
            gamma=config["gamma"],
            tau=config["tau"],
            margin=config["margin"],
            device=device
        )
    else:
        raise Exception("Only the following losses is supported: "
                        "['tripletloss', 'proxy_nca', 'proxy_anchor', 'soft_triple']. "
                        f"Got {args['loss']}")


    # Initialize test transforms
    transform_test = T.Compose([
        T.Resize((config["image_size"], config["image_size"])),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    logger.info(f"Initialized test transforms: {transform_test}")


    # Initialize test set and test loader
    test_dataset = Dataset(args["test_dir"], transform=transform_test)
    test_loader = DataLoader(
        test_dataset, batch_size,
        shuffle=False,
        num_workers=args["n_workers"],
    )
    logger.info(f"Initialized test_loader: {test_loader.dataset}")


    # Initialize reference set and reference loader
    # If reference set is not given, use train set as reference set, but without random sampling
    if not args["reference_dir"]:
        reference_set = Dataset(args["train_dir"], transform=transform_test)
    else:
        reference_set = Dataset(args["reference_dir"], transform=transform_test)
    # Sometimes reference set is too large to fit into memory,
    # therefore we only sample a subset of it.
    n_samples_per_reference_class: int = args["n_samples_per_reference_class"]
    if n_samples_per_reference_class > 0:
        reference_set = get_subset_from_dataset(reference_set, n_samples_per_reference_class)

    reference_loader = DataLoader(
        reference_set, batch_size,
        shuffle=False,
        num_workers=args["n_workers"],
    )
    logger.info(f"Initialized reference set: {reference_loader.dataset}")


    # Initialize checkpointing directory
    checkpoint_dir: str = os.path.join(args["checkpoint_root_dir"], CURRENT_TIME)
    writer = SummaryWriter(log_dir=checkpoint_dir)
    logger.info(f"Created checkpoint directory at: {checkpoint_dir}")


    # Dictionary contains all metrics
    output_dict: Dict[str, Any] = {
        "total_epoch": args["n_epochs"],
        "current_epoch": 0,
        "current_iter": 0,
        "metrics": {
            "mean_average_precision": 0.0,
            "average_precision_at_1": 0.0,
            "average_precision_at_5": 0.0,
            "average_precision_at_10": 0.0,
            "top_1_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "normalized_mutual_information": 0.0,
        }
    }
    # Start training and testing
    logger.info("Start training...")
    for _ in range(1, args["n_epochs"] + 1):
        output_dict = train_one_epoch(
            model, optimizer, loss_function,
            train_loader, test_loader, reference_loader,
            writer, device, config,
            checkpoint_dir,
            args['log_frequency'],
            args['validate_frequency'],
            output_dict
        )
    logger.info(f"DONE TRAINING {args['n_epochs']} epochs")


    # Visualize embeddings
    logger.info("Calculating train embeddings for visualization...")
    log_embeddings_to_tensorboard(train_loader, model, device, writer, tag="train")
    logger.info("Calculating reference embeddings for visualization...")
    log_embeddings_to_tensorboard(reference_loader, model, device, writer, tag="reference")
    logger.info("Calculating test embeddings for visualization...")
    log_embeddings_to_tensorboard(test_loader, model, device, writer, tag="test")


    # Visualize model's graph
    logger.info("Adding graph for visualization")
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, config["image_size"], config["image_size"]).to(device)
        writer.add_graph(model.module.features, dummy_input)


    # Save all hyper-parameters and corresponding metrics
    logger.info("Saving all hyper-parameters")
    writer.add_hparams(
        config,
        metric_dict={f"hyperparams/{key}": value for key, value in output_dict["metrics"].items()}
    )
    with open(os.path.join(checkpoint_dir, "output_dict.json"), "w") as f:
        json.dump(output_dict, f, indent=4)
    logger.info(f"Dumped output_dict.json at {checkpoint_dir}")


    end = time.time()
    logger.info(f"EVERYTHING IS DONE. Training time: {round(end - start, 2)} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--train_dir", type=str, required=True, help="Directory to training images"
    )
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Directory to test images"
    )
    parser.add_argument(
        "--reference_dir", type=str, default=None, help="Directory to reference images"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to yaml config file"
    )
    parser.add_argument(
        "--loss", type=str, required=True, help="Which loss to use: ['tripletloss', 'proxy_nca', 'proxy_anchor']"
    )
    parser.add_argument(
        "--n_samples_per_reference_class",
        type=int,
        default=-1,
        help="Number of samples per class in reference set to use. Default: use all samples"
    )
    parser.add_argument(
        "--checkpoint_root_dir", type=str, default="src/checkpoints", help="Directory to save model weight"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--n_workers", type=int, default=8, help="Number of threads used for data loading"
    )
    parser.add_argument(
        "--log_frequency", type=int, default=100, help="Number of iterations to print training logs"
    )
    parser.add_argument(
        "--validate_frequency", type=int, default=1000, help="Number of iterations to run test"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="Whether to use gpu for training"
    )
    parser.add_argument(
        "--random_seed", type=int, default=12345, help="Random seed"
    )

    args: Dict[str, Any] = vars(parser.parse_args())
    set_random_seed(args["random_seed"])

    main(args)
