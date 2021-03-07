import requests
from multiprocessing.pool import ThreadPool
import subprocess

import argparse
import os
import shutil
import logging
import sys
import glob
from pathlib import Path
from typing import Dict, Any, Set


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)


# Homepage: https://cg.cs.tsinghua.edu.cn/ThuDogs/
IMAGES_URLS: Dict[str, str] = {
    "TsinghuaDogs.zip.001": "https://cloud.tsinghua.edu.cn/f/d2031efb239c4dde9c6c/?dl=1",
    "TsinghuaDogs.zip.002": "https://cloud.tsinghua.edu.cn/f/6a242a6bba664537ba45/?dl=1",
    "TsinghuaDogs.zip.003": "https://cloud.tsinghua.edu.cn/f/d17034fa14f54e4381d8/?dl=1",
    "TsinghuaDogs.zip.004": "https://cloud.tsinghua.edu.cn/f/3740fc44cd484e1cb089/?dl=1",
    "TsinghuaDogs.zip.005": "https://cloud.tsinghua.edu.cn/f/ff5d96a0bc4e4dba9004/?dl=1",
    "TsinghuaDogs.zip.006": "https://cloud.tsinghua.edu.cn/f/d5fe5c88198c4387a7bb/?dl=1",
    "TsinghuaDogs.zip.007": "https://cloud.tsinghua.edu.cn/f/b13d6710ac85487e9487/?dl=1",
    "TsinghuaDogs.zip.008": "https://cloud.tsinghua.edu.cn/f/b6cf354fd04b4fe0b909/?dl=1",
    "TsinghuaDogs.zip.009": "https://cloud.tsinghua.edu.cn/f/06a421a528044b15838c/?dl=1"
}

ANNOTATIONS_URL: str = "https://cg.cs.tsinghua.edu.cn/ThuDogs/high-annotations.zip"
TRAIN_VAL_SPLIT_URL: str = "https://cg.cs.tsinghua.edu.cn/ThuDogs/TrainValSplit.zip"


def download_file(url: str, output_path: str) -> int:
    logging.info(f"Downloading {url} to {output_path}")

    with requests.get(url, stream=True) as response:
        with open(output_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)

    logging.info(f"Downloaded {output_path}")
    return response.status_code


def split_train_val(images_dir: str, output_dir: str, lst_path: str):
    with open(lst_path, "r") as f:
        lst: Set[str] = set([os.path.basename(line.strip()) for line in f.readlines()])

    for image_name in lst:
        image_path: str = glob.glob(os.path.join(images_dir, "*", image_name), recursive=True)[0]
        class_name: str = os.path.basename(os.path.dirname(image_path))

        dest_dir: str = os.path.join(output_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        logging.info(f"Copying {image_path} to {dest_dir}")
        shutil.copy(image_path, dest_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Tsinghua Dogs Dataset")
    parser.add_argument(
        "-o", "--output_dir", type=str, default="data/", help="Directory to download data"
    )
    args: Dict[str, Any] = vars(parser.parse_args())

    DATA_DIR = Path(args["output_dir"])
    TSINGHUA_DOGS_ROOT_DIR: str = str(DATA_DIR / "TsinghuaDogs")

    # Download and unzip images using multithreading
    with ThreadPool(processes=len(IMAGES_URLS)) as pool:
        pool.starmap(
            download_file,
            ((url, str(DATA_DIR / name)) for name, url in IMAGES_URLS.items())
        )

    logging.info(f"Unzipping {DATA_DIR}/TsinghuaDogs.zip to {TSINGHUA_DOGS_ROOT_DIR}")
    unzip_command: str = f"""
        cat {DATA_DIR}/TsinghuaDogs.zip.00* > {DATA_DIR}/TsinghuaDogs.zip
        unzip {DATA_DIR}/TsinghuaDogs.zip -d {TSINGHUA_DOGS_ROOT_DIR}
    """
    subprocess.run(unzip_command, shell=True)

    # Download and unzip annotation files
    annotations_path: str = os.path.join(TSINGHUA_DOGS_ROOT_DIR, "high-annotations.zip")
    download_file(ANNOTATIONS_URL, annotations_path)
    logging.info(f"Unzipping {annotations_path} to {TSINGHUA_DOGS_ROOT_DIR}")
    subprocess.run(f"unzip {annotations_path} -d {TSINGHUA_DOGS_ROOT_DIR}", shell=True)

    # Download and unzip train_val_split
    train_val_split_path: str = os.path.join(TSINGHUA_DOGS_ROOT_DIR, "TrainValSplit.zip")
    download_file(TRAIN_VAL_SPLIT_URL, train_val_split_path)
    logging.info(f"Unzipping {train_val_split_path}")
    subprocess.run(f"unzip {train_val_split_path} -d {TSINGHUA_DOGS_ROOT_DIR}", shell=True)

    # Prepare train set
    images_dir: str = os.path.join(TSINGHUA_DOGS_ROOT_DIR, "high-resolution")
    train_lst_path: str = os.path.join(TSINGHUA_DOGS_ROOT_DIR, "TrainAndValList", "train.lst")
    train_dir: str = os.path.join(TSINGHUA_DOGS_ROOT_DIR, "train")
    os.makedirs(train_dir, exist_ok=True)
    split_train_val(images_dir, output_dir=train_dir, lst_path=train_lst_path)

    # Prepare validation set
    val_lst_path: str = os.path.join(TSINGHUA_DOGS_ROOT_DIR, "TrainAndValList", "validation.lst")
    val_dir: str = os.path.join(TSINGHUA_DOGS_ROOT_DIR, "val")
    os.makedirs(val_dir, exist_ok=True)
    split_train_val(images_dir, output_dir=val_dir, lst_path=val_lst_path)

    # Clean up downloaded zip to free disk space
    subprocess.run(f"rm {DATA_DIR}/TsinghuaDogs.zip*", shell=True)
    logging.info(f"Cleanup {DATA_DIR}/TsinghuaDogs.zip*")

    subprocess.run(f"rm {TSINGHUA_DOGS_ROOT_DIR}/*.zip", shell=True)
    logging.info(f"Cleanup {TSINGHUA_DOGS_ROOT_DIR}/*.zip")
