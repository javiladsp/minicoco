import os
import json
import argparse
import numpy as np
from pathlib import Path
from random import sample
from pycocotools.coco import COCO
from alive_progress import alive_bar
from pybboxes import BoundingBox

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

parser = argparse.ArgumentParser()

parser.add_argument("train_annotation_file", type=str,
                    help="annotations/instances_train2017.json path file.",
                    nargs='?',
                    default="annotations/instances_train2017.json")
parser.add_argument("val_annotation_file", type=str,
                    help="annotations/instances_val2017.json path file.",
                    nargs='?',
                    default="annotations/instances_val2017.json")

parser.add_argument("-t", "--training", type=int,
                    help="number of images in the training set.",
                    default=30)
parser.add_argument("-v", "--validation", type=int,
                    help="number of images in the validation set.",
                    default=10)

parser.add_argument("-cat", "--categories", nargs='+',
                    help="category names.",
                    default=["person", "car", "dog"])

parser.add_argument("-f", "--format", type=str,
                    help="available formats: coco, yolo.",
                    choices=["coco", "yolo"],
                    default="coco")

args = parser.parse_args()

ROOT_PATH = "dataset"
TRAIN_IMAGES_PATH = f"{ROOT_PATH}/train/images"
TRAIN_LABELS_PATH = f"{ROOT_PATH}/train/labels"
VAL_IMAGES_PATH = f"{ROOT_PATH}/val/images"
VAL_LABELS_PATH = f"{ROOT_PATH}/val/labels"

Path(ROOT_PATH).mkdir(parents=True, exist_ok=True)
Path(TRAIN_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
Path(TRAIN_LABELS_PATH).mkdir(parents=True, exist_ok=True)
Path(VAL_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
Path(VAL_LABELS_PATH).mkdir(parents=True, exist_ok=True)


def process_subset(annotation_file, num_images, subset_name):
    coco = COCO(annotation_file)
    catNms = args.categories
    catIds = coco.getCatIds(catNms)
    imgIds = coco.getImgIds(catIds=catIds)

    imgOriginals = coco.loadImgs(imgIds)
    imgShuffled = sample(imgOriginals, len(imgOriginals))

    def myImages(images: list, num: int) -> list:
        return images[:num]

    def cocoJson(images: list) -> dict:
        '''getCatIds return a sorted list of id.
        for the creation of the json file in coco format, the list of ids must be successive 1, 2, 3..
        so we reorganize the ids. In the cocoJson method we modify the values of the category_id parameter.'''

        dictCOCO = {k: coco.getCatIds(k)[0] for k in catNms}
        dictCOCOSorted = dict(sorted(dictCOCO.items(), key=lambda x: x[1]))

        IdCategories = list(range(1, len(catNms)+1))
        categories = dict(zip(list(dictCOCOSorted), IdCategories))

        arrayIds = np.array([k["id"] for k in images])
        annIds = coco.getAnnIds(imgIds=arrayIds, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for k in anns:
            k["category_id"] = catIds.index(k["category_id"])+1
        cats = [{'id': int(value), 'name': key}
                for key, value in categories.items()]
        dataset = {"info": {"description": "my-project-name"}}
        dataset["images"] = images
        dataset["annotations"] = anns
        dataset["categories"] = cats

        return dataset

    def yoloJson(images: list) -> dict:
        """
        This will only work for detections, not for segmentation.
        """
        dictCOCO = {k: coco.getCatIds(k)[0] for k in catNms}
        dictCOCOSorted = dict(sorted(dictCOCO.items(), key=lambda x: x[1]))

        IdCategories = list(range(1, len(catNms)+1))
        categories = dict(zip(list(dictCOCOSorted), IdCategories))

        arrayIds = np.array([k["id"] for k in images])
        annIds = coco.getAnnIds(imgIds=arrayIds, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        yolo_annotations = []

        for ann in anns:
            image = next(img for img in images if img["id"] == ann["image_id"])
            img_width = image["width"]
            img_height = image["height"]
            image_file_name = image["file_name"]
            file_name = os.path.splitext(image_file_name)[0]

            # COCO bbox format: [x, y, width, height]
            x, y, width, height = ann["bbox"]

            coco_bbox = BoundingBox.from_coco(x, y, width, height, image_size=(img_width, img_height))
            x_center, y_center, width, height = coco_bbox.to_yolo().raw_values

            yolo_annotations.append({
                "image_id": ann["image_id"],
                "file_name": file_name,
                "category_id": catIds.index(ann["category_id"]),
                "bbox": {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                }
            })

        cats = [{'id': int(value), 'name': key}
                for key, value in categories.items()]
        dataset = {}
        dataset["images"] = images
        dataset["annotations"] = yolo_annotations
        dataset["categories"] = cats

        return dataset

    def createJson(jsonfile: json) -> None:
        with open(f"{ROOT_PATH}/{subset_name}/labels/annotations.json", "w") as outfile:
            json.dump(jsonfile, outfile)

    def createYOLOLabels(dataset: dict) -> None:
        with alive_bar(len(dataset["annotations"]), title=f'Creating YOLO labels of the {subset_name} set:') as bar:
            for annotation in dataset["annotations"]:
                file_name = annotation["file_name"]
                category_id = annotation["category_id"]
                bbox = annotation["bbox"]
                with open(f"{ROOT_PATH}/{subset_name}/labels/{file_name}.txt", "a") as file:
                    file.write(f"{category_id} {bbox['x_center']} {bbox['y_center']} {bbox['width']} {bbox['height']}\n")
                bar()

    def createClasesFile(dataset: dict) -> None:
        categories = dataset["categories"]
        with open(f"{ROOT_PATH}/classes.txt", "w") as file:
            for i, category in enumerate(categories):
                if i < len(categories) - 1:
                    file.write(f"{category['name']}\n")
                else:
                    file.write(f"{category['name']}")

    def createDataYAML(dataset: dict) -> None:
        with open(f"{ROOT_PATH}/data.yaml", "w") as file:
            file.write("train: ../train/images\n")
            file.write("val: ../val/images\n")
            file.write(f"nc: {len(dataset['categories'])}\n")
            file.write(f"names: {catNms}")

    def downloadImages(images: list, title: str) -> None:
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        images_path = f"{ROOT_PATH}/{subset_name}/images"

        with alive_bar(len(images), title=title) as bar:
            for image in images:
                if not os.path.isfile(f"{images_path}/{image['file_name']}"):
                    img_data = session.get(image['coco_url']).content
                    with open(images_path + '/' + image['file_name'], 'wb') as handler:
                        handler.write(img_data)
                bar()

    images = myImages(imgShuffled, num_images)

    downloadImages(images, title=f'Downloading images of the {subset_name} set:')

    if args.format == "coco":
        dataset = cocoJson(images)
        createJson(dataset)
    elif args.format == "yolo":
        dataset = yoloJson(images)
        createJson(dataset)
        createYOLOLabels(dataset)
        createClasesFile(dataset)
        createDataYAML(dataset)


process_subset(args.train_annotation_file, args.training, "train")
process_subset(args.val_annotation_file, args.validation, "val")
