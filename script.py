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
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

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

parser.add_argument("-anylabel", "--anylabel", type=bool,
                    help="Create a label file for each image for the AnyLabel tool.",
                    default=True)

parser.add_argument("-d", "--destination", type=str,
                    help="Destination folder.",
                    default="./")

parser.add_argument("-w", "--workers", type=int,
                    help="Number of workers.",
                    default=4)

parser.add_argument("-l", "--local", type=str,
                    help="Local path to the images.",
                    default="")

args = parser.parse_args()

ROOT_PATH = os.path.join(args.destination, "dataset")
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
        if num == -1:
            return images

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

        IdCategories = list(range(1, len(catNms) + 1))
        categories = dict(zip(list(dictCOCOSorted), IdCategories))

        arrayIds = np.array([k["id"] for k in images])
        annIds = coco.getAnnIds(imgIds=arrayIds, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        yolo_annotations = []

        image_dict = {img["id"]: img for img in images}

        with alive_bar(len(anns), title="Formatting annotations...") as bar:
            for i, ann in enumerate(anns):
                image = image_dict[ann["image_id"]]
                img_width = image["width"]
                img_height = image["height"]
                image_file_name = image["file_name"]
                file_name = os.path.splitext(image_file_name)[0]

                # COCO bbox format: [x, y, width, height]
                x, y, coco_width, coco_height = ann["bbox"]

                if any(dim is None or dim == 0 for dim in (coco_width, coco_height)):
                    continue

                coco_bbox = BoundingBox.from_coco(x, y, coco_width, coco_height, image_size=(img_width, img_height))
                x_center, y_center, width, height = coco_bbox.to_yolo(return_values=True)

                yolo_annotations.append({
                    "image_id": ann["image_id"],
                    "file_name": file_name,
                    "category_id": catIds.index(ann["category_id"]),
                    "bbox": {
                        "yolo": {
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height
                        },
                        "coco": {
                            "x": x,
                            "y": y,
                            "width": coco_width,
                            "height": coco_height
                        }
                    }
                })

                # Update the progress bar less frequently
                # if i % 100 == 0:
                bar()

        # TODO: Check id categories
        cats = [{'id': int(value) - 1, 'name': key}
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
                bbox = annotation["bbox"]["yolo"]
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

    def createAnyLabelJson(dataset: dict) -> None:
        annotations_by_image = {}

        category_dict = {cat["id"]: cat["name"] for cat in dataset["categories"]}
        image_dict = {img["id"]: img for img in dataset["images"]}

        with alive_bar(len(dataset["annotations"]), title="Formatting annotations for AnyLabel...") as bar:
            for i, annotation in enumerate(dataset["annotations"]):
                image_id = annotation["image_id"]
                file_name = annotation["file_name"]
                category_id = annotation["category_id"]
                bbox = annotation["bbox"]
                category_name = category_dict[category_id]
                image = image_dict[image_id]
                image_w, image_h = image["width"], image["height"]

                x = bbox["coco"]["x"]
                y = bbox["coco"]["y"]
                width = bbox["coco"]["width"]
                height = bbox["coco"]["height"]

                points = [
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height]
                ]

                shape = {
                    "label": category_name,
                    "score": None,
                    "points": points,
                    "group_id": category_id,
                    "description": "",
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {}
                }

                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = {
                        "version": "2.3.6",
                        "flags": {},
                        "shapes": [],
                        "imagePath": file_name + ".jpg",
                        "imageData": None,
                        "imageHeight": image_h,
                        "imageWidth": image_w
                    }

                annotations_by_image[image_id]["shapes"].append(shape)

                bar()

        def save_json_file(json_data, json_file_path):
            with open(json_file_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

        with alive_bar(len(annotations_by_image), title=f'Creating labels for AnyLabel tool of the {subset_name} set:') as bar:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(save_json_file, json_data, os.path.join(ROOT_PATH, subset_name, "images", f"{os.path.splitext(json_data['imagePath'])[0]}.json")) for json_data in annotations_by_image.values()]
                for future in as_completed(futures):
                    future.result()  # Ensure any exceptions are raised
                    bar()

    def copy_image(image, local_path, images_path):
        src_path = os.path.join(local_path, image['file_name'])
        dest_path = os.path.join(images_path, image['file_name'])
        shutil.copy2(src_path, dest_path)

    def copyImages(images: list, local_path: str, title: str) -> None:
        images_path = f"{ROOT_PATH}/{subset_name}/images"
        os.makedirs(images_path, exist_ok=True)

        with alive_bar(len(images), title=title) as bar:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(copy_image, image, local_path, images_path) for image in images]
                for future in as_completed(futures):
                    future.result()  # Ensure any exceptions are raised
                    bar()

    def download_image(session, image, images_path):
        if not os.path.isfile(f"{images_path}/{image['file_name']}"):
            img_data = session.get(image['coco_url']).content
            with open(images_path + '/' + image['file_name'], 'wb') as handler:
                handler.write(img_data)
        return image['file_name']

    def downloadImages(images: list, title: str) -> None:
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        images_path = f"{ROOT_PATH}/{subset_name}/images"
        os.makedirs(images_path, exist_ok=True)

        with alive_bar(len(images), title=title) as bar:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(download_image, session, image, images_path) for image in images]
                for future in as_completed(futures):
                    future.result()  # Ensure any exceptions are raised
                    bar()

    images = myImages(imgShuffled, num_images)

    if args.local != "":
        local = os.path.join(args.local, "dataset", subset_name, "images")
        copyImages(images, local, title=f"Copying images of the {subset_name} set:")
    else:
        downloadImages(images, title=f'Downloading images of the {subset_name} set:')

    if args.format == "coco":
        dataset = cocoJson(images)
        createJson(dataset)
    elif args.format == "yolo":
        dataset = yoloJson(images)
        createJson(dataset)
        createClasesFile(dataset)
        createDataYAML(dataset)

    if args.anylabel:
        createAnyLabelJson(dataset)
    else:
        createYOLOLabels(dataset)


process_subset(args.train_annotation_file, args.training, "train")
process_subset(args.val_annotation_file, args.validation, "val")
