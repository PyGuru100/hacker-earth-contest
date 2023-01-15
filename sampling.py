import pandas as pd
from PIL import Image
import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

CLASS = 'name'

DESIRED_NUMBER_PER_CATEGORY = 1750

data = pd.read_csv('resources/dataset/train.csv')

classes_down_sampled = []


def current_length(current_class):
    return len(data[data[CLASS] == current_class])


def down_sample():
    for unique_class in data[CLASS].unique():
        subset = data[data[CLASS] == unique_class]
        most_repeated_images = list(subset['image_path'].value_counts().index.values)
        while current_length(unique_class) > DESIRED_NUMBER_PER_CATEGORY:
            most_repeated_images: list
            image = most_repeated_images.pop(0)
            indices_to_remove = data[(data[CLASS] == unique_class) & (data['image_path'] == image)].index
            data.drop(indices_to_remove, inplace=True)


def augment():
    image = ia.quokka(size=(512, 512))
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=65, y1=100, x2=200, y2=150),
        BoundingBox(x1=150, y1=80, x2=200, y2=130)
    ], shape=image.shape)
    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),
        iaa.HorizontalFlip(),
        iaa.GaussianBlur(),
        iaa.VerticalFlip(),  # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        )])
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    Image.fromarray(image).save("pls.png")
    Image.fromarray(image_aug).save("pls_pls.png")


augment()
