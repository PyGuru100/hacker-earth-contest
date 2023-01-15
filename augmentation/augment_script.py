import uuid

import pandas as pd
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image
import numpy as np

DESIRED_TOTAL = 1750

data = pd.read_csv('../resources/dataset/train.csv')
counts = data['name'].value_counts()
augmented_dataset = data.copy(deep=True)[0:0]
for name in data['name'].unique():
    if 10 < counts[name] < DESIRED_TOTAL:
        number_desired_images = DESIRED_TOTAL - counts[name]
        image_paths_in_name = data[data['name'] == name]
        for i in range(10):
            image_row = image_paths_in_name.iloc[i:i + 1]

            image = np.asarray(Image.open(f"../resources/dataset/images/{image_row['image_path'].iloc[0]}"))

            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=image_row['xmin'].iloc[0], y1=image_row['ymin'].iloc[0],
                            x2=image_row['xmax'].iloc[0], y2=image_row['ymax'].iloc[0])
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
            image_path = f"{uuid.uuid4()}-augmented.jpg"
            augmented_dataset = augmented_dataset.append([{
                "image_path": image_path,
                "xmin": bbs_aug[0].x1,
                "xmax": bbs_aug[0].x2,
                "ymin": bbs_aug[0].y1,
                "ymax": bbs_aug[0].y2
            }], ignore_index=True)
            Image.fromarray(image_aug).save(f'./augmented_images/{image_path}')
            augmented_dataset: pd.DataFrame
            augmented_dataset.to_csv("augmented_dataset.csv")

