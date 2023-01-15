import pandas as pd
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np

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
    base_image = Image.open('resources/dataset/images/0a1ea4614a9df912eeb8d1b40bffee74.jpg')
    noise = iaa.AdditiveGaussianNoise(10, 40)
    iaa.Sequential([])
    input_noise = noise.augment_image(np.array(base_image))
    result_image = Image.fromarray(input_noise)
    result_image.save('new.jpg')


augment()
