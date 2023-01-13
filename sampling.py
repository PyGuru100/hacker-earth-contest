import pandas as pd

CLASS = 'name'

DESIRED_NUMBER_PER_CATEGORY = 1750

data = pd.read_csv('resources/dataset/train.csv')

classes_down_sampled = []


def current_length(current_class):
    return len(data[data[CLASS] == current_class])


for unique_class in data[CLASS].unique():
    subset = data[data[CLASS] == unique_class]
    most_repeated_images = list(subset['image_path'].value_counts().index.values)
    while current_length(unique_class) > DESIRED_NUMBER_PER_CATEGORY:
        most_repeated_images: list
        image = most_repeated_images.pop(0)
        indices_to_remove = data[(data[CLASS] == unique_class) & (data['image_path'] == image)].index
        data.drop(indices_to_remove, inplace=True)
print(data['name'].value_counts())

