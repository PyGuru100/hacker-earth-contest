import pandas as pd

CLASS = 'name'

DESIRED_NUMBER_PER_CATEGORY = 1750

data = pd.read_csv('resources/dataset/train.csv')

classes_down_sampled = []

for unique_class in data[CLASS].unique():
    subset = data[data[CLASS] == unique_class]
    subset_length = len(subset)
    if subset_length > DESIRED_NUMBER_PER_CATEGORY:
        classes_down_sampled.append(unique_class)
        most_repeated_images = subset['image_path'].value_counts().index.values
        for image in most_repeated_images:
            if len(data[data[CLASS] == unique_class]) > DESIRED_NUMBER_PER_CATEGORY:
                row_to_remove_index = data[(data[CLASS] == unique_class) & (data['image_path'] == image)].index
                data.drop(row_to_remove_index, inplace=True)
            else:
                break
print(data['name'].value_counts())

