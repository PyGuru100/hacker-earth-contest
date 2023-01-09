import os.path
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import uuid
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("resources/dataset/train.csv")
train_data, test_data = train_test_split(all_data)

image_class = 'class'
image_path = 'image_path'
name = 'name'
x_max = 'xmax'
x_min = 'xmin'
y_max = 'ymax'
y_min = 'ymin'


def alter_data_for_autogluon(_data):
    difference = _data.columns.difference([image_path, name])
    _data.drop(labels=difference, axis=1, inplace=True)
    _data[image_path] = _data[image_path].apply(lambda x: os.path.join(os.path.dirname(__file__),
                                                                       f"resources/dataset/images/{x}"))
    _data.rename(columns={name: "label"}, inplace=True)
    _data.rename(columns={image_path: "image"}, inplace=True)


alter_data_for_autogluon(train_data)
alter_data_for_autogluon(test_data)

model_path = f"./tmp/{uuid.uuid4().hex}-autogluon-classifier"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(
    train_data=train_data,
    time_limit=45,
)  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])
