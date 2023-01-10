import pandas as pd
from autogluon.multimodal import MultiModalPredictor

from autogluon_model_build_script import load_data, alter_data_for_autogluon

data = pd.read_csv('resources/dataset/train.csv')
train_data, test_data = load_data()

alter_data_for_autogluon(train_data)
alter_data_for_autogluon(test_data)

predictor = MultiModalPredictor.load('resources/models/latest_model/content/tmp/1673328230-autogluon-classifier')
image_index = 350
print(test_data.iloc[image_index])
print(predictor.predict(test_data.iloc[image_index: image_index + 1]))
