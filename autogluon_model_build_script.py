import warnings
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import uuid
from IPython.display import Image, display
from autogluon.multimodal.utils.misc import shopee_dataset

warnings.filterwarnings('ignore')
download_dir = './ag_automm_tutorial_imgcls'
train_data, test_data = shopee_dataset(download_dir)
print(train_data)

model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(
    train_data=train_data,
    time_limit=30,  # seconds
)  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])

image_path = test_data.iloc[0]['image']
pil_img = Image(filename=image_path)
display(pil_img)

predictions = predictor.predict({'image': [image_path]})
print(predictions)

proba = predictor.predict_proba({'image': [image_path]})
print(proba)

feature = predictor.extract_embedding({'image': [image_path]})
print(feature[0].shape)
