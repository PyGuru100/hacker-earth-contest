import pandas as pd

DESIRED_TOTAL = 1750

data = pd.read_csv('../resources/dataset/train.csv')
print(data['name'].value_counts())
