import pandas as pd


data = pd.read_csv('resources/dataset/train.csv')
image_class = 'class'
image_path = 'image_path'
name = 'name'
x_max = 'xmax'
x_min = 'xmin'
y_max = 'ymax'
y_min = 'ymin'

if __name__ == '__main__':
    print(data[name].unique())
    print(data[image_class].unique())
