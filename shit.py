import os.path

import cv2
import __main__

__main__.cv2 = cv2


def plot_one_box(x, img, color=None, label=None, line_thickness=None, Inverted=False):
    # Plots one bounding box on image img
    tl = line_thickness or 2  # line thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = tl  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 2, thickness=tf)[0]
    if Inverted == True:
        c1 = c2
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    else:
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 2, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA, )


# Using readlines()
file1 = open('resources/dataset/train.csv', 'r')
Lines = file1.readlines()

count = 0
# Strips the newline character
for line in Lines:
    if count == 0:
        count += 1
        continue
    file_id_path = line.split(',')[1]
    # open image in cv2
    if file_id_path != '53d3797457a0d2e3afe146e2f797e77e.jpg':
        continue
    img = cv2.imread(os.path.join(os.path.dirname(__file__), f"resources/dataset/images/{file_id_path}"))
    h, w, c = img.shape
    cat = line.split(',')[2]
    xmax = int(float(line.split(',')[3])) * 2
    xmin = int(float(line.split(',')[4])) * 2
    ymax = int(float(line.split(',')[5])) * 2
    ymin = int(float(line.split(',')[6])) * 2
    # plot the box
    plot_one_box([xmin, ymin, xmax, ymax], img, color=(0, 255, 0), label=cat, line_thickness=2)
    # save the image
    # you might need to create the folder "drawn" first!

    cv2.imshow('image window', img)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(os.path.dirname(__file__), f"resources/drawn/images/{file_id_path}"), img)
    print("Line {}: {}".format(count, line.strip()))
    count += 1
