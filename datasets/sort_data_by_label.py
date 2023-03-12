"""
This file is used to copy images from one folder to subfolders based on their labels. The labels are stored in a csv file.
"""



import os
import shutil

import numpy as np
import pandas as pd

# Path to the folder containing the images
data_dir = '/Users/idanlevy/Documents/medpic/brazilian database/data/images'

# Path to the csv file containing the labels
csv_file = '/Users/idanlevy/Documents/medpic/brazilian database/data/images/labels.csv'

# Path to the folder where the images will be copied
dest_dir = '/Users/idanlevy/PycharmProjects/ECGDetection/data/BR/images'

# Create the destination folder if it doesn't exist
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

# Read the csv file
df = pd.read_csv(csv_file)

# Extract the labels from the csv file
labels = df.drop('filename', axis=1).values
# create another label of "NORMAL" if all labels are 0
labels = np.append(labels, np.array([1 if np.sum(row) == 0 else 0 for row in labels]).reshape(-1, 1), axis=1)

# Extract the image names from the csv file
img_names = df['filename'].tolist()

# Create a folder for each label with the label's name, if it doesn't exist
for i in range(len(labels[0])):
    if i < len(labels[0]) - 1:
        if not os.path.exists(os.path.join(dest_dir, df.columns[i + 1])):
            os.mkdir(os.path.join(dest_dir, df.columns[i + 1]))
    elif not os.path.exists(os.path.join(dest_dir, 'NORMAL')):
        os.mkdir(os.path.join(dest_dir, 'NORMAL'))


# Copy the images to the relevant folder, if the image does not exist in the folder
for i in range(len(img_names)):
    for j in range(len(labels[0])):
        if labels[i][j] == 1:
            if j < len(labels[0]) - 1:
                if not os.path.exists(os.path.join(dest_dir, df.columns[j + 1], img_names[i] + '.png')):
                    shutil.copy(os.path.join(data_dir, img_names[i] + '.png'), os.path.join(dest_dir, df.columns[j + 1]))
            elif not os.path.exists(os.path.join(dest_dir, 'NORMAL', img_names[i] + '.png')):
                shutil.copy(os.path.join(data_dir, img_names[i] + '.png'), os.path.join(dest_dir, 'NORMAL'))


# verify that all images in each folder are labeled correctly
# load the csv file
df = pd.read_csv(csv_file)
# go over each folder
for folder in os.listdir(dest_dir):
    # if the folder's name is "NORMAL", check that all labels are 0:
    if folder == 'NORMAL':
        for img in os.listdir(os.path.join(dest_dir, folder)):
            if np.sum(df[df['filename'] == img[:-4]].drop('filename', axis=1).values) != 0:
                print(img)
        continue
    if not os.path.isdir(os.path.join(dest_dir, folder)):
        continue
    # go over each image in the folder
    for img in os.listdir(os.path.join(dest_dir, folder)):
        # if the image is not labeled correctly, print the image name
        if df[df['filename'] == img[:-4]][folder].values[0] != 1:
            print(img)
