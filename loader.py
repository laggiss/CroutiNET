import numpy as np
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt

baseDir = "C:/Users/msawada/Desktop/arnaud/croutinet/placePulse/data"

trainDir = os.path.join(baseDir,"train/train.csv")
validationDir = os.path.join(baseDir,"validation/validation.csv")
testDir = os.path.join(baseDir,"test/test.csv")

correctImgDir = os.path.join(baseDir,"correctImg")




def loadImage(name):
    return np.array(Image.open(os.path.join(correctImgDir, name )))


def load(path):
    leftImages = []
    rightImages = []
    labels = []
    with open(path, 'r') as csvfileReader:
        reader = csv.reader(csvfileReader, delimiter=',')
        for line in reader:
            if line != [] and line[2] != '0.5':
                leftImages.append(loadImage(line[0] + '.jpg'))
                rightImages.append(loadImage(line[1] + '.jpg'))
                labels.append(int(line[2]))

    leftImages = np.array(leftImages)
    rightImages = np.array(rightImages)
    labels = np.array(labels)

    leftImages = leftImages.astype('float32') / 255
    rightImages = rightImages.astype('float32') / 255

    return (leftImages, rightImages, labels)
