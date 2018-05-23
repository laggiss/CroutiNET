import os
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD

from model2 import converge_model

baseDir = "C:/Users/msawada/Desktop/arnaud/croutinet/placePulse/data"

trainDir = os.path.join(baseDir,"train/train.csv")
validationDir = os.path.join(baseDir,"validation/validation.csv")
testDir = os.path.join(baseDir,"test/test.csv")

correctImgDir = os.path.join(baseDir,"correctImg")

def show(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

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


trainLeft, trainRight, trainLabels = load(trainDir)
validationLeft, validationRight, validationLabels = load(validationDir)
testLeft, testRight, testLabels = load(testDir)

model = converge_model()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=os.path.join(baseDir, 'placePulseSecond.h5'), verbose=1, save_best_only=True)
history = model.fit([trainLeft,trainRight], trainLabels, epochs=50, batch_size=64, validation_data=([validationLeft,validationRight],validationLabels), callbacks=[checkpointer])
show(history)

# bestModel = load_model(os.path.join(baseDir, 'placePulseSecond.h5'))
# result = bestModel.evaluate([testLeft, testRight], testLabels)
# print(result)