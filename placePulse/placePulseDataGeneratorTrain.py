from random import randint

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from model2 import converge_model
from loader import load
from generator import dataGenerator as d
import numpy as np
import pandas as pd
import os
import  matplotlib.pyplot as plt


IMG_SIZE = 224

baseDir = "C:/Users/msawada/Desktop/arnaud/croutinet/placePulse/data"

trainDir = os.path.join(baseDir,"train/train.csv")
validationDir = os.path.join(baseDir,"validation/validation.csv")
testDir = os.path.join(baseDir,"test/test.csv")
correctImgDir = os.path.join(baseDir,"correctImg")
check_point_weights = os.path.join(baseDir, 'modelWithDataAugmentation.h5')
histories = []

def show(listHistory):

    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for i in range(len(listHistory)):
        acc.extend(listHistory[i].history['acc'])
        val_acc.extend(listHistory[i].history['val_acc'])
        loss.extend(listHistory[i].history['loss'])
        val_loss.extend(listHistory[i].history['val_loss'])

    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, 'go', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

all_results = np.loadtxt(trainDir, str, delimiter=',')

print(all_results)

duelsDF = pd.DataFrame(all_results, None, ['left_id', 'right_id', 'winner'])
duelsDF['left_id'] = correctImgDir + "/" + duelsDF['left_id'] + '.jpg'
duelsDF['right_id'] = correctImgDir + "/" + duelsDF['right_id'] + '.jpg'


mask_yes = duelsDF['winner'] == '1'
yes = duelsDF[mask_yes]

mask_no = duelsDF['winner'] == '0'
no = duelsDF[mask_no]

model = converge_model()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

validationLeft, validationRight, validationLabels = load(validationDir)

# For batch training, the number of iterations of training model
n_iter=10
for iteration in range(n_iter):
    print(iteration / n_iter)

    # sample positive and negative cases for current iteration. It is faster to use fit on batch of n yesno and augment
    # that batch using datagen_class_aug_test than to use fit_generator with the datagen_class_aug_test and small batch
    # sizes.
    yesno = yes.sample(64).append(no.sample(64))
    labels = dict(zip([str(x) for x in yesno.index.tolist()],
                      [1 if x == '1' else 0 for x in yesno.winner.tolist()]))
    partition = {'train': list(zip([str(x) for x in yesno.index.tolist()], zip(yesno.left_id, yesno.right_id)))}

    batchSizeAug = len(yesno.index.tolist())
    # Set-up variables for augmentation of current batch of yesno in partition
    params = {
        'dim_x': IMG_SIZE,
        'dim_y': IMG_SIZE,
        'dim_z': 3,
        'batch_size': batchSizeAug,
        'shuffle': True
    }

    datagenargs = {
        'rotation_range': 2, 'width_shift_range': 0.2, 'height_shift_range': 0.2,
        'shear_range': 0.1,
        'zoom_range': 0.25, 'horizontal_flip': True, 'fill_mode': 'nearest'
    }

    training_generator = d.myDataGeneratorAug(**params).generate(labels, partition['train'], seed=randint(1, 10000),
                                                               datagenargs=datagenargs)
    X, y = training_generator.__next__()
    # zero center images
    X = np.array(X)

    # Fitting the model. Here you can see how the call to model fit works.  Note the validation data comes from
    # preloaded numpy arrays.

    checkpointer = ModelCheckpoint(filepath=check_point_weights, verbose=1, save_best_only=True)

    if iteration != 0 :
        model.load_weights(check_point_weights);

    history = model.fit(
        [X[0], X[1]],
        y,
        batch_size=64,
        epochs=10,
        validation_data=([validationLeft, validationRight], validationLabels),
        callbacks = [checkpointer])
    histories.append(history)

show(histories)