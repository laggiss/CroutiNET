from random import randint
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK, tpe, Trials
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
from model2 import converge_model
from loader import load
from generator import dataGenerator as d
import numpy as np
import pandas as pd
import os

from representation.representation import show

IMG_SIZE = 224

baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"

trainDir = os.path.join(baseDir, "train/train.csv")
validationDir = os.path.join(baseDir, "validation/validation.csv")
testDir = os.path.join(baseDir, "test/test.csv")
roads_loubna_dir = os.path.join(baseDir, "roads_loubna")
check_point_weights = os.path.join(baseDir, 'modelWithDataAugmentation2.h5')
histories = []



all_results = np.loadtxt(trainDir, str, delimiter=',')

duelsDF = pd.DataFrame(all_results, None, ['left_id', 'right_id', 'winner'])
duelsDF['left_id'] = roads_loubna_dir + "/" + duelsDF['left_id']
duelsDF['right_id'] = roads_loubna_dir + "/" + duelsDF['right_id']
#print(duelsDF)

mask_yes = duelsDF['winner'] == '1'
yes = duelsDF[mask_yes]

mask_no = duelsDF['winner'] == '0'
no = duelsDF[mask_no]

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

validationLeft, validationRight, validationLabels = load(validationDir)

def data():

    # sample positive and negative cases for current iteration. It is faster to use fit on batch of n yesno and augment
    # that batch using datagen_class_aug_test than to use fit_generator with the datagen_class_aug_test and small batch
    # sizes.
    yesno = yes.sample(500).append(no.sample(500))
    print('yesno created')
    labels = dict(zip([str(x) for x in yesno.index.tolist()],
                      [1 if x == '1' else 0 for x in yesno.winner.tolist()]))
    print('labels created')

    partition = {'train': list(zip([str(x) for x in yesno.index.tolist()], zip(yesno.left_id, yesno.right_id)))}
    print('partition created')

    batchSizeAug = len(yesno.index.tolist())
    print('batcgSizeAug created')
    # Set-up variables for augmentation of current batch of yesno in partition
    params = {
        'dim_x': IMG_SIZE,
        'dim_y': IMG_SIZE,
        'dim_z': 3,
        'batch_size': batchSizeAug,
        'shuffle': True
    }
    print('params created')

    datagenargs = {
        'rotation_range': {{uniform(0, 10)}}, 'width_shift_range': {{uniform(0, 1)}}, 'height_shift_range': {{uniform(0, 1)}},
        'shear_range': {{uniform(0, 1)}},
        'zoom_range': {{uniform(0, 1)}}, 'horizontal_flip': True, 'fill_mode': 'nearest'
    }
    print('datagenargs created')

    training_generator = d.myDataGeneratorAug(**params).generate(labels, partition['train'], seed=randint(1, 10000),
                                                                 datagenargs=datagenargs)
    print('training generator created')

    X, y = training_generator.__next__()
    print('X,y created')
    # zero center images
    X = np.array(X)
    #X = X/255

    print('X as arrray created')
    # Fitting the model. Here you can see how the call to model fit works.  Note the validation data comes from
    # preloaded numpy arrays.

    print('one hot encoding ...')
    y = to_categorical(y)

    return (X, y, [validationLeft, validationRight], validationLabels)


def model(X_train, Y_train, X_test, Y_test):
    model = load_model(check_point_weights)
    history = model.fit(
        [X_train[0], X_train[1]],
        Y_train,
        batch_size=16,
        epochs=20,
        validation_data=( [X_test[0], X_test[1]], Y_test))
    score, acc = model.evaluate([X_test[0], X_test[1]], Y_test, verbose=0)
    print('Test accuracy:', acc)
    histories.append(history)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

show(histories)
X_train, Y_train, X_test, Y_test = data()

best_run, best_model = optim.minimize(  model=model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=5,
                                        trials=Trials())

print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
