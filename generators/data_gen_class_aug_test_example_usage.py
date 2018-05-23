import sys
import warnings
import numpy as np
from random import randint
import pandas as pd
sys.path.append(r"F:\OneDrive\a11PyCharmCNN")
from datagen_class_aug_test import myDataGeneratorAug

# retrain.txt has simple structure of imagepath1 imagepath2 0 e,g.,
# c:/gist/trainottawa/45.380724,-75.746611/2012_5465yPwRdN0BpL4eWVJtpQ.jpg c:/gist/trainottawa/45.380724,-75.746611/2014_PMqwrRaKTPzGn6oREYQD5A.jpg 0
# c:/gist/trainottawa/45.380724,-75.746611/2014_PMqwrRaKTPzGn6oREYQD5A.jpg c:/gist/trainottawa/45.380724,-75.746611/2014_ReXUBXkTgJYN3V2dsp5QQg.jpg 0
all_results = np.loadtxt('c:/gist/retrain.txt', str)

duelsDF = pd.DataFrame(all_results, None, ['left_id', 'right_id', 'winner'])
mask_yes = duelsDF['winner'] == '1'
yes = duelsDF[mask_yes]
mask_no = duelsDF['winner'] == '0'
no = duelsDF[mask_no]

# For batch training, the number of iterations of training model
n_iter=1
for iteration in range(n_iter):
    print(iteration / n_iter)

    # sample positive and negative cases for current iteration. It is faster to use fit on batch of n yesno and augment
    # that batch using datagen_class_aug_test than to use fit_generator with the datagen_class_aug_test and small batch
    # sizes.
    yesno = yes.sample(10).append(no.sample(10))
    labels = dict(zip([str(x) for x in yesno.index.tolist()],
                      [1 if x == '1' else 0 for x in yesno.winner.tolist()]))
    partition = {'train': list(zip([str(x) for x in yesno.index.tolist()], zip(yesno.left_id, yesno.right_id)))}

    batchSizeAug = len(yesno.index.tolist())
    # Set-up variables for augmentation of current batch of yesno in partition
    params = {
        'dim_x': 224,
        'dim_y': 224,
        'dim_z': 3,
        'batch_size': batchSizeAug,
        'shuffle': True
    }

    datagenargs = {
        'rotation_range': 2, 'width_shift_range': 0.2, 'height_shift_range': 0.2,
        'shear_range': 0.1,
        'zoom_range': 0.25, 'horizontal_flip': True, 'fill_mode': 'nearest'
    }

    training_generator = myDataGeneratorAug(**params).generate(labels, partition['train'], seed=randint(1, 10000),
                                                               datagenargs=datagenargs)


    X, y = training_generator.__next__()
    # zero center images
    X = np.array(X)

    # Fitting the model. Here you can see how the call to model fit works.  Note the validation data comes from
    # preloaded numpy arrays.

    # cls_wt={0:10,1:90}
    # checkpointer = ModelCheckpoint(filepath=check_point_weights, verbose=1, save_best_only=True)
    # mout = classification_model.fit([X[0], X[1]],
    #                                 y,
    #                                 batch_size=24,
    #                                 epochs=50,
    #                                 validation_data=([X_valid[:, 0], X_valid[:, 1]], y_valid),
    #                                 verbose=1,class_weight=cls_wt)