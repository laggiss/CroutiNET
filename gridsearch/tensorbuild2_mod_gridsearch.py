from __future__ import print_function

import sys
import warnings

sys.path.append(r"C:\Users\laggi\OneDrive\a11PyCharmCNN")
import dataset

from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from hyperas.distributions import uniform,choice
from numpy import argmax
from sklearn.utils import class_weight
K.set_image_dim_ordering('tf')
import numpy as np
def data():
    img_size = 224
    batch_size=16
    trainingDataPath = "C:/GIST/tensorflow/buildings/training_data"
    classes = ['C1H', 'C1L', 'C1M', 'C2H', 'C2L', 'S1L', 'W1', 'W2', 'URML', 'URMM']  # ,'URMM']#'W1','W2',
    train_path = trainingDataPath
    data = dataset.read_train_sets(train_path, img_size, classes, validation_size=.1, bcropprop=.1, randomstate=2)
    X_train = data.train.images
    y_train = data.train.labels
    X_test = data.valid.images
    y_test = data.valid.labels
    x = argmax(y_train, axis=1)
    x = x.astype('int32')
    imbalanced = class_weight.compute_class_weight(None, np.unique(x), x)
    balanced=class_weight.compute_class_weight('balanced', np.unique(x), x)
    imbalanced=dict(zip(np.unique(x),imbalanced))
    balanced = dict(zip(np.unique(x), balanced))


    datagen = ImageDataGenerator(
        rotation_range=7,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.1,
        zoom_range=0.8,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)
    lc=len(classes)
    lt=X_train.shape[0]
    return(train_gen,X_test,y_test,lt,balanced,imbalanced)

def create_model(train_gen,X_test,y_test,train_len,balanced,imbalanced):

    num_channels = 3
    img_size = 224
    num_classes=10
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    WEIGHTS_PATH_NO_TOP = 'C:/GIST/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    def VGG19_Wz(include_top=True, weights='imagenet',
                 input_tensor=None, input_shape=None,
                 pooling=None,
                 classes=1000):
        """Instantiates the VGG19 architecture.

        Optionally loads weights pre-trained
        on ImageNet. Note that when using TensorFlow,
        for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.

        The model and the weights are compatible with both
        TensorFlow and Theano. The data format
        convention used by the model is the one
        specified in your Keras config file.

        # Arguments
            include_top: whether to include the 3 fully-connected
                layers at the top of the network.
            weights: one of `None` (random initialization)
                or "imagenet" (pre-training on ImageNet).
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 224)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 48.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

        # Returns
            A Keras model instance.

        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')
        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=224,
                                          min_size=48,
                                          data_format=K.image_data_format(),
                                          include_top=include_top)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        if include_top:
            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(classes, activation='softmax', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='vgg19')

        # load weights
        if weights == 'imagenet':
            if include_top:
                weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = WEIGHTS_PATH_NO_TOP  # ('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',                                   WEIGHTS_PATH_NO_TOP,                                   )
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                layer_utils.convert_all_kernels_in_model(model)

            if K.image_data_format() == 'channels_first':
                if include_top:
                    maxpool = model.get_layer(name='block5_pool')
                    shape = maxpool.output_shape[1:]
                    dense = model.get_layer(name='fc1')
                    layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

                if K.backend() == 'tensorflow':
                    warnings.warn('You are using the TensorFlow backend, yet you '
                                  'are using the Theano '
                                  'image data format convention '
                                  '(`image_data_format="channels_first"`). '
                                  'For best performance, set '
                                  '`image_data_format="channels_last"` in '
                                  'your Keras config '
                                  'at ~/.keras/keras.json.')
        return model


    conv_base = VGG19_Wz(weights='imagenet',
                         include_top=False,
                         input_shape=(img_size, img_size, num_channels))
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten(name='flatten_1'))
    model.add(Dense(4096, name='fc_1'))
    model.add(Activation('relu', name='fc_actv_1'))
    model.add(Dense(4096, name='fc_2'))
    model.add(Activation('relu', name='fc_actv_2'))
    model.add(Dropout({{uniform(0, 0.5)}}, name='fc_dropout_2'))
    model.add(Dense(1000, name='fc_6'))
    model.add(Activation('relu', name='fc_actv_6'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, name='fc_7'))
    model.add(Activation('softmax', name='fc_actv_7'))
    conv_base.trainable = False
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.4, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    n_epoch = 3
    model.fit_generator(generator=train_gen, steps_per_epoch=50, epochs=n_epoch,verbose=1, validation_data=(X_test, y_test), class_weight={{choice([imbalanced,balanced])}})
    score, acc = model.evaluate(X_test,y_test)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


## Diagnostics ##
#c:\ProgramData\Anaconda3>python.exe C:\Users\laggi\OneDrive\a11PyCharmCNN\tensorbuild2_mod_gridsearch.py
if __name__ == '__main__':
    #train_gen, X_test, y_test, train_len = data()

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=2,
                                          trials=Trials())

    # print("Evaluation of best performing model:")
    #
    # print(best_model.evaluate(X_test,y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)