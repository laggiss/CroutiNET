from keras import Input, Model, Sequential
from keras.applications import VGG19
from keras.layers import concatenate, Conv2D, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD

IMG_SIZE = 224


def con_model():

    vision_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # In order to prevent overfitting, as advised in the keras documentation,
    # we freeze the 18 first convolutional layers (corresponding to the first 2 blocks)

    for layer in vision_model.layers[:18]:
        layer.trainable = False

    # Definition of the 2 inputs

    img_a = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    img_b = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Outputs of the vision model corresponding to the inputs
    # Note that this method applies the 'tied' weights between the branches

    out_a = vision_model(img_a)
    out_b = vision_model(img_b)

    # Concatenation of these ouputs

    concat = concatenate([out_a, out_b])

    # The classification model is the full model: it takes 2 images as input and
    # returns a number between 0 and 1. The closest the number is to 1, the more confident

    classification_model = Model([img_a, img_b], concat)

    return classification_model


def converge_model():
    m = Sequential()
    m.add(con_model())
    m.add(Conv2D(512, (3, 3), activation='relu', padding='same', name="block_converge_2"))  # ,input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    m.add(Dropout(0.2))
    m.add(Conv2D(512, (3, 3), activation='relu', padding='same', name="block_converge_3"))
    m.add(Dropout(0.3))
    m.add(Conv2D(512, (3, 3), activation='relu', padding='same', name="block_converge_4"))
    m.add(BatchNormalization())
    m.add(Flatten())
    m.add(Dense(1, activation='sigmoid', name="block_converge_5k"))

    return m

model = converge_model()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])