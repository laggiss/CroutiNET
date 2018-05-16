from keras.applications import VGG16
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras import backend as K

input_shape = (150, 150, 3)# imput shape for VGG16

left_input = Input(input_shape)
right_input = Input(input_shape)

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

encoded_l = conv_base(left_input)
encoded_r = conv_base(right_input)


both = concatenate([encoded_l, encoded_r])
prediction = Dense(1, activation='sigmoid')(both)
siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
