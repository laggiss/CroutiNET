from keras.applications import VGG16
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers

input_shape = (150, 150, 3)# imput shape for VGG16

left_input = Input(input_shape)
right_input = Input(input_shape)

# Feature extraction with VGG16 in siamese architecture
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

encoded_l = conv_base(left_input)
encoded_r = conv_base(right_input)

both = concatenate([encoded_l, encoded_r])
prediction = Dense(1, activation='sigmoid')(both)
siamese_part = Model(inputs=[left_input, right_input], outputs=prediction)

# Fusion part of the model
fusion_part = models.Sequential()
fusion_part.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(4, 4, 1)))
fusion_part.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(4, 4, 1)))
fusion_part.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(4, 4, 1)))
fusion_part.add(layers.Flatten())
fusion_part.add(layers.Dense(64, activation='relu'))
fusion_part.add(layers.Dense(1, activation='softmax'))

fusion_part.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
print(fusion_part.summary())
