import os

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from representation.representation import show
from keras.models import load_model
from loader import load

IMG_SIZE = 224

baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"

trainDir = os.path.join(baseDir, "train/train.csv")
validationDir = os.path.join(baseDir, "validation/validation.csv")
testDir = os.path.join(baseDir, "test/test.csv")


check_point_model = os.path.join(baseDir, 'modelWithDataAugmentation4.h5')
best_model = os.path.join(baseDir, 'bestModel.h5')
histories = []

model = load_model(best_model)

sgd = SGD(lr=1e-4, decay=1e-4, momentum=0.8437858241496619, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

trainLeft, trainRight, trainLabels = load(trainDir)
validationLeft, validationRight, validationLabels = load(validationDir)

checkpointer = ModelCheckpoint(filepath=check_point_model, verbose=1, save_best_only=True)

history = model.fit(
    [trainLeft, trainRight],
    trainLabels,
    batch_size=16,
    epochs=10,
    validation_data=([validationLeft, validationRight], validationLabels),
    callbacks=[checkpointer])

show([history], False)