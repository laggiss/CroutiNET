from io import BytesIO
from PIL import Image


def api_download2(panoid, ppid, heading, flat_dir, key, width=640, height=640,
                  fov=120, pitch=0, extension='jpg', year=2017):
    """
    Download an image using the official API. These are not panoramas.

    Params:
        :panoid: the panorama id
        :heading: the heading of the photo. Each photo is taken with a 360
            camera. You need to specify a direction in degrees as the photo
            will only cover a partial region of the panorama. The recommended
            headings to use are 0, 90, 180, or 270.
        :flat_dir: the direction to save the image to.
        :key: your API key.
        :width: downloaded image width (max 640 for non-premium downloads).
        :height: downloaded image height (max 640 for non-premium downloads).
        :fov: image field-of-view.
        :image_format: desired image format.

    You can find instructions to obtain an API key here: https://developers.google.com/maps/documentation/streetview/
    """

    fname = str(ppid)  # "%s_%s_%s_%s" % (year, panoid,str(ppid),str(heading))
    image_format = extension if extension != 'jpg' else 'jpeg'

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        # maximum permitted size for free calls
        "size": "%dx%d" % (width, height),
        "fov": fov,
        "pitch": pitch,
        "heading": heading,
        "pano": panoid,
        "key": key
    }

    response = requests.get(url, params=params, stream=True)
    try:
        img = Image.open(BytesIO(response.content))
        filename = '%s/%s.%s' % (flat_dir, fname, extension)
        img.save(filename, image_format)
    except:
        print("Image not found")
        filename = None
    del response
    return filename


#### Get data
import streetview
import requests
import pandas as pd
#########################################################
## Get images for duels wealthy
import csv

duels_file = 'C:/users/laggi/downloads/wealthy.csv'

duelsList = []

with open(duels_file) as votefile:
    reader = csv.reader(votefile)

    # skip header
    colnames = next(reader)

    for row in reader:
        duelsList.append(row)

duelsDF = pd.DataFrame(duelsList, None, colnames)

duelsDF[['left_lat', 'left_long', 'right_lat', 'right_long']] = duelsDF[
    ['left_lat', 'left_long', 'right_lat', 'right_long']].apply(pd.to_numeric)

# Get dataframe subset for downloading getting only unqiue ids
import numpy as np

duelsDF_head = duelsDF#[40001:131731]
k = duelsDF_head.left_id.unique()
l = duelsDF_head.right_id.unique()
kl = np.concatenate([k, l])
unique_ids = np.unique(kl)
unique_ids = unique_ids.tolist()

DIRECTORY = r'C:\gist\wealthy'

# Get list of existing files in folder that have already been downloaded
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir(DIRECTORY) if isfile(join(DIRECTORY, f))]
onlyfiles = [x.split(".")[0] for x in onlyfiles]

# Create list of unique ids from the head subset that have not already been downloaded
dlist = []
for id in unique_ids:
    if id not in onlyfiles:
        dlist.append(id)

# Get id, lat and long for files to be downloaded
downloadList = []
for id in dlist:
    dfrows = duelsDF_head[duelsDF_head.left_id == id]
    if len(dfrows) > 0:
        llat = dfrows[0:1].left_lat.values[0]
        llong = dfrows[0:1].left_long.values[0]
        downloadList.append([id, llat, llong])
    else:
        dfrows = duelsDF_head[duelsDF_head.right_id == id]
        rlat = dfrows[0:1].right_lat.values[0]
        rlong = dfrows[0:1].right_long.values[0]
        downloadList.append([id, rlat, rlong])

API_KEY = 'AIzaSyC_cKyaxPoDPTtN4IgiOJ_e_9ytbMDk4lE'
# # Output folder for panaorama images

for row in downloadList:
    ppidLeftLat, ppidLeftLon = row[1], row[2]
    panIds = streetview.panoids(ppidLeftLat, ppidLeftLon)  # Randomly select one of the n panaormas at this location
    if len(panIds) > 0:
        # pid = random.randint(0, len(panIds) - 1)
        pid = len(panIds) - 1

        img = api_download2(panIds[pid]["panoid"], row[0], 0, DIRECTORY, API_KEY, fov=80, pitch=0,
                            year=panIds[pid]["year"])

# result_y = []
# for index, row in duelsDF.head(3000).iterrows():  # .iloc([50:]).iterrows()
#     ppidLeft = row['left_id']
#     ppidLeftLat = row['left_lat']
#     ppidLeftLon = row['left_long']
#     ppidRight = row['right_id']
#     ppidRightLat = row['right_lat']
#     ppidRightLon = row['right_long']
#
#     panIds = streetview.panoids(ppidLeftLat, ppidLeftLon)  # Randomly select one of the n panaormas at this location
#     if len(panIds) > 0:
#         # pid = random.randint(0, len(panIds) - 1)
#         pid = len(panIds) - 1
#
#         img = api_download2(panIds[pid]["panoid"], row['left_id'], 0, DIRECTORY, API_KEY, fov=80, pitch=0,
#                             year=panIds[pid]["year"])
#         result_y.append([row['left_id'], row['right_id'], row['winner']])
#
#     panIds = streetview.panoids(ppidRightLat, ppidRightLon)  # Randomly select one of the n panaormas at this location
#     if len(panIds) > 0:
#         # pid = random.randint(0, len(panIds) - 1)
#         pid = len(panIds) - 1
#
#         img = api_download2(panIds[pid]["panoid"], row['right_id'], 0, DIRECTORY, API_KEY, fov=80, pitch=0,
#                             year=panIds[pid]["year"])
#         result_y.append([row['left_id'], row['right_id'], row['winner']])
#
# result_y = pd.DataFrame(result_y, columns=["left", "right", "winner"])
# result_y['left'][0]+".jpg"

# result_y = result_y[result_y.left != '5140d98dfdc9f04926003c24']
# result_y = result_y[result_y.right != '5140c87bfdc9f04926002253']
# result_y = result_y[result_y.left != '513d7e35fdc9f03587007365']
# result_y = result_y[result_y.left != '513f2a35fdc9f0358700d4d4']
# result_y = result_y[result_y.right != '513f2a35fdc9f0358700d4d4']
# result_y = result_y[result_y.left != '51421c1ffdc9f04926008516']
# result_y = result_y[result_y.right != '51421c1ffdc9f04926008516']
# result_y = result_y[result_y.left != '5140d98dfdc9f04926003c24']
# result_y = result_y[result_y.right != '5140d98dfdc9f04926003c24']
# result_y = result_y[result_y.left != '50f42c49fdc9f065f0001a11']
# result_y = result_y[result_y.right != '50f42c49fdc9f065f0001a11']

# with open("c:/GIST/result_y.txt", 'w') as f:
#     for index, row in result_y.iterrows():
#         f.write("{},{},{}\n".format(row['left'], row['right'], row['winner']))
#
# #Read in data file containing left,right and winner
#
# result_y = pd.read_csv("c:/GIST/result_y.txt", header=None, names=["left", "right", "winner"])

################
## Clean out images with no variation
################


fdir = DIRECTORY#"c:/gist/wealthy/"
from os import listdir
from os.path import isfile, join
from pathlib import Path
from scipy import misc
import numpy as np
from scipy import misc
import os

allf = [f for f in listdir(fdir) if isfile(join(fdir, f))]
allf = [x.split(".")[0] for x in allf]

for f in allf:
    ipath=fdir + f + '.jpg'
    img1 = misc.imread(ipath)
    img1 = np.array(img1)[10:100,10:100]
    print("{},{}".format(ipath,img1.std()))
    if img1.std()<4:
        os.remove(ipath)
    # img1 = img1.astype(np.float32) / 255.0
    # # X[i, 0, :, :, :] = np.array(img1)
    # X1.append(np.array(img1))
    #
    # img2 = misc.imread(self.dir + ID[1][1] + '.jpg')
    # img2 = misc.imresize(img2, (self.dim_x, self.dim_y))
    # img2 = np.array(img2)
    # img2 = img2.astype(np.float32) / 255.0
    # X2.append(np.array(img2))



import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

kz = training_generator.__next__()
plt.figure(1)#figsize=(8,20))

# gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1, 1, 1], wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)
nr=kz[0][0].shape[0]
for i in range(int(nr/5)):
    pic=i
    img_A = kz[0][0][pic,:]
    img_B = kz[0][1][pic,:]
    plot_image = np.concatenate((img_A, img_B), axis=1)
    if kz[1][pic]==0:
        kstr='Left'
    else:
        kstr='Right'
    # if (pic+1)<=16:
    #     ax=plt.subplot(nr,1,pic+1)
    # else:
    ax = plt.subplot(int(nr/5), 1, pic + 1)
    ax.axis('off')
    ax.set_title(kstr)
    plt.imshow(plot_image)
plt.show()


ax=plt.subplot(212)
ax.axis('off')
ax.set_title(kstr)
plt.imshow(plot_image)
plt.show()




X = []
y = []

# Variable corresponding to the images size in pixels. 224 is the usual value

IMG_SIZE = 224

from pathlib import Path
from scipy import misc

for index, row in duelsDF.head(10000).iterrows():

    # y.append(int(row[-1]))

    path1 = "{}/{}{}".format(DIRECTORY, row['left_id'], ".jpg")
    path2 = "{}/{}{}".format(DIRECTORY, row['right_id'], ".jpg")
    if Path(path1).is_file() & Path(path2).is_file():
        # The label is the last value of the row
        if row['winner'] == 'left':
            y.append(1)
        else:
            y.append(0)
        # We read the images and resize them
        img1 = misc.imread(path1)
        img2 = misc.imread(path2)
        img1 = misc.imresize(img1, (IMG_SIZE, IMG_SIZE))
        img2 = misc.imresize(img2, (IMG_SIZE, IMG_SIZE))
        X.append([img1, img2])

import numpy as np
from matplotlib import pyplot as plt

# plt.imshow(X[1][1])

X = np.array(X[0:8000])
y = np.array(y[0:8000])
# We use the utility prune_dataset2 to have 1000 negatives and 200 positives
# X, y = svm_tools.prune_dataset2(X, y, [1000, 200])

# Zero-centering the data and standardizing
X = X.astype(float) - np.mean(X, axis=0)
X /= np.std(X, axis=0)

plt.imshow(X[100][0])

# def doubleGenerator(generator1,generator2):
#
#     while True:
#         for (x1,y1),(x2,y2) in zip(generator1,generator2):
#             yield ([x1,x2],y1)
#
#
# input_imgen = ImageDataGenerator(rescale=1. / 255,
#                                  shear_range=0.2,
#                                  zoom_range=0.2,
#                                  rotation_range=5.,
#                                  horizontal_flip=True)
#
# test_imgen = ImageDataGenerator(rescale=1. / 255)
#
#
# def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
#     genX1 = generator.flow_from_directory(dir1,
#                                           target_size=(img_height, img_width),
#                                           class_mode='categorical',
#                                           batch_size=batch_size,
#                                           shuffle=False,
#                                           seed=7)
#
#     genX2 = generator.flow_from_directory(dir2,
#                                           target_size=(img_height, img_width),
#                                           class_mode='categorical',
#                                           batch_size=batch_size,
#                                           shuffle=False,
#                                           seed=7)
#     while True:
#         X1i = genX1.next()
#         X2i = genX2.next()
#         yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label
#
#
# inputgenerator = generate_generator_multiple(generator=input_imgen,
#                                              dir1=train_dir_1,
#                                              dir2=train_dir_2,
#                                              batch_size=batch_size,
#                                              img_height=img_height,
#                                              img_width=img_height)
#
# testgenerator = generate_generator_multiple(test_imgen,
#                                             dir1=train_dir_1,
#                                             dir2=train_dir_2,
#                                             batch_size=batch_size,
#                                             img_height=img_height,
#                                             img_width=img_height)
#
# history = model.fit_generator(inputgenerator,
#                               steps_per_epoch=trainsetsize / batch_size,
#                               epochs=epochs,
#                               validation_data=testgenerator,
#                               validation_steps=testsetsize / batch_size,
#                               use_multiprocessing=True,
#                               shuffle=False)



import numpy as np


class DataGenerator(object):
    'Generates data for Keras'

    def __init__(self, dim_x=32, dim_y=32, dim_z=32, batch_size=32, shuffle=True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, labels, list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp)

                yield X, y

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 2, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            iml = misc.imread("c:/gist/gentest/" + ID + '.jpg')
            iml = np.array(iml)
            iml = iml.astype(float)
            X[i, 0, :, :, :] = np.array(iml)
            X[i, 1, :, :, :] = np.array(iml)
            # Store class
            y[i] = labels[ID]

        return X, y  # sparsify(y)


def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 2  # Enter number of classes
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])


fdir = "c:/gist/gentest"
from os import listdir
from os.path import isfile, join

allf = [f for f in listdir(fdir) if isfile(join(fdir, f))]
allf = [x.split(".")[0] for x in allf]

partition = {'train': allf}

from random import *

randBinList = lambda n: [randint(0, 1) for b in range(1, n + 1)]
vals = randBinList(len(allf))

labels = dict(zip(allf, vals))

params = {'dim_x': 640,
          'dim_y': 640,
          'dim_z': 3,
          'batch_size': 5,
          'shuffle': True}

training_generator = DataGenerator(**params).generate(labels, partition['train'])

kz = training_generator.__next__()

img_A = kz[0][0, 0, :]
img_B = kz[0][0, 1, :]

plot_image = np.concatenate((img_A, img_B), axis=1)

plt.imshow(plot_image)
