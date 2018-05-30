import numpy as np
import pandas as pd
import os

base_dir = r"D:\Arnaud\croutinet\ottawa\data"
duelsfinal2018_dir = os.path.join(base_dir,"duelsfinal2018.csv")
loubna_roads_dir = os.path.join(base_dir, "roads_loubna")
loubna_dictionnary_dir = os.path.join(base_dir, "roads_dictionnary_loubna.csv")

all_results = np.loadtxt(duelsfinal2018_dir, str, delimiter=',')

duelsDF = pd.DataFrame(all_results, None, ['id1', 'id2', 'pano1', 'pano2', 'winner'])

duelsDF.drop('id1', axis=1)
duelsDF.drop('id2', axis=1)

images = [f for f in os.listdir(loubna_roads_dir)]
print(images)

pano1Change = False
pano2Change = False
indexToDrop = []

for i in range(len(duelsDF) - 1):
    print(i, indexToDrop)
    for image in images:

        if  duelsDF.iloc[i]['pano1'] in image:
            duelsDF.iloc[i]['pano1'] = image
            pano1Change = True

        if  duelsDF.iloc[i]['pano2'] in image:
            duelsDF.iloc[i]['pano2'] = image
            pano2Change = True

    if not(pano1Change) or not(pano2Change):
        indexToDrop.append(i)

duelsDF.drop(indexToDrop)

list_pano1 = duelsDF['pano1'].tolist()
list_pano2 = duelsDF['pano2'].tolist()
list_winner = duelsDF['winner'].tolist()

dictLeftRight = {'left':0, 'right':1}

with open(loubna_dictionnary_dir, 'a') as f:
    for i in range(len(list_pano1)):
        if list_winner[i] != 'none':
            f.write("{},{},{}\n".format(list_pano1[i], list_pano2[i], dictLeftRight[list_winner[i]]))