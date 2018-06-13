import os
import csv

baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"
correctWealthyDir = os.path.join(baseDir,"roads_dictionnary_loubna.csv")

trainDir = os.path.join(baseDir,"train/train.csv")
validationDir = os.path.join(baseDir,"validation/validation.csv")
testDir = os.path.join(baseDir,"test/test.csv")

lines = []


def write(path, data):
    with open(path, 'w') as csvfileWriter:
        writer = csv.writer(csvfileWriter, delimiter=',')
        for line in data:
            writer.writerow(line)


with open(correctWealthyDir, 'r') as csvfileReader:
    reader = csv.reader(csvfileReader, delimiter=',')
    for row in reader:
        if row != []:
            lines.append([row[0],row[1],row[2]])


train_set = []
validation_set = []
test_set = []

for i in range(int(len(lines)*0.75)):
    train_set.append(lines[i])

for i in range(int(len(lines)*0.75),int(len(lines)*0.85)):
    validation_set.append(lines[i])

for i in range(int(len(lines)*0.85),len(lines)):
    test_set.append(lines[i])


write(trainDir,train_set)
write(validationDir,validation_set)
write(testDir,test_set)

