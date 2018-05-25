import os
import csv

baseDir = "C:/Users/msawada/Desktop/arnaud/croutinet/placePulse/data"
correctWealthyDir = os.path.join(baseDir,"votesCorrectWealthy.csv")

trainDir = os.path.join(baseDir,"train/train.csv")
validationDir = os.path.join(baseDir,"validation/validation.csv")
testDir = os.path.join(baseDir,"test/test.csv")

lines = []

encode_dictionnary = { 'left':0, 'right':1}

def write(path, data):
    with open(path, 'w') as csvfileWriter:
        writer = csv.writer(csvfileWriter, delimiter=',')
        for line in data:
            writer.writerow(line)


with open(correctWealthyDir, 'r') as csvfileReader:
    reader = csv.reader(csvfileReader, delimiter=',')
    for row in reader:
        if row != [] and row[2] != 'equal':
            lines.append([row[0],row[1],row[2]])

for line in lines:
    line[2] = encode_dictionnary[line[2]]

train_set = []
validation_set = []
test_set = []

for i in range(int(len(lines)*0.75)):
    train_set.append(lines[i])

for i in range(int(len(lines)*0.75),int(len(lines)*0.80)):
    validation_set.append(lines[i])

for i in range(int(len(lines)*0.80),len(lines)):
    test_set.append(lines[i])


write(trainDir,train_set)
write(validationDir,validation_set)
write(testDir,test_set)

