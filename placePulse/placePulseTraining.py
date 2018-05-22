import os
import csv

baseDir = "C:/Users/msawada/Desktop/arnaud/croutinet/placePulse/"
correctWealthyDir = os.path.join(baseDir,"votesCorrectWealthy.csv")

with open(correctWealthyDir, 'r') as csvfileReader:
    reader = csv.reader(csvfileReader, delimiter=',')
    for row in reader:
        print(row)

