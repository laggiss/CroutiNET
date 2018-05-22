import csv
import  os

baseDir = "C:/Users/msawada/Desktop/arnaud/croutinet/placePulse/"
datasetDir = os.path.join(baseDir,"votes.csv")
wealthyDir = os.path.join(baseDir,"votesWealthy.csv")

# select wealthy limes and wrote them in votesWealthy.csv


with open(datasetDir, 'r') as csvfileReader:
    with open(wealthyDir, 'w') as csvfileWriter:
        reader = csv.reader(csvfileReader, delimiter=',')
        writer = csv.writer(csvfileWriter, delimiter=',')
        i=0
        for row in reader:
            if row[7] == 'wealthy':
                writer.writerow(row)
                i+=1
print(i)

