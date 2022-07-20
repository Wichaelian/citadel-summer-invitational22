import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import random


def getDistIndex(curVal, minVal, val_granularity):
    return math.floor((float(curVal) - minVal -1)/max(val_granularity,1))


def minMax(field_index, source_path):
    minVal, maxVal = sys.maxsize, 0
    passedFirst = False
    with open(source_path, encoding='utf-8') as csv_input:
        csv_reader = csv.reader(csv_input, delimiter=',')
        for row in csv_reader:
            if not passedFirst or len(row) <= field_index:
                passedFirst = True
                continue
            curVal = row[field_index]
            if len(curVal)>0:
                minVal, maxVal = min(minVal, float(curVal)), max(maxVal, float(curVal))
    return minVal, maxVal

def getDistributionByField(field_index, granularity, source_path):
    minVal, maxVal = minMax(field_index, source_path)

    val_granularity = (maxVal - minVal)/granularity
    totalCount = 0
    buckets = [0]*bucket_granularity

    passedFirst = False

    with open(source_path, encoding='utf-8') as csv_input:
        csv_reader = csv.reader(csv_input, delimiter=',')
        for row in csv_reader:
            if not passedFirst or len(row) <= field_index:
                passedFirst = True
                continue
            curVal = row[field_index]
            if len(curVal)>0:
                buckets[getDistIndex(curVal, minVal, val_granularity)]+=1
                totalCount += 1

    buckets = [curVal/totalCount for curVal in buckets]

    return minVal, maxVal, totalCount, buckets


def getApproximateDistributionByField(field_index, granularity, source_path):
    minVal, maxVal = sys.maxsize, 0
    passedFirst = False
    totalCount = 0
    buckets = [0]*bucket_granularity

    with open(source_path, encoding='utf-8') as csv_input:
        csv_reader = csv.reader(csv_input, delimiter=',')
        for row in csv_reader:
            if not passedFirst or len(row) <= field_index:
                passedFirst = True
                continue
            curVal = row[field_index]
            if len(curVal)>0:
                minVal, maxVal = min(minVal, float(curVal)), max(maxVal, float(curVal))
                val_granularity = (maxVal - minVal)/granularity
                buckets[getDistIndex(curVal, minVal, val_granularity)]+=1
                totalCount += 1

    buckets = [curVal/totalCount for curVal in buckets]

    return minVal, maxVal, totalCount, buckets

def create_sample(field_index, granularity, reduction, source_path, dest_path, selector, minVal, maxVal, totalSum, orgDistribution):
    #minVal, maxVal = minMax(field_index, source_path)
    cachedVals = [[]] * granularity
    val_granularity = (maxVal - minVal)/granularity

    with open(source_path, encoding='utf-8') as csv_input:
        with open(dest_path, 'w', encoding='utf-8') as csv_output:
            csv_reader = csv.reader(csv_input, delimiter=',')
            csv_writer = csv.writer(csv_output)
            passedFirst = False

            for row in csv_reader:
                if not passedFirst or len(row) <= field_index:
                    passedFirst = True
                    continue
                curVal = row[field_index]
                if len(curVal) > 0:
                    prcsd_index = getDistIndex(curVal, minVal, val_granularity)
                    cachedVals[prcsd_index].append(row)
                    if len(cachedVals[prcsd_index]) > reduction - 1:
                        picked_val = selector(len(cachedVals[prcsd_index]) - 1)
                        csv_writer.writerow(cachedVals[prcsd_index][picked_val])
                        cachedVals[prcsd_index] = []

            # Sample the remaining unsampled lots
            for i in range(granularity):
                if len(cachedVals[prcsd_index]) > 0:
                    picked_val = selector(len(cachedVals[prcsd_index]) - 1)
                    csv_writer.writerow(cachedVals[prcsd_index][picked_val])
                    cachedVals[prcsd_index] = []



"""
# Exploratory Code - To get a feel for the data and analyze some features
with open('../rawFiles/Lending_Club_Accepted_2014_2018.csv', encoding='utf-8') as csv_input:
    with open('test.csv','w') as csv_output:
        csv_reader = csv.reader(csv_input, delimiter=',')
        i = 0
        colCount = []
        colNames = []

        minVal, maxVal = 1000, 40000
        val_granularity = (maxVal - minVal)/bucket_granularity

        totalVal = 0
        buckets = [0]*bucket_granularity

        for row in csv_reader:
            if i > 500000:
                break
            for j,val in enumerate(row):
                if i == 0:
                    colCount.append(0)
                    colNames.append(val)
                else:
                    colCount[j] += (1 if len(val) > 0 else 0)
                    if j==3 and len(val)>0:
                        buckets[math.floor((float(val) - minVal -1)/max(val_granularity,1))]+=1
                        totalVal+=1
            row[j] = row[j]
            i+=1

buckets = [buckets[i]/totalVal for i in range(len(buckets))]
"""
"""
bucket_granularity = 100
#minVal, maxVal, totalSum, orgDist = getDistributionByField(3, bucket_granularity, '../rawFiles/Lending_Club_Accepted_2014_2018.csv')
minVal = 1000.0
maxVal = 40000.0
totalSum = 2029952
orgDist = [0.02819820370136831, 0.06055561904912037, 0.10178319487357336,
0.08521630068100132, 0.12323000740904218, 0.08805528406583013,
0.04805975707799987, 0.10008758827794943, 0.044087741976164954,
0.0767604357147361, 0.02749424616936755, 0.03656342613027303,
0.039360536603821175, 0.02244486569140551, 0.0345372698467747,
0.011100755091746012, 0.00552525379910461, 0.043743398858692224,
0.0014468322403682452, 0.021749282741660887]

randomChoice = lambda x: random.randrange(x)
systemicChoice = lambda x: x-1

create_sample(3, bucket_granularity, 1000, '../rawFiles/Lending_Club_Accepted_2014_2018.csv', 'test.csv', systemicChoice, minVal, maxVal, totalSum, orgDist)
_, _, totalVal, newDist = getDistributionByField(3, bucket_granularity, 'test.csv')

print(totalVal)

x_coords = np.arange(bucket_granularity)
#plt.bar(x_coords-0.2, orgDist, 0.4, align = 'center', label = 'org')
plt.bar(x_coords, newDist, align = 'center', label = 'new')
plt.show()

"""
"""
with open('../rawFiles/Lending_Club_Accepted_2014_2018.csv', encoding='utf-8') as csv_input:
    with open('test.csv','w') as csv_output:
        csv_reader = csv.reader(csv_input, delimiter=',')
        i = 0
        colCount = []
        colNames = []

        bucket_granularity = 100

        minVal = sys.maxsize
        maxVal = 0
        totalVal = 0
        buckets = [0]*bucket_granularity

        for row in csv_reader:
            if i > 500000:
                break
            for j,val in enumerate(row):
                if i == 0:
                    colCount.append(0)
                    colNames.append(val)
                else:
                    colCount[j] += (1 if len(val) > 0 else 0)
                    if j==3 and len(val)>0:
                        minVal, maxVal = min(minVal, float(val)), max(maxVal, float(val))
                        val_granularity = (maxVal - minVal)/bucket_granularity
                        buckets[math.floor((float(val) - minVal -1)/max(val_granularity,0.5)]+=1
                        totalVal+=1
            i+=1

buckets = [buckets[i]/totalVal for i in range(len(buckets))]

x_coords = np.arange(bucket_granularity)
plt.bar(x_coords, buckets, align = 'center')
plt.show()
"""

"""

rawDF = pd.read_csv("test_files/sample_by_loan_amt.csv")
cleanDF = dcu.clean_accepted_df(rawDF, [], [])

candidateRows = []

# Excluding Fields with a large number of unknown values
for col in cleanDF:
    if cleanDF[col].isna().sum() < len(cleanDF[col])/2:
        candidateRows.append(col)


strongestVal = (0,-2)

for k in range(1, 9):
    kmeans = KMeans(n_clusters=k).fit(cleanDF[['funded_amnt', 'last_fico_range_high']])
    score = silhouette_score(cleanDF[['funded_amnt', 'last_fico_range_high']], kmeans.labels_)
    if score > strongestVal[1]:
        strongestVal = (k, score)

if (k == 1):
    print("Clustering doesn't have meaningful differences")

print(strongestVal)

'''
centroids = kmeans.cluster_centers_

plt.scatter(cleanDF['funded_amnt'], cleanDF['last_fico_range_high'], s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

for i in range(len(candidateRows)):
    for j in range(i+1, len(candidateRows)):
        pass

'test_files/sample_by_loan_amt.csv'

"""
