import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import random
import argparse


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

def create_sample(field_index, granularity, reduction, source_path, dest_path, selector):
    minVal, maxVal = minMax(field_index, source_path)
    cachedVals = [[]] * granularity
    val_granularity = (maxVal - minVal)/granularity

    with open(source_path, encoding='utf-8') as csv_input:
        with open(dest_path, 'w', encoding='utf-8') as csv_output:
            csv_reader = csv.reader(csv_input, delimiter=',')
            csv_writer = csv.writer(csv_output)
            passedFirst = False

            for row in csv_reader:
                if not passedFirst:
                    csv_writer.writerow(row)
                    passedFirst = True
                    continue

                if len(row) <= field_index:
                    continue
                curVal = row[field_index]
                if len(curVal) > 0:
                    prcsd_index = getDistIndex(curVal, minVal, val_granularity)
                    cachedVals[prcsd_index].append(row)
                    if len(cachedVals[prcsd_index]) > reduction - 1:
                        picked_val = selector(len(cachedVals[prcsd_index]))
                        csv_writer.writerow(cachedVals[prcsd_index][picked_val])
                        cachedVals[prcsd_index] = []

            # Sample the remaining unsampled lots
            for i in range(granularity):
                if len(cachedVals[prcsd_index]) > 0:
                    picked_val = selector(len(cachedVals[prcsd_index]))
                    csv_writer.writerow(cachedVals[prcsd_index][picked_val])
                    cachedVals[prcsd_index] = []


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Sampling the provided file by provided field index. (i.e. python sampler.py --field_index 3'+\
    ' --granularity 50 --reduction 50 --source_path "../rawFiles/Lending_Club_Rejected_2014_2018.csv" --dest_path "test.csv")')
    parser.add_argument('--field_index', help='Index of the NUMERIC field whose distribution will form the sampling distribution', required=True)
    parser.add_argument('--granularity', help='Number of values to form the discontinuous sampling distribution (Higher number means higher accuracy)', required=True)
    parser.add_argument('--reduction', help='Factor of reduction of original data set (i.e. 2 means the original dataset is reduced by half)', required=True)
    parser.add_argument('--source_path', help='Relative path from this script to the original dataset', required=True)
    parser.add_argument('--dest_path', help='Relative path from this script to the destination for the newly sampled dataset', required=True)
    #parser.add_argument('--random', help = 'The sampling within a granularity be systemic or random, default is systemic', action='store_const', const=42)

    args = parser.parse_args()

    selector = lambda x: random.randrange(x) #if args.random is not None else (lambda x: int(x-1))

    create_sample(int(args.field_index), int(args.granularity), int(args.reduction), args.source_path, args.dest_path, selector)
