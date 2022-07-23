import csv
import sys
import os
import random

def partitioner(desired_size, source_path, dest_path):
    min_subgrade_size = sys.maxsize
    subgrade_index_map = {}
    write_indicies = []
    sub_grade_relative_indicies = {}
    chosen_indicies = {}
    sub_grade_index = -1

    with open(source_path, encoding='utf-8') as csv_input:
        csv_reader = csv.reader(csv_input, delimiter=',')
        passedFirst = False
        i = 0

        # Create a map that maps each subgrade to indicies
        for row in csv_reader:
            if not passedFirst:
                sub_grade_index = [j for j in range(len(row)) if row[j] == 'sub_grade'][0]
                passedFirst = True
            else:
                if not row[sub_grade_index] in subgrade_index_map:
                    subgrade_index_map[row[sub_grade_index]] = [i]
                else:
                    subgrade_index_map[row[sub_grade_index]].append(i)
            i+=1

        for key in subgrade_index_map:
            min_subgrade_size = min(len(subgrade_index_map[key]), min_subgrade_size)
            sub_grade_relative_indicies[key] = 0

        sub_grade_quote = int(desired_size/len(subgrade_index_map))

        if len(subgrade_index_map) * min_subgrade_size < desired_size:
            print("Desired size is too big, using number_of_subgrades * min_subgrade_size")
            sub_grade_quote = min_subgrade_size

        generated_indicies = random.sample(range(0, min_subgrade_size), sub_grade_quote)
        for index in generated_indicies:
            chosen_indicies[index] = 1

    with open(source_path, encoding='utf-8') as csv_input:
        with open(dest_path, 'w', encoding='utf-8') as csv_output:
            csv_writer = csv.writer(csv_output)
            csv_reader = csv.reader(csv_input, delimiter=',')
            passedFirst = False
            for row in csv_reader:
                if not passedFirst:
                    csv_writer.writerow(row)
                    passedFirst = True
                    continue
                if sub_grade_relative_indicies[row[sub_grade_index]] in chosen_indicies:
                    csv_writer.writerow(row)
                sub_grade_relative_indicies[row[sub_grade_index]] += 1

    print("The raw data has " + str(len(subgrade_index_map)) + " subgrades" +\
    " with the smallest subgrade being of size " + str(min_subgrade_size) + ".")


if __name__ == '__main__' :
    #parser = argparse.ArgumentParser(description='Partitioning the provided ' +\
    #'dataset, that has the same columns as "Lending_Club_Accepted_2014_2018.csv"' +\
    #' into a set where all subgrades have identical representation')
    #parser.add_argument('--desired_size', help='Number of elements in the'+\
    #' reduced set. Note the actual size of the data set will be '+\
    #'min(size_of_smallest_subgrade * number_of_subgrades, desired_size)', required=True)
    #parser.add_argument('--source_path', help='Relative path from this script to the original dataset', required=True)
    #parser.add_argument('--dest_path', help='Relative path from this script to the destination for the newly sampled dataset', required=True)

    #args = parser.parse_args()

    #partitioner(args.desired_size, args.source_path, args.dest_path)
    partitioner(1000, "../../../citadel-summer-invitational22/rawFiles/Lending_Club_Accepted_2014_2018.csv", "test_files/partition_subgrades_1000.csv")


    # JANK STUFF I HAD TO DO...
    # sys.path.insert(0, "../")
    # DATA_PATH = os.path.join((os.getcwd()), 'data/')
    # DATA_PROCESSING_PATH = os.path.join((os.getcwd()), 'data_processing/') 
    # desired_size = 30000
    # partitioner(desired_size, f"{DATA_PATH}Lending_Club_Accepted_2014_2018.csv", f"{DATA_PROCESSING_PATH}/test_files/partition_subgrades_{desired_size}.csv")
