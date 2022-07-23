import csv
import sys
import os
import random
import argparse



def partitioner(desired_size, name_of_file = None, prefix = ""):
    DATA_PATH = os.path.join((os.getcwd()), prefix, 'data\\') # Assuming the raw files are in the top-level of the data directory
    DATA_PROCESSING_PATH = os.path.join((os.getcwd()), prefix, 'data_processing\\', 'test_files\\')

    # Correctly formatting input/output path
    source_path = f"{DATA_PATH}Lending_Club_Accepted_2014_2018.csv"
    print(str(source_path))

    if name_of_file is None:
        dest_path = f"{DATA_PROCESSING_PATH}partition_subgrades_{desired_size}.csv"
    else:
        dest_path = f"{DATA_PROCESSING_PATH}{name_of_file}.csv"

    # Initializing variables
    min_subgrade_size = sys.maxsize
    subgrade_index_map = {}
    write_indicies = []
    sub_grade_relative_indicies = {}
    chosen_indicies = {}
    sub_grade_index = -1

    # Read over the raw file to get information on subgrades
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

        sub_grade_quote = int(desired_size)

        if min_subgrade_size < int(desired_size):
            print("Desired size is too big, using min_subgrade_size")
            sub_grade_quote = min_subgrade_size

        generated_indicies = random.sample(range(0, min_subgrade_size), sub_grade_quote)
        for index in generated_indicies:
            chosen_indicies[index] = 1

    # Apply learned knowledge of subgrades and partition raw file to new reduced file with equal subgrade representation
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
    parser = argparse.ArgumentParser(description='Partitioning the provided ' +\
    'dataset, that has the same columns as "Lending_Club_Accepted_2014_2018.csv"' +\
    ' into a set where all subgrades have identical representation. Should be' +\
    ' called from the root of the repository')
    parser.add_argument('--desired_size', help='Number of elements per subgrade.'+\
    'Note the actual size of each subgrade will be Min(min_subgrade_size, desired_size)', required=True)
    parser.add_argument('--name_of_file', help='Name of output file to be stored in data_processing/test_files/ without the .csv suffix', required = False)

    args = parser.parse_args()

    partitioner(args.desired_size, args.name_of_file)
