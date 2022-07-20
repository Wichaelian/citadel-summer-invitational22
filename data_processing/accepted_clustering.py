import data_cleaning_utils as dcu
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

"""
Clustering function that takes a list of feature (column) names and returns a modified dataset
that clusters data points based on the strongest K-Means clustering from the set of features,
evaluated by the silhouette method.

Args:
    col_names - The list of column names the clustering should be applied to (Minimum two fields)
    source_path - The source to a csv file that has the same columns as "Lending_Club_Accepted_2014_2018.csv"
    dest_path - The destination path for the modified csv file
    numeric_cols , categorical_cols - Refer to the documentation for data_cleaning_utils
"""
def cluster_create(col_names, source_path, dest_path, numeric_cols = [], categorical_cols = []):
    raw_df = pd.read_csv(source_path)

    # Clean raw csv
    clean_df = dcu.clean_accepted_df(raw_df, numeric_cols, categorical_cols)

    strongest_val = (0,-2 , -1, -1)

    # Iterating on provided columns to find the strongest clustering based on the silhouette method of ranking
    for i in range(len(col_names)):
        for j in range(i+1,len(col_names)):
            for k in range(2, 9):
                kmeans = KMeans(n_clusters=k).fit(clean_df[[col_names[i], col_names[j]]])
                score = silhouette_score(clean_df[[col_names[i], col_names[j]]], kmeans.labels_)
                if score > strongest_val[1]:
                    strongest_val = (k, score, i , j)

    # Recalculating the strongest clustering in lieu of storing all of them due to memory constraints
    kmeans = KMeans(n_clusters=strongestVal[0]).fit(clean_df[[col_names[strongest_val[2]], col_names[strongest_val[3]]]])
    clean_df['cluster'] = kmeans.labels_


    clean_df.to_csv(dest_path)

    print("Created dataset at: " + dest_path + " with a silhouette score of " +\
     str(strongest_val[1]) + " between columns: " + colNames[2] + " and " +\
      colNames[3] + " at a k-value of:" + str(strongest_val[0]))


#cluster_create(["loan_amnt", "last_fico_range_high"], "test_files/sample_by_loan_amt.csv", "test_files/testClustering.csv")
