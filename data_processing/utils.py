from typing import List, Tuple
import sklearn
import zipcodes
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import datetime
import scipy
from sklearn.preprocessing import StandardScaler
import sys

def clean_accepted_df(accepted_df: pd.DataFrame, numeric_cols: List[str],
        categorical_cols: List[str], one_hot_threshold: int = 30, transform_zips = True) -> Tuple[pd.DataFrame, List[str], List[str]] :
    """Utility function that takes in a dataframe with the same columns as a "Lending_Club_Accepted_2014_2018.csv" and
        applies various pre-processing stages to handle null values, encode categorical variables, and retrieve longitude information from zipcodes.
        Even though we don't end up using many of these features, we produce a DataFrame that can be used for any later analyses with ease. 

    Args:
        accepted_df (pd.DataFrame): A dataframe with the same columns as "Lending_Club_Accepted_2014_2018.csv"
        numerical_cols (List[str]): A list of strings containing the names of numerical columns whose null values are to be imputed
        besides those already modified (zip code is to be included here)
        categorical_cols (List[str]): A list of strings containing the names of categorical columns that are to be encoded using one hot
        encoding if the number of unique values is less than or equal to one_hot_threshold, and label encoded otherwise.
        one_hot_threshold (int): Number of unique vals in categorical column above which labels are label encoded

    Returns:
        accepted_df (pd.DataFrame): An in-place cleaned version of the input dataframe
        numeric_cols_out (List[str]): Names of numerical columns retained from original numerical_cols
        categorical_cols_out (List[str]): Names of categorical columns retained from original categorical_cols
    """
    numeric_cols_out = set(numeric_cols)
    categorical_cols_out = set(categorical_cols)

    cached_state_dtis = {}
    def replace_dti(dti, state):
        if dti != np.nan and dti <= 90:
            return dti
        elif state not in cached_state_dtis:
            cached_state_dtis[state] = accepted_df[accepted_df["addr_state"] == state]["dti"].mean()
            if cached_state_dtis[state] == np.nan:
                cached_state_dtis[state] == accepted_df["dti"].mean()

        return cached_state_dtis[state]

    # Replace no DTI with average DTI of corresponding state
    # Replace DTI > 90 as there seem to be spurious values in the dataset like 9999
    accepted_df["dti"] = accepted_df.apply(lambda x: replace_dti(x["dti"], x["addr_state"]), axis = 1)

    
    emp_to_index = {"< 1 year": 0, "1 year": 1}
    for i in range(2, 11):
        if (i == 10):
            emp_to_index["10+ years"] = 10
        else:
            emp_to_index[f"{i} years"] = i

    # Replace no emp_length with < 1 year, convert all to ints
    accepted_df["emp_length"] = accepted_df["emp_length"].fillna(value="< 1 year")
    accepted_df["emp_length"] = accepted_df["emp_length"].apply(lambda x: emp_to_index[x])

    # Replace no emp_title with "No Employment"
    accepted_df["emp_title"] = accepted_df["emp_title"].fillna(value="No Employment")
    accepted_df["emp_title"] = accepted_df["emp_title"].astype(str)

    # Convert zip code to longitude latitude estimate, use a cache to avoid repeated searching
    # If no matches found, latitude and longitude are set to 1000.0, 1000.0 (out of bounds)
    cached_zips = {}
    def get_coords(zip_code: str, mode: str):
        if (zip_code not in cached_zips):
            cached_zips[zip_code] = [1000.0, 1000.0]
            matches = zipcodes.similar_to(zip_code)
            if (len(matches) > 0):
                cached_zips[zip_code] = [float(matches[0]['lat']), float(matches[0]['long'])]

        return cached_zips[zip_code][0 if mode == "lat" else 1]

    if (transform_zips == True):
        accepted_df["zip_code"] = accepted_df["zip_code"].str.rstrip('xx')
        accepted_df["lat"] = accepted_df["zip_code"].apply(lambda x: get_coords(x, mode = "lat"))
        accepted_df["long"] = accepted_df["zip_code"].apply(lambda x: get_coords(x, mode = "long"))

    accepted_df.drop(["zip_code"], axis = 1, inplace= True)
    numeric_cols_out.remove("zip_code")
    
    # Replace date fields with days since 01-01-2014
    # Replace missing values in mths_since cols with max value of column
    # Replace all other missing numerical fields with 0 (fields like "total balance of installment accounts"
    # which are nan if the borrower has no other installment accounts)

    for num_col in numeric_cols_out:
        if (num_col[-2:]) == "_d":
            accepted_df[num_col] = (pd.to_datetime(accepted_df[num_col]) - datetime.datetime(2014,1,1)).dt.days.astype('float64')
        elif ("mths_since" in num_col):
            accepted_df[num_col] = accepted_df[num_col].fillna(value=accepted_df[num_col].max() + 1)

        accepted_df[num_col] = accepted_df[num_col].fillna(value=accepted_df[num_col].median())

    # Encode categorical vars using one hot if < 10 vars, else label encode (force label encoding on loan_status)
    # For the state column, we label encode using the state's GDP per capita

    state_to_index = {"DC": 1, "NY": 2, "MA": 3, "WA": 4, "CA": 5, "CT": 6, "ND": 7, "DE": 8, "NE": 9, "AK": 10,
                      "IL": 11, "CO": 12, "NJ": 13, "MN": 14, "WY": 15, "MD": 16, "NH": 17, "IA": 18, "VA": 19, "SD": 20,
                      "TX": 21, "UT": 22, "KS": 23, "PA": 24, "GA": 25, "OR": 26, "OH": 27, "HI": 28, "NC": 29, "WI": 30,
                      "IN": 31, "NV": 32, "RI": 33, "TN": 34, "MO": 35, "MI": 36, "AZ": 37, "FL": 38, "VT": 39, "ME": 40,
                      "LA": 41, "MT": 42, "SC": 43, "KY": 44, "OK": 45, "NM": 46, "ID": 47, "AL": 48, "WV": 49, "AR": 50, "MS": 51}

    accepted_df["addr_state"] = accepted_df["addr_state"].apply(lambda x: state_to_index[x])

    sorted_sub_grades = accepted_df["sub_grade"].unique()
    sorted_sub_grades.sort()
    sub_grade_to_val = {}
    for i, grade in enumerate(sorted_sub_grades):
        sub_grade_to_val[grade] = i

    accepted_df["sub_grade"] = accepted_df["sub_grade"].apply(lambda x: sub_grade_to_val[x])

    for cat_col in categorical_cols:
        if (cat_col == "emp_length" or cat_col == "addr_state"):
            continue
        nuniquecol = accepted_df[cat_col].nunique()
        if 2 < nuniquecol <= one_hot_threshold or cat_col == "loan_status" or cat_col == "sub_grade":
            one_hot = pd.get_dummies(accepted_df[cat_col], prefix = cat_col)
            if (cat_col != "sub_grade"):
                accepted_df = accepted_df.drop(cat_col,axis = 1)
                categorical_cols_out.remove(cat_col)
            accepted_df = accepted_df.join(one_hot)
        else:
            accepted_df[cat_col] = accepted_df[cat_col].astype('category')
            accepted_df[cat_col] = accepted_df[cat_col].cat.codes

    accepted_df.columns = ['_'.join(x.lower().split()) for x in accepted_df.columns]

    return accepted_df, list(numeric_cols_out), list(categorical_cols_out)

"""
Proximity function based on the cumulative distance of points away from their
cluster's center. Meant to prefer denser cluster over more spread out one.
Bias towards more clusters.

Args:
    coords - List of all coordinates of the data points
    labels - List of the labels of the clusters each coordinate belongs to. (One-to-One relationshop with coords)
    k_centers - The coordinates of the center of each cluster
"""
def proximity_fit(coords, labels, k_centers, min_k, max_k):
    totalDistance = 0
    for i in range(len(coords)):
        totalDistance += (scipy.spatial.distance.euclidean(coords[i], k_centers[labels[i]])) * (1 + len((k_centers) - min_k)/max_k)
    return totalDistance

def update_subgrade_score(matching_sub_grades: pd.Series, k: int, threshold = 0.5):
    num_elts = len(matching_sub_grades)
    val_cnts = matching_sub_grades.value_counts()
    total_seen = 0
    for val in val_cnts:
        total_seen += val
        if (total_seen >= num_elts * threshold):
            break
    
    return (total_seen) / (k * num_elts)

def sub_grade_score(cluster, clustered_df: pd.DataFrame):
    clustered_df["cluster"] = cluster.labels_
    num_clusters = clustered_df["cluster"].nunique()
    score = 0
    for k in range(1, num_clusters + 1):
        for i in range(len(clustered_df["sub_grade"].value_counts())):
            matching_sub_grades = clustered_df["cluster"][clustered_df["cluster"] == k][clustered_df["sub_grade"] == i]
            if (len(matching_sub_grades) > 0):
                score += update_subgrade_score(matching_sub_grades, k, threshold = 0.5)
    return score

"""
Function that returns the heatmaps
"""
def model_cluster_density(clustered_df, k_val):
    heat_map = []
    for k in range(k_val):
        heat_map.append([])
        for i in range(len(clustered_df["sub_grade"].value_counts())):
            heat_map[k].append(len(clustered_df["cluster"][clustered_df["cluster"] == k][clustered_df["sub_grade"] == i]))
    return heat_map


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
def cluster_create(col_names, source_path, dest_path,
                    numeric_cols = list(set([
                    "loan_amnt", "funded_amnt", "funded_amnt_inv", "int_rate", "installment", "issue_d", "annual_inc", "dti", "fico_range_low", "fico_range_high",
                    "revol_bal", "revol_util", "open_acc", "zip_code", "delinq_2yrs", "inq_last_6mths", "total_acc", "mths_since_last_delinq", "mths_since_last_record", "mths_since_rcnt_il",
                    "last_credit_pull_d", "open_il_12m", "open_il_24m", "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
                    "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util", "num_accts_ever_120_pd",
                    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_sats", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "tot_hi_cred_lim",
                    "pct_tl_nvr_dlq", "percent_bc_gt_75", "total_bal_ex_mort", "total_bc_limit","total_il_high_credit_limit", "mths_since_last_major_derog", "mths_since_recent_bc",
                    "mths_since_recent_bc_dlq", "mths_since_recent_inq", "mths_since_recent_revol_delinq"
                    ])),
                    categorical_cols = ["term", "grade", "sub_grade", "emp_title", "emp_length", "home_ownership", "verification_status", "purpose", "addr_state",
                    "initial_list_status", "application_type", "hardship_flag", "loan_status"]):
    raw_df = pd.read_csv(source_path)
    raw_df = raw_df[numeric_cols + categorical_cols]
    # Clean raw csv
    clean_df, numeric_cols, categorical_cols = clean_accepted_df(raw_df, numeric_cols, categorical_cols, one_hot_threshold = 30, transform_zips = False)

    strongest_val = (0, sys.maxsize , -1, -1)

    scaler = StandardScaler()

    k_heat_maps = []
    epss = np.linspace(0.2, 2, 10)
    for i in range(len(col_names)):
        for j in range(i+1,len(col_names)):
            scaled_dimensions = scaler.fit_transform(clean_df[[col_names[i], col_names[j]]])
            for k in range(15, 30):
                print(k)
                clustering = KMeans(n_clusters=k).fit(scaled_dimensions)
                score = sub_grade_score(clustering, clean_df)
                if score < strongest_val[1]:
                    strongest_val = (k, score, i , j)

    # Recalculating the strongest clustering in lieu of storing all of them due to memory constraints
    clustering = KMeans(n_clusters=strongest_val[0]).fit(clean_df[[col_names[strongest_val[2]], col_names[strongest_val[3]]]])
    
    clean_df['cluster'] = clustering.labels_

    clean_df.to_csv(dest_path)
    clean_df['cluster'] = clustering.labels_

    k_heat_maps = (model_cluster_density(clean_df,strongest_val[0]))
    uniq_vals = len(clean_df["sub_grade"].value_counts())
    x_axis = np.arange(0,uniq_vals)
    _, axis = plt.subplots(1, strongest_val[0])
    for j in range(len(k_heat_maps)):
        axis[j].bar(x_axis,k_heat_maps[j])
        axis[j].set_title('Distribution of sub-grades in the ' + str(j) +"th cluster")
    plt.show()

    print("Created dataset at: " + dest_path + " with a proximity score of " +\
     str(strongest_val[1]) + " between columns: " + col_names[strongest_val[2]] + " and " +\
      col_names[strongest_val[3]] + " at a k-value of:" + str(strongest_val[0]))

    return clean_df

#cluster_create(["loan_amnt", "fico_range_high"], "test_files/cluster_micro_sample.csv", "test_files/testClustering.csv")
