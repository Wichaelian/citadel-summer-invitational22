from typing import List, Tuple
import zipcodes
import pandas as pd
import numpy as np
import datetime

def clean_accepted_df(accepted_df: pd.DataFrame, numeric_cols: List[str], 
        categorical_cols: List[str], one_hot_threshold: int = 30) -> Tuple[pd.DataFrame, List[str], List[str]] : 
    """Utility function that takes in a dataframe with the same columns as a "Lending_Club_Accepted_2014_2018.csv" and 
    applies various pre-processing
        stages to handle null values, encode categorical variables, and retrieve longitude information from zipcodes.

    Args:
        accepted_df (pd.DataFrame): A dataframe with the same columns as "Lending_Club_Accepted_2014_2018.csv"
        numerical_cols (List[str]): A list of strings containing the names of numerical columns whose null values are to be median imputed,
        besides those already modified (zip code included here)
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
    # Also replace DTI > 90 as there seem to be spurious values in the dataset
    accepted_df["dti"] = accepted_df.apply(lambda x: replace_dti(x["dti"], x["addr_state"]), axis = 1)

    # Replace no emp_length with < 1 year

    emp_to_index = {"< 1 year": 0, "1 year": 1}
    for i in range(2, 11):
        if (i == 10):
            emp_to_index["10+ years"] = 10
        else:
            emp_to_index[f"{i} years"] = i
 
    accepted_df["emp_length"] = accepted_df["emp_length"].fillna(value="< 1 year")
    accepted_df["emp_length"] = accepted_df["emp_length"].apply(lambda x: emp_to_index[x])

    # Replace no emp_title with "No Employment"
    accepted_df["emp_title"] = accepted_df["emp_title"].fillna(value="No Employment")
    accepted_df["emp_title"] = accepted_df["emp_title"].astype(str)

    cached_zips = {}
    # Convert zip code to longitude latitude estimate, use a cache to avoid repeated searching
    # If no matches found, latitude and longitude are set to 1000.0, 1000.0 (out of bounds)
    def get_coords(zip_code: str, mode: str):
        if (zip_code not in cached_zips):
            cached_zips[zip_code] = [1000.0, 1000.0]
            matches = zipcodes.similar_to(zip_code)
            if (len(matches) > 0):
                cached_zips[zip_code] = [float(matches[0]['lat']), float(matches[0]['long'])]

        return cached_zips[zip_code][0 if mode == "lat" else 1]

    accepted_df["zip_code"] = accepted_df["zip_code"].str.rstrip('xx')
    accepted_df["lat"] = accepted_df["zip_code"].apply(lambda x: get_coords(x, mode = "lat"))
    accepted_df["long"] = accepted_df["zip_code"].apply(lambda x: get_coords(x, mode = "long"))
      
    # Replace date fields with days since 01-01-2014
    # Replace missing values in mths_since cols with max value of column*10
    # Replace all other missing numerical fields with 0 (these are fields like "total balance of installment accounts" 
    # which are nan if the borrower has no other installment accounts)

    for num_col in numeric_cols_out:
        if (num_col[-2:]) == "_d":
            accepted_df[num_col] = (pd.to_datetime(accepted_df[num_col]) - datetime.datetime(2014,1,1)).dt.days.astype('float64')
        elif ("mths_since" in num_col):
            accepted_df[num_col] = accepted_df[num_col].fillna(value=accepted_df[num_col].max()*10)
        
        accepted_df[num_col] = accepted_df[num_col].fillna(value=accepted_df[num_col].median())

    # Encode categorical vars using one hot if < 10 vars, else label encode (force label encoding on loan_status)
    # For the state column, we label encode using the state's GDP per capita

    state_to_index = {"DC": 1, "NY": 2, "MA": 3, "WA": 4, "CA": 5, "CT": 6, "ND": 7, "DE": 8, "NE": 9, "AK": 10, 
                      "IL": 11, "CO": 12, "NJ": 13, "MN": 14, "WY": 15, "MD": 16, "NH": 17, "IA": 18, "VA": 19, "SD": 20,
                      "TX": 21, "UT": 22, "KS": 23, "PA": 24, "GA": 25, "OR": 26, "OH": 27, "HI": 28, "NC": 29, "WI": 30,
                      "IN": 31, "NV": 32, "RI": 33, "TN": 34, "MO": 35, "MI": 36, "AZ": 37, "FL": 38, "VT": 39, "ME": 40,
                      "LA": 41, "MT": 42, "SC": 43, "KY": 44, "OK": 45, "NM": 46, "ID": 47, "AL": 48, "WV": 49, "AR": 50, "MS": 51}

    accepted_df["addr_state"] = accepted_df["addr_state"].apply(lambda x: state_to_index[x])

    for cat_col in categorical_cols:
        if (cat_col == "emp_length" or cat_col == "addr_state"):
            continue
        nuniquecol = accepted_df[cat_col].nunique()
        if nuniquecol <= one_hot_threshold or cat_col == "loan_status":
            one_hot = pd.get_dummies(accepted_df[cat_col], prefix = cat_col)
            accepted_df = accepted_df.drop(cat_col,axis = 1)
            categorical_cols_out.remove(cat_col)
            accepted_df = accepted_df.join(one_hot)
        else:
            accepted_df[cat_col] = accepted_df[cat_col].astype('category')
            accepted_df[cat_col] = accepted_df[cat_col].cat.codes

    accepted_df.drop(["zip_code"], axis = 1, inplace= True)
    numeric_cols_out.remove("zip_code")

    accepted_df.columns = ['_'.join(x.lower().split()) for x in accepted_df.columns]

    return accepted_df, list(numeric_cols_out), list(categorical_cols_out)
