from typing import List
import zipcodes
import pandas as pd
import numpy as np
import datetime as dt

def clean_accepted_df(accepted_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], one_hot_threshold: int = 10) -> pd.DataFrame:
    """Utility function that takes in a dataframe with the same columns as a "Lending_Club_Accepted_2014_2018.csv" and
    applies various pre-processing
        stages to handle null values, encode categorical variables, and retrieve longitude information from zipcodes.

    Args:
        accepted_df (pd.DataFrame): A dataframe with the same columns as "Lending_Club_Accepted_2014_2018.csv"
        numerical_cols (List[str]): A list of strings containing the names of numerical columns whose null values are to be median imputed,
        besides those already modified
        categorical_cols (List[str]): A list of strings containing the names of categorical columns that are to be encoded using one hot
        encoding if the number of unique values is less than or equal to one_hot_threshold, and label encoded otherwise.
        one_hot_threshold (int): Number of unique vals in categorical column above which labels are label encoded

    Returns:
        pd.DataFrame: An in-place cleaned version of the input dataframe
    """
    # Replace no DTI with average DTI -- should use something better here...
    accepted_df["dti"] = accepted_df["dti"].fillna(value=accepted_df["dti"].mean())
    accepted_df["dti"] = accepted_df["dti"].astype('float64')

    # Replace no description with "No Description"
    accepted_df["desc"] = accepted_df["desc"].fillna(value="No Description")
    accepted_df["desc"] = accepted_df["desc"].astype(str)

    # Replace no emp_length with No Record
    accepted_df["emp_length"] = accepted_df["desc"].fillna(value="No Record")

    # Replace no emp_title with "No Employment"
    accepted_df["emp_title"] = accepted_df["emp_title"].fillna(value="No Employment")
    accepted_df["emp_title"] = accepted_df["emp_title"].astype(str)

    # Replace no hardship_type with "No Hardship"
    accepted_df["hardship_type"] = accepted_df["hardship_type"].fillna(value="No Hardship")
    accepted_df["hardship_type"] = accepted_df["hardship_type"].astype(str)

    # Replace no hardship_reason with "No Hardship"
    accepted_df["hardship_reason"] = accepted_df["hardship_reason"].fillna(value="No Hardship")
    accepted_df["hardship_reason"] = accepted_df["hardship_reason"].astype(str)

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


    # Replace all other missing numerical fields with median
    for num_col in numeric_cols:
        accepted_df[num_col] = accepted_df[num_col].fillna(value=accepted_df[num_col].median())

    # Encode categorical vars using one hot if < 10 vars, else label encode (force label encoding on loan_status)
    for cat_col in categorical_cols:
        nuniquecol = accepted_df[cat_col].nunique()
        if nuniquecol <= one_hot_threshold or cat_col == "loan_status":
            one_hot = pd.get_dummies(accepted_df[cat_col], prefix = cat_col)
            accepted_df = accepted_df.drop(cat_col,axis = 1)
            accepted_df = accepted_df.join(one_hot)
        else:
            accepted_df[cat_col] = accepted_df[cat_col].astype('category')
            accepted_df[cat_col] = accepted_df[cat_col].cat.codes

    accepted_df.drop(["zip_code"], axis = 1, inplace= True)
    accepted_df.columns = ['_'.join(x.lower().split()) for x in accepted_df.columns]


    # Convert any date values to days since 01 January 2014

    # Get every date column
    date_cols = [col for col in accepted_df.columns if col.endswith("_d") or col.endswith("_date")]

    org_date = dt.datetime(2014, 1, 1)
    for date_col in date_cols:
        accepted_df[date_col] = accepted_df[date_col].apply(lambda x: x if (pd.isnull(x)) else (dt.datetime.strptime(str(x), '%b-%Y') - org_date).days)

    return accepted_df
