from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import numpy as np
import pandas as pd

def read_posterior_data(data: str) -> pd.DataFrame:
    """
    Read posterior data from a CSV file and return a DataFrame with relevant columns.
    Parameters:
    - data: str or pd.DataFrame, path to the CSV file or DataFrame containing posterior data.
    Returns:
    - pd.DataFrame with columns: PatientID, TrueLabel, Probability.
    """
    data = pd.read_csv(data)
    return data[["PatientID", "TrueLabel", "Probability"]]

def calculate_DeLong(rocA: pd.DataFrame, rocB: pd.DataFrame) -> float:
    """
    Calculate the DeLong test for comparing two ROC curves.
    Parameters:
    - rocA: pd.DataFrame, DataFrame containing PatientID, TrueLabel, and Probability for model A.
    - rocB: pd.DataFrame, DataFrame containing PatientID, TrueLabel, and Probability for model B.
    Returns:
    - p-value from the DeLong test.
    """
    # Merge to align
    roc = pd.merge(
        left=rocA, 
        right=rocB,
        how='inner',
        left_on=['PatientID', 'TrueLabel'],
        right_on=['PatientID', 'TrueLabel'],
        suffixes=('_A', '_B')
    )
    # Check if len stayed the same
    if len(roc) != len(rocA) or len(roc) != len(rocB):
        raise ValueError("ROCs should keep the same length, but this is not the case, meaning there are non-overlapping patients")

    # Activate automatic conversion of numpy arrays to R vectors
    numpy2ri.activate()

    # Import pROC R package
    pROC = importr('pROC')

    # Call roc() for each model
    y_true = roc["TrueLabel"].to_numpy()
    roc1 = pROC.roc(y_true, roc["Probability_A"].to_numpy())
    roc2 = pROC.roc(y_true, roc["Probability_B"].to_numpy())

    # Run DeLong test and return p-value
    DeLong = pROC.roc_test(roc1, roc2, method="delong", paired=True)
    return DeLong.rx2('p.value')[0]
