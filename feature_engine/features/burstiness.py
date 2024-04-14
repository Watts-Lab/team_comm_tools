import pandas as pd
import numpy as np

def get_team_burstiness(df, timediff):

    # Check for any NA values and drop them accordingly
    df[timediff] = df[timediff].replace('NULL_TIME', np.nan)
    wait_times = (df[timediff].dropna()).astype(float).values

    if len(wait_times) <= 1:
        return 0
    
    # Compute coefficient of variation measure B (Goh & Barabasi 2008)
    standard_deviation = np.std(wait_times)
    mean = np.mean(wait_times)
    B = (standard_deviation - mean) / (standard_deviation + mean)
    return B
