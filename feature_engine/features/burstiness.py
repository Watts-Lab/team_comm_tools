import pandas as pd
import numpy as np

def burstiness(df, timediff):

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

def get_team_burstiness(df, timediff):
    # Applies burstiness function to overall dataframe and then groups coefficient by conversation number
    burstiness_coeff = df.groupby("conversation_num").apply(lambda x : burstiness(x, timediff)).reset_index().rename(columns={0: "team_burstiness"})
    return burstiness_coeff
