import pandas as pd
import numpy as np

def ahead_task_parser(ahead, sampling_rate):
    """
    Parse the ahead task.
    """
    T = pd.to_timedelta(sampling_rate)
    if ahead == 'day':
        return pd.to_timedelta('1d') // T, pd.to_timedelta('7d') // T
    elif ahead == 'week':
        return pd.to_timedelta('7d') // T, pd.to_timedelta('30d') // T
    elif ahead == 'month':
        return pd.to_timedelta('30d') // T, pd.to_timedelta('120d') // T
    else:
        raise ValueError('Invalid ahead task.')