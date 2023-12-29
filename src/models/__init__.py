import numpy as np
import pandas as pd
import tensorflow as tf

SUITS_COLORS = {"S": "black", "C": "black", "H": "red", "D": "red"}
CARDS = [
    '9S', 'TS', 'JS', 'QS', 'KS', 'AS',
    '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
    '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
    '9C', 'TC', 'JC', 'QC', 'KC', 'AC'
]
BID_VALUES = [0, 4, 5, 6, 'pfeffer']
BID_SUITS = ['S', 'H', 'D', 'C', 'NT']  # NT represents No-Trump


def find_index(lst, value):
    if isinstance(lst, list):
        return lst.index(value)
    elif isinstance(lst, np.ndarray):
        return np.where(lst == value)[0][0]
    elif isinstance(lst, pd.Series):
        return lst[lst == value].index[0]
    elif isinstance(lst, dict):
        return list(lst.values()).index(value)
    elif tf.is_tensor(lst):  # check if lst is a tensorflow Tensor
        eq = tf.equal(lst, value)  # get Boolean tensor
        where = tf.where(eq)  # get indices where condition is True
        if len(where) > 0:  # check if there are any True values
            return where[0][0].numpy()  # return the first True index
        else:
            raise ValueError("Value not found in tensor.")
    else:
        raise ValueError("Unsupported list type.")
