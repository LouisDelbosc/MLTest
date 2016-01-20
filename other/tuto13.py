import Quandl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

api_key = open('../../quandlKey.txt', 'r').read()

def mortgage_30y():
    df = Quandl.get("FMAC/MORTG", trim_start="1975-01-01", authtoken=api_key)
    # print df
    # #df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    # print df.head()
    # return df

mortgage_30y()
