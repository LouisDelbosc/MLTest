import Quandl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

api_key = open('../../quandlKey.txt', 'r').read()

def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]

def grab_initial_state_data():
    states = state_list()
    main_df = pd.DataFrame()
    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = Quandl.get(query, authtoken=api_key)
        print query
        df = df.pct_change() # add for the 2nd version 
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
    #main_df.to_pickle('fiddy_states.pickle') # 1st version of the function
    main_df.to_pickle('fiddy_states2.pickle') # new pickle

def grab_initial_state_data_bis():
    states = state_list()
    main_df = pd.DataFrame()
    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = Quandl.get(query, authtoken=api_key)
        print query
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        print df.head()
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
    main_df.to_pickle('fiddy_states3.pickle')

# grab_initial_state_data_bis()
# HPI_data = pd.read_pickle('fiddy_states.pickle')
# HPI_data = pd.read_pickle('fiddy_states3.pickle')

# HPI_data.plot()
# plt.legend().remove()
# plt.show()

def HPI_Benchmark():
    df = Quandl.get("FMAC/HPI_USA", authtoken=api_key)
    key = df.keys()[0]
    #print df[key]
    df[key] = (df[key]-df[key][0]) / df[key][0] * 100.0
    return df

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

HPI_data = pd.read_pickle('fiddy_states3.pickle')
benchmark = HPI_Benchmark()
HPI_data.plot(ax=ax1)
benchmark.plot(color='k', ax=ax1, linewidth=10)

HPI_state_correlation = HPI_data.corr()
print(HPI_state_correlation.describe())

# plt.legend().remove()
# plt.show()


