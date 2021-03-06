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
    state = state_list()
    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = Quandl.get(query, authtoken=api_key)
        print(query)
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        print df.head()
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
    main_df.to_pickle('fiddy_state3.pickle', 'wb')

def HPI_Benchmark():
    df = Quandl.get("FMAC/HPI_USA", authtoken=api_key)
    key = df.key()[0]
    df[key] = (df[key] - df[key][0]) / df[key][0] * 100.0
    return df

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))
HPI_data = pd.read_pickle('fiddy_states3.pickle')
HPI_State_Correlation = HPI_data.corr()
TX1yr= HPI_data['TX'].resample('A')#A for year, look for panda documentation for other aliases
# print TX1yr.head()
HPI_data['TX'].plot(ax=ax1)
TX1yr.plot(color='k', ax=ax1)

plt.legend().remove()
plt.show()
