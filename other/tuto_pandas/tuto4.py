import Quandl
import pandas as pd

api_key = open('../../quandlKey.txt', 'r').read()

df = Quandl.get("FMAC/HPI_TX", authtoken=api_key)
fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')

#print df.head()
#print fiddy_states # fiddy state => list of dataframe
#print fiddy_states[0] # first dataframe of fiddy_states

for abbv in fiddy_states[0][0][1:]:
    # print abbv
    print "FMAC/HPI_" + str(abbv)
