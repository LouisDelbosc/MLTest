import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import style

# style.use('fivethirtyeight')

web_stats = {'Day': [1,2,3,4,5,6],
             'Visitors': [43,34,65,56,29,76],
             'Bounce_Rate': [65,67,78,65,45,52]}

df = pd.DataFrame(web_stats)
df = df.set_index('Day')

# print(df.head())
# print(df.tail())
print(df['Visitors']) #Same as : print df.Visitors
print(df[['Visitors', 'Bounce_Rate']])

# Display data graphs
# df['Visitors'].plot()
# plt.show()
# df.plot()
# plt.show()

