import pandas as pd

df1 = pd.DataFrame({ 'HPI': [80, 85, 88, 85],
                     'Int_rate': [2, 3, 2, 2],
                     'US_GDP_Thousands': [50, 55, 65, 55]},
                   index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({ 'HPI': [80, 85, 88, 85],
                     'Int_rate': [2, 3, 2, 2],
                     'US_GDP_Thousands': [50, 55, 65, 55]},
                   index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({ 'HPI': [80, 85, 88, 85],
                     'Unemployment': [7, 8, 9, 6],
                     'Low_tier_HPI': [50, 52, 50, 53]},
                   index = [2001, 2002, 2003, 2004])

# for 2nd part of the tuto
df4 = pd.DataFrame({
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55],
                    'Year':[2001, 2002, 2003, 2004]
                    })

df5 = pd.DataFrame({
                    'Unemployment':[7, 8, 9, 6],
                    'Low_tier_HPI':[50, 52, 50, 53],
                    'Year':[2001, 2003, 2004, 2005]})

# print pd.merge(df1, df2, on=['HPI', 'Int_rate'])

# df4 = pd.merge(df1, df3, on='HPI')
# df4.set_index('HPI', inplace=True)
# print df4

df1.set_index('HPI', inplace=True)
df3.set_index('HPI', inplace=True)

# joined =df1.join(df3)
# print joined

merged = pd.merge(df4, df5, on='Year', how='outer') # left/right/outer/inner join
merged.set_index('Year', inplace=True)
print merged
