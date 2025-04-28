from nixtla import NixtlaClient
import pandas as pd 
from utilsforecast.preprocessing import fill_gaps


nixtla_client = NixtlaClient(
    api_key = 'CUSTOM_KEY_FOR_USER'
)

print(nixtla_client.validate_api_key())

# Load and clean data
df = pd.read_csv('/Users/cosmin/Desktop/analiza articole/electricitate 2024/consum/Consum 2023-2024 National copy.csv')
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M', dayfirst=True)
df = df.set_index('date')

# Sort index
df = df.sort_index()

# Remove duplicates
df = df[~df.index.duplicated(keep='first')]

# Ensure hourly frequency
df = df.asfreq('H')

# Fill missing values
df = df.interpolate()

# print(test)
# print(df.head())
# fig = nixtla_client.plot(df, time_col='date', target_col='RO Load')
# fig.savefig('plot.png', bbox_inches='tight')



timegpt_fcst_df = nixtla_client.forecast(
    df=df, 
    h=24, 
    finetune_steps=5,    #incepusem cu 0 steps si avea o acuratete mult mai proasta
    time_col='date', 
    target_col='RO Load')

print(timegpt_fcst_df.head())
timegpt_fcst_df.to_excel('output-TimeGPT2.xlsx', index=True)