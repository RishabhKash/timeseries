import os
import pandas as pd
from datetime import datetime
from library.training_module import train_predict

# changing the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv('/Users/rishabhkashyap/Downloads/MOCK_DATA.csv')
data['year'] = pd.to_datetime(data['year']).apply(lambda x: x.strftime("%Y-%m"))
data['revenue'] = data['revenue'].apply(lambda x: x[1:])
data['revenue'] = pd.to_numeric(data['revenue'])
data = data.groupby('year').agg({'revenue': 'sum'}).reset_index()

data = data[data['year'] != '2022-12']
data.tail()
performance, predictions = train_predict(processed_data=data, project='revenue')
