import pandas as pd

def process_data(dataset):
    dataset['datetime'] = dataset.apply(lambda x: f"{x['date']} {x['time']}", axis=1)
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset['weekday'] = dataset['datetime'].dt.weekday
    dataset['month'] = dataset['datetime'].dt.month
    dataset['hour'] = dataset['datetime'].dt.weekday

    dataset = dataset.drop(['date', 'time'], axis=1)

    return dataset
