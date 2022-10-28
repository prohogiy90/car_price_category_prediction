import os
import dill
import json
import pandas as pd
import datetime as datetime

from os.path import basename, splitext

path = os.environ.get('PROJECT_PATH', 'C:\\Users\\Konstantin\\airflow_hw')


def get_model_name():
    name = str()
    date_time = 0
    for model_name in os.listdir(f'{path}/data/models/'):
        full_name, ext = splitext(basename(model_name))
        if int(full_name.split('_')[2]) > date_time:
            name = full_name
            date_time = int(full_name.split('_')[2])
    return name


def call_model(df, name):
    with open(f'{path}/data/models/{name}.pkl', 'rb') as file:
        model = dill.load(file)
    y = model.predict(df)
    result = pd.DataFrame({'id': df.id, 'price': df.price, 'pred': y[0]})
    return result


def read_car_info(filename):
    with open(f'{path}/data/test/{filename}', 'r') as js_file:
        data = json.load(js_file)
    df = pd.DataFrame([data])
    return df


def predict():
    results = pd.DataFrame(columns=['id', 'price', 'pred'])
    for filename in os.listdir(f'{path}/data/test'):
        X = read_car_info(filename)
        name = get_model_name()
        result = call_model(X, name)
        results = pd.concat([results, result], axis=0)
    results.to_csv(os.path.join(f'{path}/data/predictions',
                                f'prediction_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.csv'))


if __name__ == '__main__':
    predict()
