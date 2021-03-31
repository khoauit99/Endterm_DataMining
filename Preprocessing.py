"""
Pre-processing data:

* input: .csv file
* output: .csv file

1. Features:
    - Drop RISK_MM
    - Separate 'Date' columns into 3 new columns: 'Year', 'Month', 'Date'
2. Missing data:
    - Numeric: replaced by mean value.
    - Nominal (categorical): replaced by popular string.
    - Drop null (option)
3. Outliers:
    - Using IQR
    - Z-score (option)
4. String to int (categorical columns):
    - One hot encoding
    - Label encoder
5. Normalization
    - Min-max
    - Standard (option)

"""


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessing:

    def __init__(self, output_file):
        self.output_file = output_file

    def preprocessing(self, input_file):
        data = pd.read_csv(input_file)

        # drop features
        data = data.drop(columns=['RISK_MM'])

        # remove 'Date' column
        # create 3 new columns: date, month, year
        data['Date'] = pd.to_datetime(data['Date'])
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data.drop('Date', axis=1, inplace=True)

        # separate columns to numeric and nominal(categorical)
        numeric = []
        nominal = []
        list_columns = data.columns.tolist()
        for i in list_columns:
            if data[i].dtypes == 'float64' or data[i].dtypes == 'int64':
                numeric.append(i)
            else:
                nominal.append(i)

        # with null data in numeric columns, we can replace it by mean value of column
        for i in numeric:
            mean = data[i].mean()
            data[i].replace(to_replace=np.nan, value=mean, inplace=True)

        # with null data in nominal columns, we can replace it by popular string
        for i in nominal:
            popular_str = data[i].mode()[0]
            data[i].replace(to_replace=np.nan, value=popular_str, inplace=True)

        # another method to deal with null data -> drop all null data
        # drop_null
        # data = data.dropna()

        # outliers
        # create a list include features that maybe contain outliers
        outliers = ['Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Evaporation']

        # detect and remove outliers by using z-score
        # z = np.abs(stats.zscore(data))
        # data = data[(z < 3).all(axis=1)]

        # detect and remove outliers by using IQR
        for i in outliers:
            IQR = data[i].quantile(0.75) - data[i].quantile(0.25)
            lower_fence = data[i].quantile(0.25) - (IQR * 1.5)
            upper_fence = data[i].quantile(0.75) + (IQR * 1.5)
            data[i] = np.where(data[i] < lower_fence, lower_fence, data[i])
            data[i] = np.where(data[i] > upper_fence, upper_fence, data[i])

        # one_hot_encoding
        # convert to int
        # all categorical columns
        # data = pd.get_dummies(data, columns=nominal)

        # label_encoder
        # covert label to int
        # Yes -> 1, No -> 0
        le = LabelEncoder()
        for i in nominal:
            data[i] = le.fit_transform(data[i])
        # normalization

        # min-max
        scale = MinMaxScaler()
        data = pd.DataFrame(scale.fit_transform(data), columns=data.columns)

        # standard
        # scale = StandardScaler()
        # data = pd.DataFrame(scale.fit_transform(data), columns=data.columns)

        # output_data
        data.to_csv(self.output_file, index=False, header=True)





