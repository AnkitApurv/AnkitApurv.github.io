---
layout: default
categories: "DataScience"
permalink: /:categories/:title
title: "Feature Engineering - Part 4 of 5"
---

# Feature Engineering

In this 5 part walk-through, I'll demonstrate a simple Machine Learning project to build a classifier model.

In this step, we will do the following three tasks:  

1. Feature Engineering: to create new, meaningful variables which may help with our classification problem.
2. Normalization.
3. Outlier Detection.
<!--end-excerpt-->

## Pre-Requisites
#### Importing libraries


```python
#data organizing
import pandas #storage
import numpy as np #data-type conversion
from os import getcwd #to get relative path so dataset may be easy and simple to find and load

#Scaling/Normalization
from sklearn.preprocessing import StandardScaler

#outlier removal to achieve better distribution
from sklearn.ensemble import IsolationForest

#splitting the dataset - simple method
from sklearn.model_selection import train_test_split
```

#### Importing the dataset


API Docs: [Pandas DataFrame read_excel()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html)


```python
url = getcwd() + '\\default of credit card clients.xls'
ccd = pandas.read_excel(io = url, \
                        sheet_name='Data', header = 1, index_col = 0, \
                        dtype = {'LIMIT_BAL': np.int32, 'AGE': np.int32, 'BILL_AMT1': np.int32, 'BILL_AMT2': np.int32, 'BILL_AMT3': np.int32, 'BILL_AMT4': np.int32, 'BILL_AMT5': np.int32, 'BILL_AMT6': np.int32, 'PAY_AMT1': np.int32, 'PAY_AMT2': np.int32, 'PAY_AMT3': np.int32, 'PAY_AMT4': np.int32, 'PAY_AMT5': np.int32, 'PAY_AMT6': np.int32})
```

__dtype__ changed from int64 to int32 to save space and speed up computation, however, while doing so, we should firstly know that there won't be any overflow or underflow of data.

#### Bringing variables' names upto convention

This step is only to bring the dataset's variables' name upto convention, if any aren't.

[Pandas DataFrame rename()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html)

In this case PAY started from 0 and had no PAY_1.

The target variable has it's spaces removed since spaces can cause issues with some libraries.


```python
ccd.rename(columns = {'PAY_0': 'PAY_1'}, inplace = True)
ccd.rename(columns = {'default payment next month': 'default_payment_next_month'}, inplace = True)
```

## 1. Feature Engineering

Feature Engineering requires domain knowledge and insights gained from Exploratory Data Analysis, what we do in this step isn't set in stone and we can get creative with it.

Through feature engineering, one can isolate key information, highlight patterns. These things depend on domain expertise as well as what one can glean off of Exploratory Data Analysis.

#### PAY {PAY_1 to PAY_6}

1. Using mode to aggregate. An entry may have multiple mode values (same frequency), to resolve, using severest class.

2. Why severest value? To ensure fiscally fit population of credit users.


```python
ccdPayHistory = ccd[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

ccdPayHistoryMode = ccdPayHistory.mode(axis = 'columns')
ccdPayModeSeverest = ccdPayHistoryMode.apply(func = max, axis = 'columns')

ccd['PAY_MODE_SEVEREST'] = list(ccdPayModeSeverest)
```

#### BILL_AMT {BILL_AMT1 to BILL_AMT6}

Using mean for total credit used


```python
ccdSpent = ccd[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]

ccd['BILL_AMT_MEAN'] = np.int32(ccdSpent.mean(axis = 'columns').round())
```

#### PAY_AMT {PAY_AMT1 to PAY_AMT6}

Using mean for total credit settled


```python
ccdSettled = ccd[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

ccd['PAY_AMT_MEAN'] = np.int32(ccdSettled.mean(axis = 'columns').round())
```

## 2. Normalization

Scaling: Only to reduce the effect of very large continuous variables (in distance based estimators).

Normalization: Also reduce the effect of skewness in variables.


```python
varsToScale = ['LIMIT_BAL', 'AGE', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
               'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'BILL_AMT_MEAN', 'PAY_AMT_MEAN']
scaler = StandardScaler(copy = False)

for var in varsToScale:
    ccd[var] = scaler.fit_transform(ccd[var].values.reshape(-1, 1))

```

## 3. Outlier Detection

Since data is highly skewed with the higher end being very sparse, having mostly outliers,

It may be better to remove those outliers so rest of the dataset has better distribution for better prediction
And outlier datapoints could be have a separate classifier model

Should be done before data split to ensure distribution of train, dev and test sets are not different from each other.

To tune this Outlier detection algorithm, we can tune the value of __contamination__ parameter.


```python
isolationForest = IsolationForest(n_estimators = 100, max_samples = 0.2, contamination = 0.001,
    n_jobs = -1, random_state = 39)

isolationForest.fit(ccd)
```

!!! example "Output"

    ```python
    IsolationForest(contamination=0.001, max_samples=0.2, n_jobs=-1,
        random_state=39)
    ```

```python
outlierLabels = isolationForest.predict(ccd)

ccd['IsOutlier'] = list(outlierLabels)

ccd['IsOutlier'].value_counts()
```

!!! example "Output"

    | IsOutlier | Count |
    | --------: | ----: |
    | 1         | 29970 |
    | -1        |    30 |


```python
ccd[ccd['IsOutlier'] == -1]
```

??? example "Output"

    | LIMIT_BAL | SEX       | EDUCATION | MARRIAGE | AGE | PAY_1     | PAY_2 | PAY_3 | PAY_4 | PAY_5 | ... | PAY_AMT2 | PAY_AMT3  | PAY_AMT4  | PAY_AMT5  | PAY_AMT6  | default_payment_next_month | PAY_MODE_SEVEREST | BILL_AMT_MEAN | PAY_AMT_MEAN | IsOutlier |
    |-----------|-----------|-----------|----------|-----|-----------|-------|-------|-------|-------|-----|----------|-----------|-----------|-----------|-----------|----------------------------|-------------------|---------------|--------------|-----------|
    | ID        |
    | 1895      | -0.520128 | 1         | 2        | 2   | -0.595102 | 1     | 3     | 7     | 6     | 7   | ...      | -0.256990 | 0.271165  | 0.330267  | -0.314136 | -0.293382                  | 1                 | 7.0           | 0.757355     | 0.137285  | -1 |
    | 1993      | 3.179422  | 1         | 3        | 1   | 3.418893  | 2     | 2     | 3     | 3     | 4   | ...      | 0.153766  | 1.126807  | 1.332635  | 1.845825  | -0.281288                  | 0                 | 3.0           | 6.967616     | 1.438458  | -1 |
    | 2198      | 6.416528  | 2         | 1        | 1   | 1.249166  | 0     | 0     | 0     | -1    | 0   | ...      | 1.944483  | 50.595281 | 2.883583  | 2.958533  | 2.533615                   | 0                 | 0.0           | 13.157468    | 18.349659 | -1 |
    | 2250      | 2.948200  | 2         | 2        | 1   | -0.378129 | 0     | 0     | 0     | 0     | 0   | ...      | 7.018117  | 0.498352  | -0.033581 | 10.031686 | 1.328923                   | 1                 | 0.0           | 5.828201     | 5.984819  | -1 |
    | 2688      | 2.562830  | 2         | 1        | 1   | 0.923707  | -2    | -1    | 0     | 0     | 0   | ...      | 0.024515  | -0.283511 | -0.302318 | 10.624170 | 20.660179                  | 0                 | 0.0           | 2.503397     | 14.516176 | -1 |
    | 4337      | 1.252573  | 2         | 2        | 1   | 0.381275  | 8     | 7     | 6     | 5     | 4   | ...      | -0.256990 | -0.296801 | -0.308063 | -0.314136 | -0.293382                  | 1                 | 8.0           | 5.011108     | -0.520354 | -1 |
    | 5297      | 2.562830  | 2         | 1        | 1   | -0.269643 | -2    | -2    | -1    | -1    | -2  | ...      | 72.842986 | 6.622793  | 5.926498  | 24.510169 | 1.211863                   | 0                 | -1.0          | 5.196440     | 37.524671 | -1 |
    | 5925      | 1.329647  | 2         | 2        | 2   | -0.486615 | 4     | 3     | 2     | -1    | -1  | ...      | -0.256990 | -0.240005 | -0.032304 | 18.535258 | 0.157027                   | 1                 | -1.0          | 5.397547     | 4.433197  | -1 |
    | 6774      | 2.331608  | 2         | 2        | 2   | -0.378129 | 0     | 0     | 0     | 0     | 0   | ...      | 0.434706  | 0.378568  | 7.351887  | 0.103456  | 13.769599                  | 1                 | 0.0           | 5.319030     | 11.056937 | -1 |
    | 6913      | 3.256496  | 1         | 1        | 1   | 2.984947  | 0     | 0     | 0     | -1    | 0   | ...      | 2.395515  | 18.162114 | 2.245254  | 0.994931  | 2.575466                   | 0                 | 0.0           | 7.254434     | 8.112603  | -1 |
    | 7857      | 1.021351  | 2         | 1        | 2   | -1.137534 | 1     | 5     | 4     | 3     | 2   | ...      | -0.256990 | 0.271165  | 0.330267  | -0.314136 | -0.293382                  | 1                 | 3.0           | 4.165451     | -0.191584 | -1 |
    | 9762      | 1.483795  | 1         | 2        | 1   | -0.269643 | 1     | 2     | 2     | 7     | 6   | ...      | 3.962756  | -0.296801 | -0.308063 | -0.314136 | -0.293382                  | 0                 | 2.0           | 4.810665     | 1.323048  | -1 |
    | 12967     | 1.792091  | 1         | 2        | 1   | 0.489762  | 7     | 6     | 5     | 4     | 3   | ...      | -0.256990 | -0.296801 | -0.301296 | -0.164837 | 2.103906                   | 1                 | 7.0           | 5.399223     | 0.219552  | -1 |
    | 13187     | 4.489679  | 2         | 1        | 2   | -0.812074 | 1     | -1    | -1    | 0     | 0   | ...      | 17.147287 | 0.157004  | 0.394100  | 27.044720 | -0.120126                  | 0                 | -1.0          | 3.583817     | 17.416911 | -1 |
    | 14554     | 2.177460  | 1         | 2        | 2   | -0.595102 | 0     | 0     | 0     | 0     | 0   | ...      | 0.524244  | 1.130783  | 27.276056 | 0.798571  | 0.831656                   | 1                 | 0.0           | 7.488912     | 8.233438  | -1 |
    | 18868     | 0.096463  | 2         | 2        | 2   | -0.812074 | 8     | 7     | 6     | 5     | 4   | ...      | -0.256990 | -0.296801 | -0.308063 | 0.078584  | -0.293382                  | 1                 | 8.0           | 2.272017     | -0.421713 | -1 |
    | 20893     | 2.948200  | 1         | 1        | 2   | -0.052670 | 2     | 2     | 2     | 2     | 2   | ...      | 0.741253  | 0.725539  | -0.308063 | 0.888439  | 0.728097                   | 0                 | 2.0           | 8.654093     | 1.132375  | -1 |
    | 21382     | 3.950162  | 1         | 2        | 2   | 3.093434  | 0     | 0     | 0     | 0     | 0   | ...      | 0.741253  | 0.912968  | 0.968596  | 0.831298  | 0.662901                   | 0                 | 0.0           | 7.957140     | 1.465584  | -1 |
    | 22851     | 3.102348  | 2         | 2        | 1   | 1.140680  | 0     | 0     | 0     | 0     | 0   | ...      | 0.832397  | 0.564179  | 0.440825  | 2.492505  | 23.444930                  | 0                 | 0.0           | 6.758905     | 8.298246  | -1 |
    | 23040     | 0.404759  | 2         | 1        | 1   | 0.598248  | 8     | 7     | 6     | 5     | 4   | ...      | -0.256990 | -0.296801 | -0.308063 | -0.314136 | -0.293382                  | 1                 | 8.0           | 3.013374     | -0.520354 | -1 |
    | 23378     | 2.948200  | 2         | 2        | 1   | -0.378129 | 2     | 0     | 0     | 0     | 0   | ...      | 0.735307  | 9.223569  | 0.585598  | -0.032687 | 8.598022                   | 1                 | 0.0           | 6.836142     | 5.857474  | -1 |
    | 25147     | 2.562830  | 1         | 1        | 1   | 2.008570  | 2     | 2     | 2     | 0     | 0   | ...      | -0.256990 | 0.782335  | 0.904763  | 0.864025  | 0.550397                   | 1                 | 2.0           | 7.528826     | 1.288031  | -1 |
    | 25870     | 2.639904  | 2         | 1        | 2   | 0.489762  | 8     | 7     | 6     | 5     | 4   | ...      | -0.256990 | -0.296801 | -0.308063 | -0.314136 | 0.494145                   | 1                 | 8.0           | 6.512539     | -0.290225 | -1 |
    | 26548     | 2.716978  | 2         | 3        | 1   | 2.008570  | 0     | 0     | 0     | 0     | 0   | ...      | 1.066767  | 1.407099  | 0.649431  | 0.667665  | -0.293382                  | 0                 | 0.0           | 8.243942     | 1.435992  | -1 |
    | 27441     | 2.562830  | 1         | 1        | 1   | -0.052670 | 2     | 0     | 0     | -1    | 0   | ...      | -0.002134 | 28.568910 | 0.846611  | 0.606793  | 0.550903                   | 1                 | 0.0           | 6.292272     | 9.116571  | -1 |
    | 27468     | 2.948200  | 1         | 3        | 1   | 1.574625  | 2     | 2     | 2     | 0     | 0   | ...      | -0.256990 | 0.786141  | 1.970517  | -0.314136 | 0.728659                   | 1                 | 2.0           | 7.123908     | 1.525361  | -1 |
    | 28004     | 2.639904  | 2         | 1        | 2   | -0.595102 | -1    | -1    | -1    | -1    | 0   | ...      | 53.000654 | 9.050623  | -0.275508 | 0.094227  | -0.037436                  | 0                 | -1.0          | 3.267249     | 30.655414 | -1 |
    | 28625     | 1.329647  | 2         | 2        | 2   | -0.486615 | 5     | 4     | 4     | 3     | 2   | ...      | -0.256990 | -0.296801 | -0.244230 | -0.031378 | 15.906160                  | 1                 | 4.0           | 6.349465     | 4.350931  | -1 |
    | 28717     | 1.329647  | 2         | 1        | 3   | 0.706734  | -1    | -1    | -1    | -1    | -1  | ...      | 52.496715 | 50.197875 | 39.332179 | 0.994931  | 7.863147                   | 0                 | -1.0          | 2.167496     | 61.361453 | -1 |
    | 29964     | 3.410644  | 1         | 1        | 2   | -0.486615 | 0     | -1    | 2     | -1    | -1  | ...      | -0.187330 | 19.547669 | 15.659359 | 17.430208 | 12.086317                  | 0                 | -1.0          | 4.177212     | 22.747961 | -1 |



## Post-processing
#### Splitting the dataset

Though splitting the dataset is done after Exploratory Data Analysis and Feature Engineering, it is simply being show-cased here.

As far as the ratio of sizes of test, train and dev sets is concerned, it depends on:

1. Size of the dataset. (10<sup>3</sup> - 100<sup>3</sup> data-points vs. 10<sup>6</sup> data-points)
2. Learning algorithms being used. (Deep learning convoluted neural networks vs. Shallow learning Decision Trees)
3. Problem type (simple classification, regression vs. Natural Language Processing)


```python
ccdY = ccd['default_payment_next_month']
ccdX = ccd.drop(['default_payment_next_month'], axis = 'columns')
```

For this dataset split, deciding factors are:

1. Size: 30000 data-points.
2. Algorithms: Shallow learning, ensemble of shallow learning algorithms.
3. Problem type: Simple Classification.


```python
trainX, testX, trainY, testY = train_test_split(ccdX, ccdY, test_size = 0.25, random_state = 44)
devX, testX, devY, testY = train_test_split(testX, testY, test_size = 0.5, random_state = 44)
```

<sup><sub>
Posted: 02nd September, 2020, 00:20 UTC+5:30  
</sub></sup>
