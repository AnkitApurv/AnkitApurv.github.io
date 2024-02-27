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
```


```python
ccdPayHistoryMode = ccdPayHistory.mode(axis = 'columns')
ccdPayModeSeverest = ccdPayHistoryMode.apply(func = max, axis = 'columns')
```


```python
ccd['PAY_MODE_SEVEREST'] = list(ccdPayModeSeverest)
```

#### BILL_AMT {BILL_AMT1 to BILL_AMT6}

Using mean for total credit used


```python
ccdSpent = ccd[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]
```


```python
ccd['BILL_AMT_MEAN'] = np.int32(ccdSpent.mean(axis = 'columns').round())
```

#### PAY_AMT {PAY_AMT1 to PAY_AMT6}

Using mean for total credit settled


```python
ccdSettled = ccd[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
```


```python
ccd['PAY_AMT_MEAN'] = np.int32(ccdSettled.mean(axis = 'columns').round())
```

## 2. Normalization

Scaling: Only to reduce the effect of very large continuous variables (in distance based estimators).

Normalization: Also reduce the effect of skewness in variables.


```python
varsToScale = ['LIMIT_BAL', 'AGE', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
               'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'BILL_AMT_MEAN', 'PAY_AMT_MEAN']
scaler = StandardScaler(copy = False)
```


```python
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
```


```python
isolationForest.fit(ccd)
```




    IsolationForest(contamination=0.001, max_samples=0.2, n_jobs=-1,
                    random_state=39)




```python
outlierLabels = isolationForest.predict(ccd)
```


```python
ccd['IsOutlier'] = list(outlierLabels)
```


```python
ccd['IsOutlier'].value_counts()
```




     1    29970
    -1       30
    Name: IsOutlier, dtype: int64




```python
ccd[ccd['IsOutlier'] == -1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>...</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default_payment_next_month</th>
      <th>PAY_MODE_SEVEREST</th>
      <th>BILL_AMT_MEAN</th>
      <th>PAY_AMT_MEAN</th>
      <th>IsOutlier</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1895</th>
      <td>-0.520128</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>-0.595102</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>6</td>
      <td>7</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>0.271165</td>
      <td>0.330267</td>
      <td>-0.314136</td>
      <td>-0.293382</td>
      <td>1</td>
      <td>7.0</td>
      <td>0.757355</td>
      <td>0.137285</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>3.179422</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3.418893</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>0.153766</td>
      <td>1.126807</td>
      <td>1.332635</td>
      <td>1.845825</td>
      <td>-0.281288</td>
      <td>0</td>
      <td>3.0</td>
      <td>6.967616</td>
      <td>1.438458</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2198</th>
      <td>6.416528</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1.249166</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>1.944483</td>
      <td>50.595281</td>
      <td>2.883583</td>
      <td>2.958533</td>
      <td>2.533615</td>
      <td>0</td>
      <td>0.0</td>
      <td>13.157468</td>
      <td>18.349659</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2250</th>
      <td>2.948200</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>-0.378129</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7.018117</td>
      <td>0.498352</td>
      <td>-0.033581</td>
      <td>10.031686</td>
      <td>1.328923</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.828201</td>
      <td>5.984819</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2688</th>
      <td>2.562830</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.923707</td>
      <td>-2</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.024515</td>
      <td>-0.283511</td>
      <td>-0.302318</td>
      <td>10.624170</td>
      <td>20.660179</td>
      <td>0</td>
      <td>0.0</td>
      <td>2.503397</td>
      <td>14.516176</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4337</th>
      <td>1.252573</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0.381275</td>
      <td>8</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>-0.296801</td>
      <td>-0.308063</td>
      <td>-0.314136</td>
      <td>-0.293382</td>
      <td>1</td>
      <td>8.0</td>
      <td>5.011108</td>
      <td>-0.520354</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>5297</th>
      <td>2.562830</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>-0.269643</td>
      <td>-2</td>
      <td>-2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>...</td>
      <td>72.842986</td>
      <td>6.622793</td>
      <td>5.926498</td>
      <td>24.510169</td>
      <td>1.211863</td>
      <td>0</td>
      <td>-1.0</td>
      <td>5.196440</td>
      <td>37.524671</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>5925</th>
      <td>1.329647</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>-0.486615</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>-0.240005</td>
      <td>-0.032304</td>
      <td>18.535258</td>
      <td>0.157027</td>
      <td>1</td>
      <td>-1.0</td>
      <td>5.397547</td>
      <td>4.433197</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6774</th>
      <td>2.331608</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>-0.378129</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.434706</td>
      <td>0.378568</td>
      <td>7.351887</td>
      <td>0.103456</td>
      <td>13.769599</td>
      <td>1</td>
      <td>0.0</td>
      <td>5.319030</td>
      <td>11.056937</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6913</th>
      <td>3.256496</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.984947</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>2.395515</td>
      <td>18.162114</td>
      <td>2.245254</td>
      <td>0.994931</td>
      <td>2.575466</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.254434</td>
      <td>8.112603</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>7857</th>
      <td>1.021351</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>-1.137534</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>0.271165</td>
      <td>0.330267</td>
      <td>-0.314136</td>
      <td>-0.293382</td>
      <td>1</td>
      <td>3.0</td>
      <td>4.165451</td>
      <td>-0.191584</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>9762</th>
      <td>1.483795</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>-0.269643</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>6</td>
      <td>...</td>
      <td>3.962756</td>
      <td>-0.296801</td>
      <td>-0.308063</td>
      <td>-0.314136</td>
      <td>-0.293382</td>
      <td>0</td>
      <td>2.0</td>
      <td>4.810665</td>
      <td>1.323048</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>12967</th>
      <td>1.792091</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.489762</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>-0.296801</td>
      <td>-0.301296</td>
      <td>-0.164837</td>
      <td>2.103906</td>
      <td>1</td>
      <td>7.0</td>
      <td>5.399223</td>
      <td>0.219552</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>13187</th>
      <td>4.489679</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>-0.812074</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17.147287</td>
      <td>0.157004</td>
      <td>0.394100</td>
      <td>27.044720</td>
      <td>-0.120126</td>
      <td>0</td>
      <td>-1.0</td>
      <td>3.583817</td>
      <td>17.416911</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>14554</th>
      <td>2.177460</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>-0.595102</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.524244</td>
      <td>1.130783</td>
      <td>27.276056</td>
      <td>0.798571</td>
      <td>0.831656</td>
      <td>1</td>
      <td>0.0</td>
      <td>7.488912</td>
      <td>8.233438</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18868</th>
      <td>0.096463</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>-0.812074</td>
      <td>8</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>-0.296801</td>
      <td>-0.308063</td>
      <td>0.078584</td>
      <td>-0.293382</td>
      <td>1</td>
      <td>8.0</td>
      <td>2.272017</td>
      <td>-0.421713</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>20893</th>
      <td>2.948200</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-0.052670</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0.741253</td>
      <td>0.725539</td>
      <td>-0.308063</td>
      <td>0.888439</td>
      <td>0.728097</td>
      <td>0</td>
      <td>2.0</td>
      <td>8.654093</td>
      <td>1.132375</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>21382</th>
      <td>3.950162</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3.093434</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.741253</td>
      <td>0.912968</td>
      <td>0.968596</td>
      <td>0.831298</td>
      <td>0.662901</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.957140</td>
      <td>1.465584</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>22851</th>
      <td>3.102348</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1.140680</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.832397</td>
      <td>0.564179</td>
      <td>0.440825</td>
      <td>2.492505</td>
      <td>23.444930</td>
      <td>0</td>
      <td>0.0</td>
      <td>6.758905</td>
      <td>8.298246</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>23040</th>
      <td>0.404759</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.598248</td>
      <td>8</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>-0.296801</td>
      <td>-0.308063</td>
      <td>-0.314136</td>
      <td>-0.293382</td>
      <td>1</td>
      <td>8.0</td>
      <td>3.013374</td>
      <td>-0.520354</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>23378</th>
      <td>2.948200</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>-0.378129</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.735307</td>
      <td>9.223569</td>
      <td>0.585598</td>
      <td>-0.032687</td>
      <td>8.598022</td>
      <td>1</td>
      <td>0.0</td>
      <td>6.836142</td>
      <td>5.857474</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>25147</th>
      <td>2.562830</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.008570</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>0.782335</td>
      <td>0.904763</td>
      <td>0.864025</td>
      <td>0.550397</td>
      <td>1</td>
      <td>2.0</td>
      <td>7.528826</td>
      <td>1.288031</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>25870</th>
      <td>2.639904</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0.489762</td>
      <td>8</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>-0.296801</td>
      <td>-0.308063</td>
      <td>-0.314136</td>
      <td>0.494145</td>
      <td>1</td>
      <td>8.0</td>
      <td>6.512539</td>
      <td>-0.290225</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>26548</th>
      <td>2.716978</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>2.008570</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.066767</td>
      <td>1.407099</td>
      <td>0.649431</td>
      <td>0.667665</td>
      <td>-0.293382</td>
      <td>0</td>
      <td>0.0</td>
      <td>8.243942</td>
      <td>1.435992</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>27441</th>
      <td>2.562830</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-0.052670</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>-0.002134</td>
      <td>28.568910</td>
      <td>0.846611</td>
      <td>0.606793</td>
      <td>0.550903</td>
      <td>1</td>
      <td>0.0</td>
      <td>6.292272</td>
      <td>9.116571</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>27468</th>
      <td>2.948200</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1.574625</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>0.786141</td>
      <td>1.970517</td>
      <td>-0.314136</td>
      <td>0.728659</td>
      <td>1</td>
      <td>2.0</td>
      <td>7.123908</td>
      <td>1.525361</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>28004</th>
      <td>2.639904</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>-0.595102</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>53.000654</td>
      <td>9.050623</td>
      <td>-0.275508</td>
      <td>0.094227</td>
      <td>-0.037436</td>
      <td>0</td>
      <td>-1.0</td>
      <td>3.267249</td>
      <td>30.655414</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>28625</th>
      <td>1.329647</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>-0.486615</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>-0.256990</td>
      <td>-0.296801</td>
      <td>-0.244230</td>
      <td>-0.031378</td>
      <td>15.906160</td>
      <td>1</td>
      <td>4.0</td>
      <td>6.349465</td>
      <td>4.350931</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>28717</th>
      <td>1.329647</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0.706734</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>52.496715</td>
      <td>50.197875</td>
      <td>39.332179</td>
      <td>0.994931</td>
      <td>7.863147</td>
      <td>0</td>
      <td>-1.0</td>
      <td>2.167496</td>
      <td>61.361453</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>29964</th>
      <td>3.410644</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-0.486615</td>
      <td>0</td>
      <td>-1</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-0.187330</td>
      <td>19.547669</td>
      <td>15.659359</td>
      <td>17.430208</td>
      <td>12.086317</td>
      <td>0</td>
      <td>-1.0</td>
      <td>4.177212</td>
      <td>22.747961</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 28 columns</p>
</div>



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
