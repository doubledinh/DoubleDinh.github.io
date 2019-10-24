## Regression Project - Uber/Life Prices


```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
import numpy as np
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
#Importing in the dataset
cab = pd.read_csv('Data/cab.csv')

#Cleaning out the df
df = cab.drop(columns = ['id', 'product_id', 'source'])

#Shuffling df
df = df.sample(frac=1).reset_index(drop=True)

#Splitting df and strictly selecting 20000 rows
df = df.iloc[0: 20000]
df = df.dropna()
df = df.reset_index()
```


```python
#Establishing the categorical variables into quantitative values
one_hot = pd.get_dummies(df['destination'])
df = df.drop('destination', axis = 1)

one_hot2 = pd.get_dummies(df['name'])
df = df.drop('name', axis = 1)

one_hot3 = pd.get_dummies(df['cab_type'])
df = df.drop('cab_type', axis = 1)

one_hot = one_hot.join(one_hot2)
one_hot.rename(columns = {'Lyft': 'Lyft_Standard'}, inplace = True)
one_hot = one_hot.join(one_hot3)
df = df.drop(columns = 'index')
```


```python
#Determining Coorelation
df.corr()['price'].sort_values(ascending=False)
```




    price               1.000000
    distance            0.339564
    surge_multiplier    0.230941
    time_stamp         -0.000821
    Name: price, dtype: float64




```python
#Producing a sns
sns.pairplot(df.drop('price', axis = 1), height = 1.2, aspect=1.5)
```




    <seaborn.axisgrid.PairGrid at 0x1a26d70518>




![png](Regression%20Project_files/Regression%20Project_5_1.png)



```python
#Establishing the Response Variable
y = df['price']

#Establishing the Numerical Values
df = df.drop(columns = ['price'])
```


```python
#Determining what transformation to do to the X variables
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3,random_state = 99)

mindegree = 0
maxdegree = 17

train_error = []
test_error = []

for deg in range(mindegree, maxdegree):
    model = make_pipeline(PolynomialFeatures(deg),LinearRegression())
    model.fit(X_train,y_train)
    train_error.append(mean_squared_error(y_train,model.predict(X_train)))
    test_error.append(mean_squared_error(y_test,model.predict(X_test)))

print(train_error,test_error)
plt.plot(np.arange(mindegree,maxdegree), train_error, color='green', label='train')
plt.plot(np.arange(mindegree,maxdegree), test_error, color='red', label='test')
plt.ylabel('mean squared error')
plt.xlabel('degree')
plt.legend(loc='upper left')
```

    [88.82923505639245, 72.32884392432562, 72.3290121267229, 72.32918691063526, 72.32936827998981, 72.32955623869177, 72.3297507906243, 72.32995193964851, 72.33015968960346, 72.33037404430603, 72.33059500755088, 72.33082258311052, 72.33105677473513, 72.33129758615259, 72.33154502106838, 72.3317990831656, 72.33205977610496] [86.26933174214516, 71.04994143183451, 71.05069524851348, 71.05145547298491, 71.05222210336544, 71.0529951413975, 71.05377458879742, 71.05456044725658, 71.05535271844127, 71.05615140399243, 71.05695650552579, 71.05776802463174, 71.05858596287572, 71.05941032179686, 71.0602411029098, 71.06107830770361, 71.06192193780815]





    <matplotlib.legend.Legend at 0x1a233892b0>




![png](Regression%20Project_files/Regression%20Project_7_2.png)



```python
# Applying Standard Scaler to numerical values
model = make_pipeline(PolynomialFeatures(1), StandardScaler())
model.fit(df)
```




    Pipeline(memory=None,
             steps=[('polynomialfeatures',
                     PolynomialFeatures(degree=1, include_bias=True,
                                        interaction_only=False, order='C')),
                    ('standardscaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True))],
             verbose=False)




```python
#Joining the onehots into the og dataset
x = one_hot.join(pd.DataFrame(model.transform(df)))
```


```python
#Applying a regular regression to the data without testing or training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state = 99)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

    0.9259680202801058



```python
#TESTING THE MODEL VS ACTUAL DATA
print("Model's Prediction:", model.predict([x.iloc[0]]))
print("Actual Value:", y.iloc[0])
print("Difference between the model and actual:", model.predict([x.iloc[0]]) - y.iloc[0])
```

    Model's Prediction: [27.70872498]
    Actual Value: 26.0
    Difference between the model and actual: [1.70872498]



```python
print(model.coef_)
print(model.intercept_)
```

    [ 8.02790500e+10  8.02790500e+10  8.02790500e+10  8.02790500e+10
      8.02790500e+10  8.02790500e+10  8.02790500e+10  8.02790500e+10
      8.02790500e+10  8.02790500e+10  8.02790500e+10  8.02790500e+10
      4.63798037e+00  1.44334362e+01  2.68472105e-01  5.75076117e+00
      1.49702252e+01 -7.84693436e+00 -2.08745478e+00 -1.06151195e+01
     -7.05488025e+00 -6.08661622e+00 -2.94479118e-01 -6.07579851e+00
      4.40132272e-01 -4.40132272e-01  0.00000000e+00  3.21355523e+00
      1.33026249e-04  1.91053895e+00]
    -80279049994.187



```python
#Applying ridge regression model
model1 = RidgeCV(alphas=[0.0001, 0.1, 1, 10], store_cv_values= True)
model1.fit(x, y)

print('R2:', model1.score(x, y))
print('Best Alpha: ', model1.alpha_)
print(f'MSE corresponding to best alpha {model1.alpha_}: {np.mean(model1.cv_values_, axis = 0)[0]}')
print(f"Adjusted R2: {1 - (1-model.score(x, y))*(len(y)-1)/(len(y)-x.shape[1]-1)}")
```

    R2: 0.9269276141873448
    Best Alpha:  0.0001
    MSE corresponding to best alpha 0.0001: 6.462622107032688
    Adjusted R2: 0.9267655407188023



```python
print('Coefficients for the model:', model1.coef_)
print('Intercept for the model:', model1.intercept_)
```

    Coefficients for the model: [-8.21148765e-02 -3.28881410e-01  1.75447401e-01 -2.77632290e-01
      4.45172150e-01  1.08060848e-01  3.33448560e-02  3.13941517e-02
      3.98346812e-02 -1.70080141e-01  1.13472264e-01 -8.80176306e-02
      4.58556963e+00  1.44274421e+01  2.81797572e-01  5.77794982e+00
      1.50271100e+01 -7.88912639e+00 -2.10718633e+00 -1.06561204e+01
     -7.02340736e+00 -6.05354767e+00 -3.01134429e-01 -6.06934658e+00
      4.34424309e-01 -4.34424309e-01  0.00000000e+00  3.19080954e+00
     -4.19487897e-03  1.88847413e+00]
    Intercept for the model: 16.552488135975047



```python
#TESTING THE MODEL VS ACTUAL DATA
print("Model's Prediction:", model1.predict([x.iloc[0]]))
print("Actual Value:", y.iloc[0])
print("Difference between the model and actual:", model1.predict([x.iloc[0]]) - y.iloc[0])
```

    Model's Prediction: [27.72356594]
    Actual Value: 26.0
    Difference between the model and actual: [1.72356594]

