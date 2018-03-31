# machine-learning-regression-using-iris-data-set
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
regressor=LinearRegression()
dataframe=pd.read_csv('iris.data')
labelencoder_y=LabelEncoder()
x=dataframe.iloc[:,:-1].values 
y=dataframe.iloc[:,-1].values
y[:]=labelencoder_y.fit_transform(y[:])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.0,random_state=0)
regressor.fit(x_train,y_train)
x_test=[[6.1,2.9,4.7,1.4],[4.4,2.9,1.4,0.2]]
y_predict=regressor.predict(x_test)
y_predict
