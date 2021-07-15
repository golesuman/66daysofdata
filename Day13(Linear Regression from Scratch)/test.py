from scipy.sparse import data
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
x,y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=40)
from linearregression import LinearRegression
model=LinearRegression(lr=0.001)
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(pred)
