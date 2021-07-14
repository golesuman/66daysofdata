from sklearn.datasets import load_iris
import numpy as np
data=load_iris()
import pandas as pd
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2)
from knnFromScratch import KNN
clf=KNN(k=5)
clf.fit(x_train,y_train)
predictions=clf.predict(x_test)
acc=np.sum(predictions==y_test)/len(y_test)
print(predictions)
print(acc)