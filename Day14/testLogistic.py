from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()
x=dataset.data
y=dataset.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from logistic_regression import LogisticRegression
clf=LogisticRegression(lr=0.01,n_iters=1000)
clf.fit(x_train,y_train)
y_pre=clf.predict(x_test)
print(y_pre)


