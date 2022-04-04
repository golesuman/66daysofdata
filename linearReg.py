import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



class LinearRegression():
    def __init__(self,learning_rate=0.01,iterations=1000) -> None:
        self.learning_rate = learning_rate
        self.iterations=iterations

    def fit(self,X,Y):
        self.X=X
        self.Y=Y
        self.m,self.n=X.shape
        self.W=np.zeros(self.n)
        self.b=0
        for i in range(self.iterations):
            self.update_weights()
        return self


    def update_weights(self):
        y_pred=self.predict(self.X)
        dw=-(2*(self.X.T).dot(self.Y-y_pred))/self.m
        db=-2*np.sum(self.Y-y_pred)/self.m
        self.W = self.W - self.learning_rate * dw
        self.b = self.b -self.learning_rate * db

        return self


    def predict(self,X):
        return X.dot(self.W)+self.b



def main():
    df=pd.read_csv('Student_Marks.csv')
    x=df.iloc[:,:-1].values
    y=df.iloc[:,1].values
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f'The predicted values: {np.round(y_pred[:5],2)}')
    print(f'The real values are : {np.round(y_test[:5],2)}')

if __name__ == '__main__':
    main()
        