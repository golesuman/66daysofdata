import numpy as np

# linear regression class
class LinearRegression:
    def __init__(self,lr=0.001,n_iters=1000) -> None:
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None

# function for fiting
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iters):
            y_predicted=np.dot(x,self.weights)+self.bias
            dw=(1/n_samples)*np.dot(x.T,(y_predicted-y))
            db=(1/n_samples)*np.sum(y_predicted-y)

            self.weights -= self.lr*dw
            self.bias-=self.lr*db
# function for prediction
    def predict(self,x):
         y_predicted=np.dot(x,self.weights)+ self.bias
         return y_predicted


