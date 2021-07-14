# importing numpy 
import numpy as np
#importing the counter which counts the maximum common value
from collections import Counter

def euclidean_distance(x1,x2): #it calculates the distance between new data point and the nearest class
    return np.sqrt(np.sum(x1-x2)**2)

# KNN it fits and predicts the new data point
class KNN:
    def __init__(self,k) -> None:
        self.k=k
    
    def fit(self,x,y):
        self.x_train=x
        self.y_train=y

    def predict(self,X): #it returns the array of predicted labels
        predicted_labels=[self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self,x):
        # compute the distances 
        distances=[euclidean_distance(x, x_train)  for x_train in self.x_train]
        #get k nearest samples,labels
        k_indices=np.argsort(distances)[:self.k] #sort the distances
        k_nearest_labels=[self.y_train[i] for i in k_indices] # get the labels
        #majority vote
        most_common=Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]




