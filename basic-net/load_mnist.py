import numpy as np
import sklearn.model_selection as model_selection
import scipy.io.arff as arff
import pandas as pd

arffData = arff.loadarff("../datasets/mnist_784.arff")
pdata = pd.DataFrame(arffData[0])

data, target = pdata[pdata.columns[:-1]], pdata[pdata.columns[-1]]

(rest_data, x_test, rest_target, y_test) = model_selection.train_test_split(data,target, test_size=10000,shuffle=True, random_state=24, stratify=target)

(x_train, x_validate, y_train, y_validate) = model_selection.train_test_split(rest_data,rest_target, test_size=10000,shuffle=True, random_state=24, stratify=rest_target)

def gety(y):
    z = np.zeros((10,1))
    z[int(y)] = 1.0
    return z

training_set = [(x.reshape(784,1), gety(y)) for x,y in zip(x_train.values, y_train)]
validation_set = [(x.reshape(784,1), gety(y)) for x,y in zip(x_validate.values, y_validate)]
test_set = [(x.reshape(784,1), gety(y)) for x,y in zip(x_test.values, y_test)]