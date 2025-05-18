import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def mean_sq_error(ay):
    d = [yi - ai for ai, yi in ay]
    ds = [sum(di * di) for di in d]
    return (1 / (2 * len(ay))) * sum(ds)

def mean_sq_error_prime(a, y):
    return (a - y) * a * (1 - a)

def cross_entropy_error(ay):
    d = [(y * np.log(a) + (1-y) * np.log(1-a)) for a, y in ay]
    return (-1/len(ay)) * sum(np.array(d).flatten())

def cross_entropy_error_prime(a, y):
    return a - y

def accuracy_percent(ay):
    correct = sum([int(np.argmax(y) == np.argmax(a)) for a,y in ay])
    percent = correct * 100 / len(ay)
    return percent