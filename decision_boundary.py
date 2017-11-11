###

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# adpted from http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/


def plot_decision_boundary(pred_func, X, y): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 



def generate_moon_data(N = 50, noise = 0.2, plot=False):
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(N, noise=noise)
    if plot:
        plt.figure()
        plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
        plt.show()
    return X, y

def log_regression(X, y, plot=False):
    # Train the logistic rgeression classifier
    clf = sklearn.linear_model.LogisticRegressionCV() 
    clf.fit(X, y)
    
    if plot:
    # Plot the decision boundary
        plot_decision_boundary(lambda x: clf.predict(x), X, y)
        plt.title("Logistic Regression")
        plt.show()
    return clf

def lda(X, y, plot=False):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    if plot:
        plot_decision_boundary(lambda x: clf.predict(x), X, y)
        plt.title("LDA")
        plt.show()
    return clf

def qda(X, y, plot=False):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X, y)
    if plot:
        plot_decision_boundary(lambda x: clf.predict(x), X, y)
        plt.title("QDA")
        plt.show()
    return clf

if __name__ == "__main__":
    print("hello")
    X, y = generate_moon_data(N = 5000, noise = 0.2, plot = True)
    clf_LDA = lda(X,y, plot=True)
    clf_log = log_regression(X,y, plot=True)
    clf_log = qda(X,y, plot=True)