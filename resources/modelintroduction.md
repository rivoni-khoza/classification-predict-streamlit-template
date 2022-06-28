# Models Introduction

**Stack Classifier**

![<an image representing a the way a stacked classifier works>](resources/imgs/Stack classifier.png)

Stacked generalization consists in stacking the output of individual estimator and use a classifier to compute the final prediction. Stacking allows to use the strength of each individual estimator by using their output as input of a final estimator.
Note that estimators_ are fitted on the full X while final_estimator_ is trained using cross-validated predictions of the base estimators using cross_val_predict.


**Linear SVM**

![<an image of the SVC model>](resources/imgs/SVC Model pic.png)

Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes. Support vectors are the data points that lie closest to the decision surface (or hyperplane).
SVMs maximize the margin around the separating hyperplane.
The decision function is fully specified by a subset of training samples, the support vectors.


**Logistic Regression** 

![<an image of the Logistic Regression model>](resources/imgs/logistic regression 2.png)

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes.Logistic Regression  algorithms are used for the classification problems, it is a predictive analysis algorithm and based on the concept of probability. Logistic Regression uses a complex cost function, this cost function can be defined as the ‘Sigmoid function’ or also known as the ‘logistic function’ instead of a linear function. The Sigmoid function maps any real value into another value between 0 and 1. In machine learning, we use sigmoid to map predictions to probabilities.
