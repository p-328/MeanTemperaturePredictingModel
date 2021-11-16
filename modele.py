import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model

dataframe = pd.read_csv("Summary of Weather.csv", low_memory=False)
dataframe = dataframe[["MaxTemp", "MinTemp", "MeanTemp"]]
value_to_predict = "MeanTemp"

X = np.array(dataframe.drop([value_to_predict], axis=1))
Y = np.array(dataframe[value_to_predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print(f"Accuracy: {str(accuracy)}")
print(f"Coefficient: {str(linear.coef_)}")
print(f"Intercept: {str(linear.intercept_)}")
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(f"Prediction: {str(predictions[i])}", x_test[i], f"Actual: {str(y_test[i])}")
