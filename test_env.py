import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

X = np.array([[1],[2],[3],[4]])
y = np.array([2,4,6,8])

model = LinearRegression().fit(X, y)
print("R^2:", model.score(X, y))
print("pandas OK:", pd.DataFrame({"x":[1,2], "y":[3,4]}).shape)
