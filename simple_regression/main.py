import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


df = pd.read_csv("EPL_Soccer_MLR_LR.csv")
df.info()
df.describe()

# df.corr()

plt.scatter(df["Cost"], df["Score"])

x = df["Cost"]
y = df["Score"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, test_size=0.25, random_state=100
)

# Fit the model

lr = sm.OLS(y_train, x_train).fit()
lr.params
lr.summary()

# force intercept term
x_train_with_intercept = sm.add_constant(x_train)
lr = sm.OLS(y_train, x_train_with_intercept).fit()
lr.summary()

x_test_with_intercept = sm.add_constant(x_test)
y_test_fitted = lr.predict(x_test_with_intercept)

plt.scatter(x_test, y_test)
plt.plot(x_test, y_test_fitted, "r")
plt.show()

print(lr.params)
b0 = lr.params[0]
b1 = lr.params[1]

plt.scatter(x_train, y_train)
plt.plot(x_train, b0 + b1 * x_train, "r")
plt.show()
