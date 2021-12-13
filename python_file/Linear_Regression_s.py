# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # 8 Steps Of Machine Learning
# * Data Gathering
# * Data pre-processing
# * Feature Engineering
# * Choosing Model
# * Training Model
# * Test Model/ Model Evaluation
# * Parameter Tuning
# * Prediction

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
# Data Loading
data = pd.read_csv(r'.\FuelConsumption\FuelConsumption.csv')
data.head()


# %%
data.info()


# %%
data.describe()


# %%
data.drop(columns=["MODELYEAR","MAKE","MODEL", "TRANSMISSION"], inplace=True)


# %%
data[['FUELTYPE']].value_counts()


# %%
data[['VEHICLECLASS']].value_counts()


# %%
data.drop(columns=['VEHICLECLASS', 'FUELTYPE'], inplace=True)


# %%
data.head()


# %%
data.describe()


# %%
data[['ENGINESIZE']].hist()


# %%
data[['CYLINDERS']].hist()


# %%
data[['FUELCONSUMPTION_COMB']].hist()


# %%
# Feature Selection
data.corr()


# %%
sns.heatmap(data.corr(), annot=True)


# %%
data.drop(columns=['FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG'], inplace=True)


# %%
sns.heatmap(data.corr(), annot=True)


# %%
data.head()


# %%
feature = data.drop(columns=['CO2EMISSIONS'])
feature


# %%
target = data[['CO2EMISSIONS']]
target


# %%
# Spliting Data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(feature, target, test_size=0.25, random_state=1)


# %%
xtrain.shape, xtest.shape


# %%
ytrain.shape, ytest.shape


# %%
# Choosing Model

# plt.scatter(feature.ENGINESIZE ,target.values)
# plt.title("EngineSize vs Co2Emission")
# plt.xlabel("ENGINESIZE")
plt.scatter(feature.FUELCONSUMPTION_CITY ,target.values)
plt.title("FUELCONSUMPTION_CITY vs Co2Emission")
plt.xlabel("FUELCONSUMPTION_CITY")
plt.ylabel("CO2EMISSION")
plt.show()


# %%
# Modeling
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# model.fit(xtrain[['ENGINESIZE']], ytrain.values.ravel())
model.fit(xtrain[['FUELCONSUMPTION_CITY']], ytrain.values.ravel())


# %%
model.intercept_


# %%
model.coef_


# %%
# Prediction
# ypred = model.predict(xtest[['ENGINESIZE']])
# xtest[['ENGINESIZE']].iloc[0], ypred[0], ytest.values[0]

ypred = model.predict(xtest[['FUELCONSUMPTION_CITY']])
xtest[['FUELCONSUMPTION_CITY']].iloc[0], ypred[0], ytest.values[0]


# %%
ytest.values[0] - ypred[0] 


# %%
# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("Absoulute Error: ", mean_absolute_error(ytest, ypred))
print("Mean Squarred Error: ", mean_squared_error(ytest, ypred))
print("R2 Score: ", r2_score(ytest, ypred))


# %%
model2 = LinearRegression()
model2.fit(xtrain[['CYLINDERS']], ytrain.values.ravel())
# Prediction
ypred = model2.predict(xtest[['CYLINDERS']])

print("Absoulute Error: ", mean_absolute_error(ytest, ypred))
print("Mean Squarred Error: ", mean_squared_error(ytest, ypred))
print("R2 Score: ", r2_score(ytest, ypred))


# %%
# Linear Regression with FUELCONSUMPTION_CITY


