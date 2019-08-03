import numpy as np
import pandas as p
import matplotlib.pyplot as plt

data=p.read_csv('4.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=100,random_state=0)
reg.fit(x,y)

ypred=reg.predict([[6,2.5,5.7]])