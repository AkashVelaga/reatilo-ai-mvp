# train_model.py

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

X = np.array([
    [10,2,1,0],
    [15,3,2,1],
    [8,1,0,0],
    [20,4,3,1],
    [12,2,1,0]
])

y = np.array([100,150,80,200,120])

model = RandomForestRegressor()
model.fit(X,y)

joblib.dump(model,"demand_model.pkl")

print("Model saved")
