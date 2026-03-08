import xgboost as xgb
import numpy as np

# Dummy training data
X = np.array([
    [10, 2, 1, 0],
    [15, 3, 2, 1],
    [8, 1, 0, 0],
    [20, 4, 3, 1]
])

y = np.array([100, 150, 80, 200])

# Train model
model = xgb.XGBRegressor()
model.fit(X, y)

# Save model in native format
model.get_booster().save_model("demand_model.json")

print("Model saved as demand_model.json")
