import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#######################################################
# Step 1: Create a synthetic dataset
#######################################################
data = {
    'Wind Speed (km/h)': [25, 15, 40, 20, 35, 30, 50, 10, 45, 25],
    'Temperature (°C)': [30, 25, 35, 28, 32, 30, 40, 22, 38, 27],
    'Humidity (%)': [45, 60, 40, 50, 55, 48, 35, 65, 42, 52],
    'Fire Spread Rate (km/h)': [2.5, 1.8, 3.1, 2.0, 2.8, 2.3, 3.5, 1.5, 3.0, 2.2]
}

df = pd.DataFrame(data)

#######################################################
# Step 2: Train a machine learning model
#######################################################
X = df[['Wind Speed (km/h)', 'Temperature (°C)', 'Humidity (%)']]
y = df['Fire Spread Rate (km/h)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#######################################################
# Step 3: Make predictions and evaluate the model
#######################################################
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Output results
print("Predicted Fire Spread Rates:", y_pred)
print("Actual Fire Spread Rates:", y_test.values)
print(f"Mean Absolute Error: {mae}")

#######################################################
# Step 4: Predict for new data
#######################################################
new_data = np.array([[30, 33, 50]])  # New example data (Wind Speed = 30 km/h, Temp = 33°C, Humidity = 50%)
predicted_spread = model.predict(new_data)

print(f"Predicted Fire Spread Rate: {predicted_spread[0]} km/h")



