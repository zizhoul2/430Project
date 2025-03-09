# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import os
print(os.getcwd())  # Check current working directory    
os.chdir("/Users/noumana2/Downloads")
import pandas as pd
# Load CSV file
df = pd.read_csv("AQI_Updated_Dataset_V3.csv")

# Display the first few rows
print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Define the independent variables and target variable
Weather_Condition= ["temp", "humidity", "WindGustKmph", "precipMM"]
target = "AQI"

# Plot scatter plots for each feature vs. AQI
plt.figure(figsize=(12, 8))
for i, col in enumerate(Weather_Condition, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=df[col], y=df[target], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(f"{col} vs {target}")

plt.tight_layout()
plt.show()

print(df[['temp', 'humidity', 'WindGustKmph', 'precipMM', 'AQI']].corr())

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Load dataset 
data = pd.read_csv("AQI_Updated_Dataset_V3.csv", parse_dates=['Date'], index_col='Date')

# Set figure size and style
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Plot the AQI time series
plt.plot(data.index, data['humidity'], color='y', linewidth=2, label='Humidity')

# Add labels and title
plt.xlabel("Date", fontsize=12)
plt.ylabel("Humidity", fontsize=12)
plt.title("Time Series of AQI Over Time", fontsize=14)
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# Show the plot
plt.show()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("AQI_Updated_Dataset_V3.csv")
# Define X (features) and y (target variable, e.g., AQI)
X = data[['temp', 'humidity', 'windspeedKmph', 'precipMM']].values  # Features matrix
y = data['AQI'].values  # Target variable (AQI)

# Initialize PolynomialFeatures (degree 2 for quadratic)
poly = PolynomialFeatures(degree=2)

# Transform features to polynomial features
X_poly = poly.fit_transform(X)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
y_pred = model.predict(X_poly)

# Evaluate the model
r2 = model.score(X_poly, y)
mse = np.mean((y - y_pred) ** 2)

print(f'R-squared: {r2}')
print(f'Mean Squared Error: {mse}')

# Visualize Actual vs Predicted AQI
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', lw=2)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs Predicted AQI')
plt.show()

pip install statsmodels
import statsmodels.api as sm
# Step 3: Initialize PolynomialFeatures with degree 2 (quadratic, you can try higher degrees)
poly = PolynomialFeatures(degree=2)

# Step 4: Transform the features to polynomial features
X_poly = poly.fit_transform(X)

# Step 5: Add a constant (intercept) to the polynomial features
X_poly_with_intercept = sm.add_constant(X_poly)

# Step 6: Fit the model using OLS (Ordinary Least Squares)
model = sm.OLS(y, X_poly_with_intercept)  # OLS regression
results = model.fit()  # Fit the model

# Step 7: Get the summary of the model
print(results.summary())

# Step 8: Visualize Actual vs Predicted AQI
y_pred = results.predict(X_poly_with_intercept)
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k', lw=2)  # Add a reference line
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs Predicted AQI')
plt.show()

# Select only the relevant columns (temperature, humidity, wind speed, and precipitation)
X = data[['temp', 'humidity', 'windspeedKmph', 'precipMM']]  # Only the weather variables
y = data['AQI']  # Target variable (AQI)

# Transform the features using PolynomialFeatures (for non-linear regression)
poly = PolynomialFeatures(degree=2)  # You can adjust the degree for more non-linearity
X_poly = poly.fit_transform(X)

# Add constant to the model (intercept)
X_poly = sm.add_constant(X_poly)

# Fit the polynomial regression model
model = sm.OLS(y, X_poly).fit()

# Print the summary
print(model.summary())
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

# --- PLOT 1: Actual vs. Predicted AQI ---
plt.figure(figsize=(8,6))
sns.scatterplot(x=y, y=y_pred, alpha=0.6)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs. Predicted AQI")
plt.axline([0, 0], [1, 1], color="red", linestyle="--")  # 1:1 reference line
plt.show()

residuals = y - y_pred  # Residuals = Actual AQI - Predicted AQI

# --- PLOT 2: Residual Plot ---
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle="--")
plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# --- PLOT 3: Histogram of Residuals ---
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# --- PLOT 4: QQ-Plot ---
sm.qqplot(residuals, line='45', fit=True)
plt.title("QQ-Plot of Residuals")
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF for each feature (excluding the constant term)
vif_data = pd.DataFrame()
vif_data["Feature"] = poly.get_feature_names_out(X.columns)
vif_data["VIF"] = [variance_inflation_factor(X_poly, i) for i in range(X_poly.shape[1])]

print(vif_data)