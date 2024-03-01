"""
PGA Tour Data Analysis Project

This project aims to analyze PGA Tour data to gain insights into professional golf performance.

It includes data loading, preprocessing, visualization, and machine learning modeling for prediction.

"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load PGA Tour data from a CSV file into a pandas DataFrame
pro_df = pd.read_csv(r'C:\Users\danie\OneDrive\Documents\Py_Project\PgaTourData.csv')

# Display descriptive statistics of the dataset
pro_df.describe()

# Display the first few rows of the DataFrame to understand the structure and content of the data
pro_df.head()

# Sorting the DataFrame in ascending order based on the 'POINTS' column
pro_df.sort_values("POINTS")

# Import Matplotlib and create a histogram of the 'AVG_SCORE' column in pro_df
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(pro_df["AVG_SCORE"])

# Plotting histograms of "AVG_SCORE" for pro_df and Scratch_data, and setting axis labels
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(pro_df["AVG_SCORE"])
ax.set_xlabel("AVG_SCORE")
ax.set_ylabel("HOLES_PLAYED")
plt.show()

# Creating a scatter plot of 'AGE' vs 'AVG_CARRY_DISTANCE' using Matplotlib
fig, ax = plt.subplots()
ax.scatter(pro_df["AGE"], pro_df["AVG_CARRY_DISTANCE"], c=pro_df.index)
ax.set_xlabel("AGE")
ax.set_ylabel("AVG_CARRY_DISTANCE")
plt.show()

# Dropping duplicate rows from the DataFrame
pro_df.drop_duplicates()

# Calculating the number of missing data points per column
missing_values_count = pro_df.isnull().sum()

# Importing Seaborn and Matplotlib libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting a histogram of "AVG_SCORE" using Seaborn
fig, ax = plt.subplots()
ax.hist(pro_df["AVG_SCORE"])
plt.show()

# Converting the 'POINTS_BEHIND_LEAD' column to numeric, filling missing values with 0, and converting to integer type
pro_df['POINTS_BEHIND_LEAD'] = pd.to_numeric(pro_df['POINTS_BEHIND_LEAD'], errors='coerce').fillna(0).astype(int)
pro_df.dtypes

# Displaying the minimum and maximum values of the 'AGE' column
print(pro_df['AGE'].min())
print(pro_df['AGE'].max())

# Creating a boxplot of 'AGE' using Seaborn
sns.boxplot(data=pro_df,x="AGE")
plt.show()

# Computing mean and standard deviation of the DataFrame
pro_df.agg(["mean", "std"])

# Grouping by 'MAKES_BOGEY%' and summing the 'Player' column
pro_by_bogey_percentage = pro_df.groupby("MAKES_BOGEY%")["Player"].sum()

# Creating a bar plot using Seaborn
sns.barplot(data=pro_df.groupby("MAKES_BOGEY%")["Player"].sum(), x="ROUNDS_PLAYED", y="MAKES_BOGEY%")
plt.show()

# Checking for missing data points in the DataFrame
print(pro_df.isna().sum())

# Define features (X) and target variable (y)
X = pro_df[['AGE', 'AVG_CARRY_DISTANCE', 'SG_PUTTING_PER_ROUND', 'TOTAL_SG:PUTTING']]
y = pro_df['AVG_SCORE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Example of using the model to predict a single data point
new_data_point = np.array([[26, 300, -0.003, -2]]) 
new_data_point_scaled = scaler.transform(new_data_point)
predicted_avg_score = model.predict(new_data_point_scaled)
print('Predicted AVG_SCORE:', predicted_avg_score)

# Initialize and train the Random Forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
print('Random Forest Mean Squared Error:', mse_rf)

# Example of using the model to predict a single data point
new_data_point = np.array([[26, 300, -0.003, -2]])  # Example data point
new_data_point_scaled = scaler.transform(new_data_point)
predicted_avg_score_rf = rf_model.predict(new_data_point_scaled)
print('Random Forest Predicted AVG_SCORE:', predicted_avg_score_rf)

# Initialize and train the Gradient Boosting regression model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred_gb = gb_model.predict(X_test_scaled)

# Evaluate the model
mse_gb = mean_squared_error(y_test, y_pred_gb)
print('Gradient Boosting Mean Squared Error:', mse_gb)

# Example of using the model to predict a single data point
new_data_point = np.array([[26, 300, -0.003, -2]])  # Example data point
new_data_point_scaled = scaler.transform(new_data_point)
predicted_avg_score_gb = gb_model.predict(new_data_point_scaled)
print('Gradient Boosting Predicted AVG_SCORE:', predicted_avg_score_gb)
