#!/usr/bin/env python
# coding: utf-8

# ## House Price Prediction Using Linear Regression
# 
# ### Steve Matindi (Stevemats) - https://github.com/stevemats/House_Price_Prediction

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def load_dataset(file_path):
    """Load the dataset from a CSV file"""
    return pd.read_csv(file_path)


# In[3]:


def preprocess_data(data):
    """Preprocess the dataset by handling missing values"""
    # Fill missing values with the mean of the respective column
    data = data.fillna(data.mean())
    return data


# In[4]:


def explore_dataset(data):
    """Display the first few rows and information of the dataset"""
    print("First few rows of the dataset:")
    print(data.head())
    print("\nDataset information:")
    print(data.info())


# In[5]:


def visualize_data(data):
    """Visualize the data"""
    # Pairplot to visualize relationships between features
    sns.pairplot(data, x_vars=data.columns[:-1], y_vars=['medv'], kind='scatter')
    plt.title("Pairplot of Features vs. Target")
    plt.show()

    # Heatmap to visualize correlations between features
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()


# In[6]:


def split_data(data):
    """Split the dataset into features (X) and target (y)"""
    X = data.drop('medv', axis=1)
    y = data['medv']
    return X, y


# In[7]:


def train_model(X_train, y_train):
    """Train a linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# In[8]:


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


# In[9]:


def visualize_predictions(y_test, y_pred):
    """Visualize the actual vs. predicted prices"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs. Predicted Prices")
    plt.show()


# In[10]:


def predict_new_house_price(model, new_house_features):
    """Predict the price of a new house"""
    predicted_price = model.predict(new_house_features)
    formatted_price = "${:,.2f}".format(predicted_price[0])
    print("Predicted Price of the House:", formatted_price)


# In[11]:


def main():
    # Load the dataset
    file_path = 'dataset/BostonHousing.csv'
    boston = load_dataset(file_path)

    # Preprocess the dataset
    boston = preprocess_data(boston)

    # Explore the dataset
    explore_dataset(boston)
    visualize_data(boston)

    # Split the data into training and testing sets
    X, y = split_data(boston)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Make predictions
    y_pred = model.predict(X_test)
    visualize_predictions(y_test, y_pred)

    # Predict the price of a new house
    new_house_features = np.array([[0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]])
    predict_new_house_price(model, new_house_features)


# In[12]:


if __name__ == "__main__":
    # Hide warning concerning feature names
    import warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LinearRegression was fitted with feature names")
    
    main()


# In[ ]:




