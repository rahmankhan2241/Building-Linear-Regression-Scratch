# 🧠 Linear Regression From Scratch

This repository demonstrates the **implementation of Linear Regression from scratch using Python**, without relying on machine learning libraries like scikit-learn. It is intended as a hands-on guide to deeply understand how linear regression works behind the scenes.

---

## 📌 What is Linear Regression?

Linear regression is one of the simplest yet most powerful algorithms in statistics and machine learning. It is used to model the relationship between a **dependent variable** and one or more **independent variables** by fitting a linear equation to observed data.

---

## 🎯 Why Use Linear Regression?

- To **predict** a numeric outcome (e.g., sales, prices, weight).
- To **understand relationships** between variables.
- To assess **trends over time**.

Common applications include:
- House price predictions
- Stock market trends
- Advertising and marketing performance
- Medical risk factors analysis

---

## 🧮 Mathematical Formula

The formula for a linear regression model with one feature is:

\[
y = \beta_0 + \beta_1 x + \epsilon
\]

- \( y \): Dependent variable (what you're trying to predict)
- \( x \): Independent variable (the input feature)
- \( \beta_0 \): Intercept (bias)
- \( \beta_1 \): Coefficient (slope)
- \( \epsilon \): Error term

For multiple features, the equation becomes:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
\]

In matrix form:

\[
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}
\]

---

## ⚙️ How It Works (Behind the Scenes)

The goal is to **minimize the error** between predicted values and actual values. This is usually done by minimizing the **cost function**:

\[
J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

This is also known as the **Mean Squared Error (MSE)**.

In this project, we use the **normal equation** to compute coefficients directly:

\[
\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
\]

While elegant, this method becomes computationally expensive for datasets with **a large number of features**, as it involves matrix inversion — an operation with **O(n³)** complexity.

---

## ⚠️ Limitations of Normal Equation Approach

- **Computationally Inefficient**: Inverting a matrix becomes very slow as the number of features grows.
- **Memory Intensive**: Especially with high-dimensional data.
- **Numerical Instability**: When features are highly correlated (multicollinearity), the matrix \( X^TX \) might be non-invertible or ill-conditioned.

---

## ✅ Why Gradient Descent is Preferred in Practice

- **Scales Better** to large datasets
- Avoids costly matrix inversion
- Works well with **regularization techniques** (like Ridge, Lasso)
- Easier to implement in **online learning scenarios**

---

## 📊 Linear Regression Demo with Visualization

Below is a simple Python script to demonstrate linear regression on dummy data:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate dummy data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Demo")
plt.legend()
plt.show()
```
<img src = 'https://res.cloudinary.com/dkjob6qvb/image/upload/v1748590466/output_s2teyi.png'></img>
