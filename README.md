# 📈 Simple Linear Regression — Theory and Mathematical Derivation

## Problem Statement

We are building a Machine Learning model to predict **Sales** based on **TV advertising spend**.

- Input (Feature): `x` → TV advertising spend
- Output (Target): `y` → Sales

Our goal is to find the best-fitting straight line that predicts sales from TV spend.

---

# 1️⃣ Simple Linear Regression Model

We assume a linear relationship between input and output:

\[
\hat{y} = w_0 + w_1 x
\]

Where:

- \( w_0 \) → Intercept (bias)
- \( w_1 \) → Slope (effect of TV advertising on sales)
- \( \hat{y} \) → Predicted value

For a dataset of \( n \) observations:

\[
(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)
\]

Our objective is to determine optimal values of \( w_0 \) and \( w_1 \).

---

# 2️⃣ Residuals (Errors)

For each data point:

\[
e_i = y_i - \hat{y}_i
\]

Substituting the model:

\[
e_i = y_i - (w_0 + w_1 x_i)
\]

The residual represents the vertical distance between the actual value and predicted value.

---

# 3️⃣ Residual Sum of Squares (RSS)

To avoid cancellation of positive and negative residuals, we square them:

\[
RSS = \sum_{i=1}^{n} (y_i - (w_0 + w_1 x_i))^2
\]

This is the objective function we want to minimize.

---

# 4️⃣ Mean Squared Error (MSE)

The average squared error:

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - (w_0 + w_1 x_i))^2
\]

MSE provides a scale-independent measure of error.

---

# 5️⃣ Root Mean Squared Error (RMSE)

Since MSE has squared units, we take the square root:

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i - (w_0 + w_1 x_i))^2
}
\]

RMSE is interpreted as the average prediction error in the original unit (sales).

---

# 6️⃣ Matrix Representation

Instead of scalar form, we express the model using matrices.

## Define Matrices

Target vector:

\[
Y =
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}
\]

Design matrix:

\[
X =
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
\vdots & \vdots \\
1 & x_n
\end{bmatrix}
\]

Weight vector:

\[
W =
\begin{bmatrix}
w_0 \\
w_1
\end{bmatrix}
\]

Model equation:

\[
\hat{Y} = XW
\]

---

# 7️⃣ RSS in Matrix Form

Residual vector:

\[
E = Y - XW
\]

RSS becomes:

\[
RSS = (Y - XW)^T (Y - XW)
\]

This is equivalent to summing squared residuals.

---

# 8️⃣ Derivation of the Normal Equation

We minimize:

\[
RSS = (Y - XW)^T (Y - XW)
\]

### Expand:

\[
RSS = Y^T Y - 2W^T X^T Y + W^T X^T X W
\]

### Take derivative with respect to \( W \):

\[
\frac{\partial RSS}{\partial W}
=
-2X^T Y + 2X^T X W
\]

### Set derivative to zero:

\[
X^T X W = X^T Y
\]

### Solve for \( W \):

\[
\boxed{
W = (X^T X)^{-1} X^T Y
}
\]

This is called the **Normal Equation**.

---

# 9️⃣ Why This Minimizes RSS

- RSS is a quadratic function of \( W \).
- Quadratic functions are convex.
- Setting derivative to zero gives the global minimum.
- \( X^T X \) is positive semi-definite.
- If invertible, solution is unique.

---

# 🔟 Geometric Interpretation

The problem can be written as:

\[
\min_W \|Y - XW\|^2
\]

This means:

- We are projecting vector \( Y \) onto the column space of \( X \).
- The predicted vector \( XW \) is the orthogonal projection.
- Residual vector is perpendicular to the column space.

Thus, linear regression is fundamentally a projection problem.

---

# 1️⃣1️⃣ Model Assumptions

Simple Linear Regression assumes:

1. Linearity (relationship is linear)
2. Independence of errors
3. Homoscedasticity (constant variance)
4. Errors are normally distributed (for inference)
5. No perfect multicollinearity

---

# 1️⃣2️⃣ Training and Testing

To avoid overfitting:

- Split dataset into training and testing sets.
- Train model using training set.
- Evaluate performance using test set.
- Compute RMSE on unseen data.

---

# 1️⃣3️⃣ Final Optimization Objective

The entire learning process reduces to solving:

\[
\min_W \|Y - XW\|^2
\]

Solution:

\[
W = (X^T X)^{-1} X^T Y
\]

This provides the optimal parameters \( w_0 \) and \( w_1 \).

---

# 1️⃣4️⃣ Summary

Simple Linear Regression:

- Models relationship between one input and one output.
- Minimizes squared error.
- Has a closed-form solution.
- Can be interpreted algebraically, geometrically, and statistically.
- Forms the foundation for advanced ML models.

---

# 🚀 Next Steps

After mastering this, you can extend to:

- Multiple Linear Regression
- Gradient Descent Optimization
- Ridge and Lasso Regularization
- Logistic Regression
- Neural Networks

---

This project demonstrates the complete theoretical foundation and mathematical derivation behind Simple Linear Regression.
