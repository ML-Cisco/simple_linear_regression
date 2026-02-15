# 📈 Simple Linear Regression — Theory and Mathematical Derivation

## Problem Statement

We are building a Machine Learning model to predict **Sales** based on **TV advertising spend**.

![](./asserts/00.png)

Our goal is to find the best-fitting straight line that predicts sales from TV spend.

---

## 1️⃣ Simple Linear Regression Model

We assume a linear relationship between input and output:

![](./asserts/01.png)

Where:

![](./asserts/02.png)

For a dataset of \( n \) observations:

![](./asserts/03.png)

Our objective is to determine optimal values of

![](./asserts/04.png)

---

## 2️⃣ Residuals (Errors)

For each data point:

![](./asserts/05.png)

Substituting the model:

![](./asserts/06.png)

The residual represents the vertical distance between the actual value and predicted value.

---

## 3️⃣ Residual Sum of Squares (RSS)

To avoid cancellation of positive and negative residuals, we square them:

![](./asserts/07.png)

This is the objective function we want to minimize.

---

## 4️⃣ Mean Squared Error (MSE)

The average squared error:

![](./asserts/08.png)

MSE provides a scale-independent measure of error.

---

## 5️⃣ Root Mean Squared Error (RMSE)

Since MSE has squared units, we take the square root:

![](./asserts/09.png)

RMSE is interpreted as the average prediction error in the original unit (sales).

---

## 6️⃣ Matrix Representation

Instead of scalar form, we express the model using matrices.

### Define Matrices

Target vector:

![](./asserts/10.png)

Design matrix:

![](./asserts/11.png)

Weight vector:

![](./asserts/12.png)

Model equation:

![](./asserts/13.png)

---

## 7️⃣ RSS in Matrix Form

Residual vector:

![](./asserts/14.png)

RSS becomes:

![](./asserts/15.png)

This is equivalent to summing squared residuals.

---

## 8️⃣ Derivation of the Normal Equation

Expand:

![](./asserts/16.png)

Take derivative with respect to 𝑊

![](./asserts/17.png)

Set equal to zero:

![](./asserts/18.png)

Solve:

![](./asserts/19.png)

This is called the **Normal Equation**.

---

## 9️⃣ What This Code is Doing (Conceptually)

1. Load dataset
2. Split into train and test sets
3. Convert scalar form to matrix form
4. Apply:
   ![](./asserts/19.png)
5. Predict:
   ![](./asserts/13.png)
6. Evaluate using RMSE
7. Plot residuals to validate assumptions

---

This project demonstrates the complete theoretical foundation and mathematical derivation behind Simple Linear Regression.
