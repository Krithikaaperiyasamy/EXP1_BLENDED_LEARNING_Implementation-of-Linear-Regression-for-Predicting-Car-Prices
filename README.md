# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries & Load Dataset
2. Divide the dataset into training and testing sets.
3. Select a suitable ML model, train it on the training data, and make predictions.
4. Assess model performance using metrics and interpret the results.

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df = pd.read_csv('CarPrice_Assignment.csv')
df.head()

X=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

print('Name:KRITHIKAA P ')
print('Reg.No:212225040193')
print("MODEL COEFFICIENTS:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test,y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test,y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(y_test,y_pred):>10.2f}")

print(f"{'MAE':>12}: {mean_absolute_error(y_test,y_pred):>10.2f}")
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price (S)")
plt.grid(True)
plt.show()

residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic:{dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

plt.figure(figsize=(10,5))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

fig,(ax1, ax2)=plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:

<img width="1179" height="330" alt="Screenshot 2026-02-15 205103" src="https://github.com/user-attachments/assets/78b1f221-bbf6-4fcc-8cbf-4b0220707579" />


![Screenshot_15-2-2026_204131_localhost](https://github.com/user-attachments/assets/bd9139af-c3c4-4777-915c-75845173368d)

![Screenshot_15-2-2026_204554_localhost](https://github.com/user-attachments/assets/16ee03c7-58c8-452c-a8a9-f3cc03313b10)

![Screenshot_15-2-2026_20469_localhost](https://github.com/user-attachments/assets/ee9faf06-2d2b-43f8-b481-d53fabf94c81)

![Screenshot_15-2-2026_204622_localhost](https://github.com/user-attachments/assets/0f05d7bc-cae0-42ad-8628-b8a76dc4f26a)

![Screenshot_15-2-2026_204634_localhost](https://github.com/user-attachments/assets/1e2d619b-b7a2-49f8-a082-ca00f35b6a10)

![Screenshot_15-2-2026_204649_localhost](https://github.com/user-attachments/assets/7e4cde74-0922-4af9-a130-ff195b03b389)

![Screenshot_15-2-2026_20473_localhost](https://github.com/user-attachments/assets/5dbbb61f-a2f5-4031-b065-1de358ef4b72)

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
