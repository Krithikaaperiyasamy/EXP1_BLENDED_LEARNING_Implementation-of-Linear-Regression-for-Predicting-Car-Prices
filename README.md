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

<img width="1231" height="220" alt="Screenshot 2026-02-15 210658" src="https://github.com/user-attachments/assets/8e962263-1b17-4adc-ae4e-503c3b0bf9f0" />

<img width="1194" height="83" alt="Screenshot 2026-02-15 211226" src="https://github.com/user-attachments/assets/b0738b40-3db5-4b12-bf28-3ac1af69ec61" />

<img width="1263" height="594" alt="{1D1B9BBA-31E5-452B-8063-D7CC140F4103}" src="https://github.com/user-attachments/assets/fea4e9b0-e92d-48ad-9109-cfccb8cb9939" />

<img width="1200" height="83" alt="Screenshot 2026-02-15 211603" src="https://github.com/user-attachments/assets/dc8fdad2-bb5b-4d2d-a680-7d605e66fa2f" />

<img width="1210" height="590" alt="Screenshot 2026-02-15 211629" src="https://github.com/user-attachments/assets/282af8f9-7896-4479-a866-db4ea5252eb9" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
