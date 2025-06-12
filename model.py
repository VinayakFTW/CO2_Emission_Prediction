import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("1mil_Canada dataset.csv")


df.dropna(inplace=True)



enc_df = df.copy()

enc = OrdinalEncoder(categories=[['X','Z','E','D','N']])
enc_df[["Fuel Type"]] = enc.fit_transform(enc_df[["Fuel Type"]])



x = enc_df.drop(["CO2 Emissions(g/km)","Make","Model","Fuel Consumption Hwy (L/100 km)","Fuel Consumption City (L/100 km)","Vehicle Class","Transmission"],axis=1)


y = df["CO2 Emissions(g/km)"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)


model = XGBRegressor(device='cuda',n_estimators=6000,early_stopping_rounds=50,random_state=42,max_depth=5,reg_alpha=0.0,reg_lambda=1.0,learning_rate=0.01,subsample=0.5)
model.fit(x_train, y_train,eval_set=[(x_test, y_test)],verbose=500)


x_test = enc_df.drop(["CO2 Emissions(g/km)","Make","Model","Fuel Consumption Hwy (L/100 km)","Fuel Consumption City (L/100 km)","Vehicle Class","Transmission"],axis=1)
y_test = enc_df["CO2 Emissions(g/km)"]


y_pred = model.predict(x_test)


pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(pred_df.head())


importances = model.feature_importances_
feature_names = x.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

r2_test = r2_score(y_test,model.predict(x_test))
r2_train = r2_score(y_train,model.predict(x_train))
rms = mean_squared_error(y_test,y_pred)**0.5
print(f"RMS:{rms}")
print(f"Test R2 = {r2_test}")
print(f"Train R2 = {r2_train}")
