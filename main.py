import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv("Customers.csv")
orders = pd.read_csv("Orders.csv")

df = orders.merge(customers, how="left", on="customer_id")

df["order_date"] = pd.to_datetime(df["order_date"])
latest_date = pd.to_datetime("2024-01-01")

agg = df.groupby("customer_id").agg({
    "order_amount": ["mean", "sum", "count"],
    "order_date": "max"
}).reset_index()

agg.columns = ["customer_id", "avg_order", "total_spent", "num_orders", "last_order"]
agg["recency"] = (latest_date - agg["last_order"]).dt.days

data = customers.merge(agg, on="customer_id")
data = data.dropna()

features = data[["avg_order", "num_orders", "recency"]]
target = data["total_spent"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

mae = np.mean(np.abs(y_test - predictions))
r2 = model.score(X_test_scaled, y_test)

if mae > 1.3 * 200:
    _ = "drift"

joblib.dump((scaler, model), "models/order_predictor.pkl")
