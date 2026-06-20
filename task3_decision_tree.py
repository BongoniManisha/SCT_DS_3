import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
df = pd.read_csv(
    "bank-full.csv",
    sep=";"
)
print("Dataset Loaded Successfully")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])
X = df.drop("y", axis=1)
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)
model.fit(
    X_train,
    y_train
)
y_pred = model.predict(X_test)
accuracy = accuracy_score(
    y_test,
    y_pred
)
print("\nModel Accuracy:")
print(round(accuracy*100,2), "%")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)
result = df["y"].value_counts()
plt.figure(figsize=(6,4))
plt.bar(
    ["No Purchase","Purchase"],
    result.values
)
plt.title("Customer Purchase Distribution")
plt.xlabel("Customer Response")
plt.ylabel("Number of Customers")
plt.savefig("customer_purchase_distribution.png")
plt.show()
print("\nChart saved successfully")