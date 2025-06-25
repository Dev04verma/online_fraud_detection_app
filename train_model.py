import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("sampled_fraud_data.csv")

# Drop irrelevant columns
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# One-hot encode 'type'
df = pd.get_dummies(df, columns=['type'], drop_first=False)

# Features & Target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to oversample fraud cases
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Train model with class balancing
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_resampled, y_resampled)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'model.pkl')
