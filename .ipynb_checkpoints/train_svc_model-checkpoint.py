from sklearn.svm import SVC
import pandas as pd
import pickle

# Load your dataset (make sure this path is correct)
df = pd.read_csv("Training.csv")

# Separate features and target
X = df.iloc[:, :-1]  # first 132 columns
y = df["prognosis"]  # last column

# Train the model
svc = SVC()
svc.fit(X, y)

# Save the trained model as svc.pkl
with open("svc.pkl", "wb") as f:
    pickle.dump(svc, f)

print("✅ Model trained and saved as svc.pkl")
