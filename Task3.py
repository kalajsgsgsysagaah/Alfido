# Task 3: Titanic Survival Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset (Titanic dataset from seaborn or CSV)
# If using seaborn: 
import seaborn as sns
titanic = sns.load_dataset("titanic")

# 2. Select useful columns
data = titanic[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]

# 3. Handle missing values
data["age"].fillna(data["age"].median(), inplace=True)
data["embarked"].fillna(data["embarked"].mode()[0], inplace=True)

# 4. Encode categorical variables
le_sex = LabelEncoder()
data["sex"] = le_sex.fit_transform(data["sex"])

le_embarked = LabelEncoder()
data["embarked"] = le_embarked.fit_transform(data["embarked"])

# 5. Features & Target
X = data.drop("survived", axis=1)
y = data["survived"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
print("🚢 Titanic Survival Prediction (Logistic Regression)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
