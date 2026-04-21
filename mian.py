import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("cancer.csv")


data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data = data.drop(["id", "Unnamed: 32"], axis=1)
data = data.dropna()


X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\nModel Accuracy:")
print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("Decision Tree     :", accuracy_score(y_test, dt_pred))


print("\nSample Predictions (Actual vs Predicted):\n")

cancer_index = None
no_cancer_index = None


for i in range(len(y_test)):
    if y_test.iloc[i] == 1 and cancer_index is None:
        cancer_index = i
    elif y_test.iloc[i] == 0 and no_cancer_index is None:
        no_cancer_index = i
    
    if cancer_index is not None and no_cancer_index is not None:
        break


for label, idx in [("Cancer Case", cancer_index), ("No Cancer Case", no_cancer_index)]:
    sample = X_test[idx].reshape(1, -1)
    
    prediction = lr_model.predict(sample)
    prob = lr_model.predict_proba(sample)

    actual = "Cancer" if y_test.iloc[idx] == 1 else "No Cancer"
    predicted = "Cancer" if prediction[0] == 1 else "No Cancer"

    confidence = prob[0][1]*100 if prediction[0] == 1 else prob[0][0]*100

    print(f"{label}:")
    print(f"  Actual    : {actual}")
    print(f"  Predicted : {predicted}")
    print(f"  Confidence: {confidence:.2f}%\n")