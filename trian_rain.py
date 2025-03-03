import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

file_path = "seattle-weather2.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)

label_encoders = {}
for col in ["weather"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  

X = df.drop(columns=["weather"])
y = df["weather"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
    "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Genetic Algorithm (MLP)": MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
}

results = {}
train_times = {}
cv_scores = {}

for name, model in models.items():
    start_time = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_cv_score = np.mean(scores)

    end_time = time.time()
    train_time = end_time - start_time
    
    train_times[name] = train_time
    results[name] = accuracy
    cv_scores[name] = mean_cv_score

    print(f"{name} | Train Time: {train_time:.4f} sec | Accuracy (Test Set): {accuracy:.4f} | Mean CV Score: {mean_cv_score:.4f}")

voting_clf = VotingClassifier(
    estimators=[(name, models[name]) for name in models],
    voting='soft'
)

start_time = time.time()
voting_clf.fit(X_train, y_train)
end_time = time.time()

y_pred_voting = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, y_pred_voting)

results["Voting Classifier"] = voting_accuracy
train_times["Voting Classifier"] = end_time - start_time
cv_scores["Voting Classifier"] = np.mean(cross_val_score(voting_clf, X_train, y_train, cv=5))

print(f"Voting Classifier | Train Time: {train_times['Voting Classifier']:.4f} sec | Accuracy: {voting_accuracy:.4f} | Mean CV Score: {cv_scores['Voting Classifier']:.4f}")

max_accuracy = max(results.values())
best_models = [name for name, acc in results.items() if acc == max_accuracy]

if len(best_models) > 1:
    best_models = [m for m in best_models if m != "Voting Classifier"]
    best_model = min(best_models, key=lambda m: train_times[m]) if best_models else "Voting Classifier"
else:
    best_model = best_models[0]

final_model = {"model": models[best_model] if best_model != "Voting Classifier" else voting_clf, "label_encoders": label_encoders}
joblib.dump(final_model, "best_model.pkl")

print(f"Best Model: {best_model} | Accuracy: {results[best_model]:.4f}")

plt.figure(figsize=(12, 6))  
plt.bar(results.keys(), results.values(), color='skyblue')
plt.xticks(rotation=45, ha='right', fontsize=12) 
plt.ylabel("Accuracy", fontsize=14)
plt.xlabel("Model", fontsize=14)
plt.title("Model Accuracy Comparison", fontsize=16)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  
plt.show()

plt.figure(figsize=(12, 6))  
plt.bar(train_times.keys(), train_times.values(), color='salmon')
plt.xticks(rotation=45, ha='right', fontsize=12)  
plt.ylabel("Training Time (seconds)", fontsize=14)
plt.xlabel("Model", fontsize=14)
plt.title("Model Training Time Comparison", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  
plt.show()