import pandas as pd

# Load dataset
df = pd.read_csv("data/AccidentsBig.csv", low_memory=False)

print("Original Shape:", df.shape)

# Remove completely empty rows
df = df.dropna(how='all')

# Remove rows where target (Accident_Severity) is missing
df = df.dropna(subset=['Accident_Severity'])

print("After Cleaning Shape:", df.shape)

# Show first 5 rows
print(df.head())

print("\nDataset Info:")
print(df.info())

# -----------------------------
# STEP 9: Severity Distribution
# -----------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.countplot(x='Accident_Severity', data=df)
plt.title("Accident Severity Distribution")
plt.xlabel("Severity Class")
plt.ylabel("Count")
plt.savefig("images/severity_distribution.png")
plt.show()

#--------------------------
# select important features
#-------------------------

features = [
    'Speed_limit',
    'Weather_Conditions',
    'Road_Surface_Conditions',
    'Light_Conditions',
    'Number_of_Vehicles',
    'Number_of_Casualties',
    'Urban_or_Rural_Area'
]

X = df[features]
y = df['Accident_Severity']

print("Selected Features Shape:", X.shape)
print("Target Shape:", y.shape)

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# -----------------------------
# STEP 6: Decision Tree Model
# -----------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create model
dt_model = DecisionTreeClassifier(
    criterion='gini',      # can also try 'entropy'
    max_depth=5,           # prevents overfitting
    random_state=42
)

# Train model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate
print("\nDecision Tree Accuracy:",
      accuracy_score(y_test, y_pred_dt))

print("\nDecision Tree Classification Report:\n",
      classification_report(y_test, y_pred_dt))

#-------------------------
# Feature Importance
#--------------------------
print("\nFeature Importance:")
importances = dt_model.feature_importances_
feature_names = X.columns

for name, importance in zip(feature_names, importances):
    print(name, ":", importance)

# -----------------------------
# STEP 7: Logistic Regression
# -----------------------------

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)

# Train
log_model.fit(X_train, y_train)

# Predict
y_pred_log = log_model.predict(X_test)

# Evaluate
print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_log))

print("\nLogistic Regression Classification Report:\n",
      classification_report(y_test, y_pred_log))

# -----------------------------
# STEP 8: Naive Bayes Model
# -----------------------------

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

print("\nNaive Bayes Accuracy:",
      accuracy_score(y_test, y_pred_nb))

print("\nNaive Bayes Classification Report:\n",
      classification_report(y_test, y_pred_nb))

# -----------------------------
# STEP 9: KNN Model
# -----------------------------

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

print("\nKNN Accuracy:",
      accuracy_score(y_test, y_pred_knn))

print("\nKNN Classification Report:\n",
      classification_report(y_test, y_pred_knn))

# -----------------------------
# Model Comparison
# -----------------------------

print("\nModel Comparison:")

print("Decision Tree Accuracy:",
      accuracy_score(y_test, y_pred_dt))

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_log))

print("Naive Bayes Accuracy:",
      accuracy_score(y_test, y_pred_nb))

print("KNN Accuracy:",
      accuracy_score(y_test, y_pred_knn))

accuracy = [
    accuracy_score(y_test, y_pred_dt),
    accuracy_score(y_test, y_pred_log),
    accuracy_score(y_test, y_pred_nb),
    accuracy_score(y_test, y_pred_knn)
]

models = ['Decision Tree', 'Logistic Regression', 'Naive Bayes', 'KNN']

import matplotlib.pyplot as plt

plt.figure()

bars = plt.bar(models, accuracy)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Models")

# Show values on top
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{height:.2f}',
        ha='center',
        va='bottom'
    )

plt.ylim(0, 1)   # keeps scale clean
plt.savefig("images/acuracy_comparison.png")
plt.show()


# -----------------------------
# STEP 10: Precision, Recall, F1-Score Comparison
# -----------------------------
from sklearn.metrics import precision_score, recall_score, f1_score

# Store metrics for each model

models = ['Decision Tree', 'Logistic Regression', 'Naive Bayes', 'KNN']

precision = [
    precision_score(y_test, y_pred_dt, average='weighted'),
    precision_score(y_test, y_pred_log, average='weighted'),
    precision_score(y_test, y_pred_nb, average='weighted'),
    precision_score(y_test, y_pred_knn, average='weighted')
]

recall = [
    recall_score(y_test, y_pred_dt, average='weighted'),
    recall_score(y_test, y_pred_log, average='weighted'),
    recall_score(y_test, y_pred_nb, average='weighted'),
    recall_score(y_test, y_pred_knn, average='weighted')
]

f1 = [
    f1_score(y_test, y_pred_dt, average='weighted'),
    f1_score(y_test, y_pred_log, average='weighted'),
    f1_score(y_test, y_pred_nb, average='weighted'),
    f1_score(y_test, y_pred_knn, average='weighted')
]

# Print values
print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# -----------------------------
# precision, recall, f1 score comparison plot
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(models))

plt.figure()

bars1 = plt.bar(x - 0.2, precision, width=0.2, label='Precision')
bars2 = plt.bar(x, recall, width=0.2, label='Recall')
bars3 = plt.bar(x + 0.2, f1, width=0.2, label='F1 Score')

plt.xticks(x, models)
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Model Comparison (Precision, Recall, F1 Score)")
plt.legend()

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:.2f}',   # shows 2 decimal values
            ha='center',
            va='bottom'
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
plt.savefig("images/model_comparison.png")
plt.show()

# -----------------------------
# STEP 8: Confusion Matrix
# -----------------------------

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Decision Tree Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure()
sns.heatmap(cm_dt, annot=True, fmt='d')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("images/confusion_dt.png")
plt.show()

# Logistic Regression Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)

plt.figure()
sns.heatmap(cm_log, annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("images/confusion_lr.png")
plt.show()


# -----------------------------
# User Input Prediction
# -----------------------------

print("\nEnter Accident Conditions to Predict Severity")

speed = float(input("Enter Speed Limit: "))
weather = float(input("Enter Weather Condition code: "))
road_surface = float(input("Enter Road Surface Condition code: "))
light = float(input("Enter Light Condition code: "))
vehicles = float(input("Enter Number of Vehicles involved: "))
casualties = float(input("Enter Number of Casualties: "))
area = float(input("Enter Area (1 = Urban, 2 = Rural): "))

user_data = [[
    speed,
    weather,
    road_surface,
    light,
    vehicles,
    casualties,
    area
]]

prediction = dt_model.predict(user_data)

print("\nPredicted Accident Severity:", prediction[0])