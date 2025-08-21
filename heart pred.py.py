import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load CSV with semicolon delimiter
file_path = "cardio_train.csv"
df = pd.read_csv(file_path, sep=';')

print("✅ CSV loaded successfully!\n")
print(df.head())
print("\n📊 Dataset info:")
print(df.info())

# --- Data Preprocessing ---

# Convert age from days to years
df['age'] = (df['age'] / 365).astype(int)

# Drop 'id' column
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Filter unrealistic blood pressure values
df = df[(df['ap_hi'] > 80) & (df['ap_hi'] < 250)]
df = df[(df['ap_lo'] > 40) & (df['ap_lo'] < 200)]

# Scale numeric columns
scaler = StandardScaler()
num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\n✅ Data preprocessing complete!")
print(df.head())

# -----data visulization ------
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for plots
sns.set(style="whitegrid")

# 1. Histogram of Age
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age (standardized)')
plt.ylabel('Count')
plt.show()

# 2. Count plot for target variable 'cardio'
plt.figure(figsize=(6, 4))
sns.countplot(x='cardio', data=df)
plt.title('Count of Heart Disease (cardio) Labels')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 3. Count plot for Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=df)
plt.title('Gender Distribution')
plt.xlabel('Gender (1 = Male, 2 = Female)')
plt.ylabel('Count')
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.show()
# ------model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Prepare features and target
X = df.drop('cardio', axis=1)
y = df['cardio']

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
# -------print the best----------------
# Track best model
best_model_name = None
best_accuracy = 0.0

print("\n🔍 Model Accuracies:")
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")

    # Update best model if current one is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name

# Print best model
print(f"\n✅ Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
