# 🫀 Heart Disease Prediction

## 📌 Overview
This project predicts the likelihood of **cardiovascular disease (CVD)** using machine learning.  
It uses the **heart_train.csv** dataset, performs preprocessing, exploratory data analysis (EDA), and trains multiple models to identify the best one for classification.

---

## 📊 Dataset
- **File Used:** `heart_train.csv`
- **Format:** CSV (semicolon-delimited)
- **Features:**
  - **age** → Patient’s age (in days, converted to years in preprocessing)  
  - **gender** → 1 = Male, 2 = Female  
  - **height** → Height (cm)  
  - **weight** → Weight (kg)  
  - **ap_hi** → Systolic blood pressure  
  - **ap_lo** → Diastolic blood pressure  
  - **cholesterol** → Cholesterol levels (1 = normal, 2 = above normal, 3 = well above normal)  
  - **gluc** → Glucose level (1 = normal, 2 = above normal, 3 = well above normal)  
  - **smoke** → Smoking status (0 = no, 1 = yes)  
  - **alco** → Alcohol intake (0 = no, 1 = yes)  
  - **active** → Physical activity (0 = no, 1 = yes)  
  - **cardio** → Target variable (0 = No disease, 1 = Has disease)  

---

## 🛠️ Project Workflow
1. **Data Preprocessing**
   - Convert `age` from days → years  
   - Drop `id` column if present  
   - Filter out unrealistic blood pressure values  
   - Standardize numeric features (`age`, `height`, `weight`, `ap_hi`, `ap_lo`) using `StandardScaler`

2. **Exploratory Data Analysis (EDA)**
   - Age distribution histogram  
   - Gender distribution  
   - Target distribution (`cardio`)  
   - Correlation heatmap of features  

3. **Model Training**
   Models implemented:
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - Decision Tree  
   - Random Forest  

4. **Model Evaluation**
   - Accuracy score for each model  
   - Best model is identified and reported  

---

## 🚀 How to Run
### Prerequisites
Install Python libraries:
```bash
pip install pandas scikit-learn matplotlib seaborn
