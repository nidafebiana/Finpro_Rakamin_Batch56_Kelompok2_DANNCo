# Rakamin Final Project - Batch 56
## Anggota
- 👨‍💼 Project Manager         → Dadin Tajudin
- 🛠️ Data Engineer           → Athariq Marsha Nugraha
- 🧑‍🔬 Data Scientist          → Nada Paradita
- 📊 Business & Data Analyst → Nida Febiana

# 🚀 Employee Churn Prediction: A Data-Driven Strategy for Workforce Retention - Rakamin Finpro DS56 Kelompok 2.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modeling-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-yellow)

---

## Introduction  
This project aims to build a machine learning model to predict *employee churn* (employees leaving within the first 1–3 months). With this prediction, companies can take preventive actions to improve retention, reduce recruitment and training costs, and maintain organizational stability.  

---

## Problem Statement  
The company is facing a high turnover rate:  
- Out of 1,000 new employees, **629 (62.9%) churned within the first 1–3 months**.  
- This number is far above the global benchmark (≤20%).  
- Impact: recruitment & onboarding costs of ~IDR 2.5 billion, loss of productivity, and workplace culture disruption.  

**Goals:**  
- Build a churn prediction model with *Recall ≥ 80%*, *F2 ≥ 75%*, and *AUC ≥ 80%* and implemented into Streamlit.  
- Reduce recruitment and training costs by 15% within 6 months.  
- Reduce churn by 12% within 3 months of implementing the model.  

---

## Dataset  
- **Size:** 1,000 employees  
- **Features:** 19 original + 6 engineered features  
- **Target:** *Churn* (Yes/No)  
- **Key Characteristics:**  
  - 70% male, majority are single and diploma graduates  
  - 50% work in urban areas  
  - 40% churn in the first month → onboarding stage is critical  
- **Data Quality:**  
  - No missing values  
  - No duplicates  
  - No multicollinearity (correlations < 0.7)  

---

## Methodology  

### 1. Exploratory Data Analysis (EDA)  
- Highest churn observed among employees with:  
  - Low *job satisfaction*  
  - Long *working hours*  
  - Longer *distance to office*  
- Even top performers churn at 50% → issues go beyond performance.  

### 2. Data Preprocessing  
- Checked duplicates & missing values → none  
- Outlier handling  
- Feature extraction (education, work location, churn period, etc.)  
- Categorical encoding (Label & One-Hot Encoding)  
- Train-test split (80:20)  
- Addressed class imbalance  

### 3. Model Selection  
- Baseline: Logistic Regression, KNN  
- Tree-based models: Random Forest, Decision Tree, CatBoost, XGBoost  
- **XGBoost** chosen for its balanced performance (Recall, F2, AUC).  

### 4. Model Training & Evaluation  
- **Best model: XGBoost**  
- Test data (n=200):  
  - Recall: **91.26%**  
  - F2 Score: **86.59%**  
  - ROC-AUC: **77.9%** (close to 80% target)

 ### 5. Key Predictors (SHAP Analysis)
 - Target Achievement
 - Job Satisfaction
 - Manager Support
 - Perfomance Gap
 - Distance to Office

---

## Results & Visualization  
- **Business Impact:**  
  - Potential cost savings of up to **Rp 2.3 billion** (if 100% effective).  
  - Even with 25% effectiveness → savings of Rp 575M, exceeding target (Rp 377M).  
- **Key Visualizations:**  
  - Feature correlation heatmap  
  - Churn distribution by demographics & work factors  
  - Confusion matrix for XGBoost  
  - SHAP feature importance  
  - Fairness analysis (gender, marital status, education, location) → model proven unbiased.  

---

## Deployment  
- Model deployed with **Streamlit App** in two modes:  
  - Individual prediction  
  - Batch prediction  
- Access links:  
  - [Streamlit Deployment](https://bit.ly/Deployment_Kelompok2)  
  - [Batch Simulation Data](https://bit.ly/Data_Simulation_Kelompok2)  

---

## Challenges & Learnings  

**Challenges:**  
- Handling class imbalance in churn data  
- Balancing the trade-off between *Recall* and *False Positives*  
- Hyperparameter tuning to improve AUC  

**Learnings:**  
- *Recall* is more critical than *Precision* in HR → false alarms are acceptable to avoid missing real churn cases.  
- Non-financial factors (job satisfaction, manager support, distance) drive churn more than salary.  
- Fairness testing is essential to ensure no bias toward gender, marital status, education, or work location.  

---

## Conclusion & Next Steps  
- The XGBoost model achieved its main targets (*Recall* > 80%, *F2* > 75%).  
- Implementation can significantly reduce churn and HR costs.  
- **Next Steps:**  
  - Retrain model quarterly with new data.  
  - Integrate into HR systems for real-time prediction.  
  - Focus on onboarding, job satisfaction monitoring, and manager support as key retention strategies.  

---

### 📁 Folder Sturcture
```
project-folder/
├── app.py
├── eda_module.py
├── utils.py
├── employee_churn_prediction_updated.csv
├── requirements.txt
└── ...
```
---
### ⚙️ Installation
1. **Clone the repository or copy the files to your local folder**
2. **Activate a virtual environment** (optional but recommended)
3. **Install dependencies**

```bash
pip install -r requirements.txt

If requirements.txt is not available, install manually:
pip install streamlit pandas scikit-learn matplotlib seaborn

Running The Application
streamlit run app.py
