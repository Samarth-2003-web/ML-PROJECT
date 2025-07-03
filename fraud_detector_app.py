#!/usr/bin/env python
# coding: utf-8

# <p style="text-align: center;font-size:35px; color:blue;"><b>Machine Learning Using Fraud Dataset</b></p> 
# 

# <p style="text-align: center;font-size:35px; color:red;"><b>--------------------------------------------------------------------------</b></p> 
# 

# In[63]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# <p style="text-align: center;font-size:35px; color:red;"><b>Load Dataset</b></p> 
# 

# In[64]:


data= pd.read_csv("ML_Data.csv")


# In[65]:


data.shape


# In[66]:


data.tail()


# In[67]:


data.info()


# In[68]:


data.isnull().sum()


# In[69]:


data.describe()


# 
# <p style="text-align: center;font-size:35px; color:red;"><b>(Yes,No) count</b></p> 
# 

# In[70]:


data['is_fraud'].value_counts()


# <p style="text-align: center;font-size:35px; color:red;"><b>Skewness</b></p> 
# 

# ***
# 
# 

# In[71]:


numeric_cols = data.select_dtypes(include=[np.number])
print(numeric_cols.skew().sort_values(ascending=False))


# <p style="text-align: center;font-size:35px; color:red;"><b>Checking Outliers</b></p> 
# 

# In[72]:


outlier= {}
for col in numeric_cols.columns:
    Q1=data[col].quantile(0.25)
    Q3=data[col].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers = data[(data[col] < lower_bound) & (data[col] > upper_bound)]
    outlier[col] = len(outliers)
outlier


# 
# <p style="text-align: center;font-size:35px; color:red;"><b>Correlation</b></p> 
# 

# In[73]:


correlation = data.select_dtypes(include=[np.number]).corr()
correlation


# <p style="text-align: center;font-size:35px; color:red;"><b>Correlation heatmap</b></p> 
# 

# In[74]:


numeric_data=data.select_dtypes(include=[np.number])
correlation_matrix=numeric_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

threshold=0.95
upper=correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper.columns if any(abs(upper[column]) > threshold)]

print("Highly correlated features to drop:")
print(high_corr_features)
data_reduced=data.drop(columns=high_corr_features)
#data['transaction_amount_log'] = np.log1p(data['transaction_amount'])



# In[75]:


categorical_cols = [
    'browser', 'os', 'location_change', 'email_verified', 'phone_verified',
    'transaction_type', 'merchant_category'
]

numerical_cols = [
    'transaction_amount', 'transaction_hour', 'user_age', 'account_age_days',
    'num_transactions_24h', 'num_failed_logins', 'ip_risk_score', 'previous_fraud_activity'
]



# In[76]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Use this version:
encoder = OneHotEncoder(handle_unknown='ignore')

column_transformer = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numerical_cols),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)


# <p style="text-align: center;font-size:35px; color:red;"><b>applying model </b></p> 
# 

# In[77]:


pipeline = Pipeline([
    ('preprocessor', column_transformer),
    ('classifier', LogisticRegression(class_weight='balanced'))
])



# In[78]:


y = data['is_fraud']
X = data.drop(columns='is_fraud')


# <span style="font-size:30px; color:blue;"><b>Training  & Testing</b></span>
# 

# In[79]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2 ,random_state=42)


# In[80]:


pipeline.fit(X_train, y_train)


# In[81]:


#print(X_train.columns)  # Print the column names of X_train


# In[82]:


y_pred = pipeline.predict(X_test)
#print(y_pred)


# <p style="text-align: center;font-size:35px; color:red;"><b>Report </b></p> 
# 

# In[83]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean() * 100)

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
print("F1-Score:", scores.mean())


# In[84]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# <div class="alert alert-block alert-warning">
# <span style="font-size:30px; color:blue;"><b>SVM </b></span>
# 
# </div>

# In[85]:


corr_matrix = data.corr(numeric_only=True)

target_corr = corr_matrix['is_fraud'].drop('is_fraud')

selected_features = target_corr[abs(target_corr) > 0.3].index.tolist()
print("Selected features based on correlation:", selected_features)

X = data[selected_features]
y = data['is_fraud']

X


# In[86]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# <span style="font-size:30px; color:blue;"><b>Training & Testing </b></span>
# 

# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

y_svm_pred = svm.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_svm_pred))
print("Accuracy Score:", accuracy_score(y_test, y_svm_pred))
cv_scores = cross_val_score(svm, X_scaled, y, cv=10, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:",cv_scores.mean() * 100)


# In[88]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:





# 
# <p style="text-align:center; font-size:35px; color:red; text-shadow: 4px 4px 8px #aaa;">
#   <b>ROC Curve</b>
# </p>
# 

# In[89]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_svm_pred)
auc_log = roc_auc_score(y_test, y_pred)
auc_svm = roc_auc_score(y_test, y_svm_pred)


# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.2f})', color='blue')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[90]:


import joblib

# Save your pipeline
joblib.dump(pipeline, 'pipeline_model.pkl')


# In[91]:


import pandas as pd
import joblib

# Load your pipeline model
pipeline = joblib.load('pipeline_model.pkl')

# One sample input
input_data = pd.DataFrame([{
    'browser': 'Chrome',
    'os': 'Windows',
    'location_change': 'No',
    'email_verified': 'Yes',
    'phone_verified': 'No',
    'transaction_type': 'Online',
    'merchant_category': 'Retail',

    'transaction_amount': 250.75,
    'transaction_hour': 14,
    'user_age': 28,
    'account_age_days': 365,
    'num_transactions_24h': 5,
    'num_failed_logins': 1,
    'ip_risk_score': 45,
    'previous_fraud_activity': 0
}])

# Predict
prediction = pipeline.predict(input_data)
print("Prediction:", "Fraud" if prediction[0] == 1 else "Not Fraud")


# In[92]:


import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline model
pipeline = joblib.load('pipeline_model.pkl')

# Title
st.title("🔍 Real-Time Fraud Detection App")

st.markdown("Fill the transaction details below:")

# Input form
with st.form("fraud_form"):
    browser = st.selectbox("Browser", ['Chrome', 'Firefox', 'Safari', 'Edge', 'Other'])
    os = st.selectbox("Operating System", ['Windows', 'macOS', 'Linux', 'Android', 'iOS'])
    location_change = st.selectbox("Location Change", ['Yes', 'No'])
    email_verified = st.selectbox("Email Verified", ['Yes', 'No'])
    phone_verified = st.selectbox("Phone Verified", ['Yes', 'No'])
    transaction_type = st.selectbox("Transaction Type", ['Online', 'In-Store'])
    merchant_category = st.selectbox("Merchant Category", ['Retail', 'Food', 'Electronics', 'Travel'])

    transaction_amount = st.number_input("Transaction Amount", min_value=0.0)
    transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
    user_age = st.slider("User Age", 10, 100, 30)
    account_age_days = st.number_input("Account Age (days)", min_value=0)
    num_transactions_24h = st.number_input("No. of Transactions in 24h", min_value=0)
    num_failed_logins = st.number_input("No. of Failed Logins", min_value=0)
    ip_risk_score = st.slider("IP Risk Score", 0, 100, 50)
    previous_fraud_activity = st.selectbox("Previous Fraud Activity", [0, 1])

    submitted = st.form_submit_button("Predict Fraud")

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        'browser': browser,
        'os': os,
        'location_change': location_change,
        'email_verified': email_verified,
        'phone_verified': phone_verified,
        'transaction_type': transaction_type,
        'merchant_category': merchant_category,
        'transaction_amount': transaction_amount,
        'transaction_hour': transaction_hour,
        'user_age': user_age,
        'account_age_days': account_age_days,
        'num_transactions_24h': num_transactions_24h,
        'num_failed_logins': num_failed_logins,
        'ip_risk_score': ip_risk_score,
        'previous_fraud_activity': previous_fraud_activity
    }])

    prediction = pipeline.predict(input_data)[0]
    result = " Fraud Detected!" if prediction == 1 else "Transaction is Legitimate"
    st.success(result)


# In[ ]:



    
