# %%
# LOAN APPROVAL

# PROBLEM DESCRIPTION AND HOW MACHINE LEARNING CAN HELP

# Loan approval is the formal decision by a lender (like a bank or credit union) that you are qualified to receive the money you've asked to borrow, based on their assessment of your ability to repay it.

# Machine Learning can analyze large amounts of customer data and automatically learn patterns that indicate whether a person is eligable for the loan. By training a model on past loan approval records, it can:

# Predict risk levels accurately based on key health indicators.

# Support lender in making faster and more informed decisions.

# Identify hidden patterns that may not be obvious in manual analysis.

# Enable early intervention, which can improve  outcomes and reduce loses.

# %%
# !pip install xgboost
import xgboost as xgb

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import sklearn
import xgboost as xgb
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys


# %%
print("Platform:", sys.platform)
print("Python version:", sys.version)
print("---" * 47)

# Libraries versions
print("matplotlib version:", matplotlib.__version__)
print("seaborn version:", sns.__version__)
print("xgboost version:", xgb.__version__)
print("sklearn version:", sklearn.__version__)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)

# %%
df = pd.read_csv('loan_approval.csv')
df.head()

# %%
df.isnull().sum()

# %%
df.info()

# %%
df.describe()

# %%
df.shape

# %%
df.dtypes

# %%
(df.isnull().sum() >=0).sum()

# %%
df.duplicated()

# %%
df.columns

# %%
numerical = list(df.dtypes[(df.dtypes=='int64') | (df.dtypes=='float64')].index)
categorical = list(df.dtypes[(df.dtypes=='object') | (df.dtypes=='bool')].index)
numerical, categorical

# %%
df['loan_approved'] = df['loan_approved'].replace({'True':1,'False':0})
df['loan_approved'].value_counts()

# %%
df['loan_approved'] = df['loan_approved'].T.astype(int)
df[['loan_approved']]

# %%
df['loan_approved'] = df['loan_approved'].replace({'True':1,'False':0})
df['loan_approved'].value_counts()

# %%
# VISUALIZATION OF PATTERNS

# %%
sns.countplot(x='loan_approved', data=df)
plt.title('loan_approved')
plt.xlabel('loan_approved')
plt.ylabel('Count')
plt.show()

# %%
data_numeric = df.select_dtypes(include=['number']) 
plt.figure(figsize=(9, 6))
sns.heatmap(data_numeric.corr(), annot=True, linewidths=.5, cmap='YlOrRd')
plt.title('Heatmap showing correlations between numerical data')
plt.show()

# %%
df[numerical].hist(figsize = (13, 13))

# %%
# VISUALIZE CONTINUOUS VARIABLE VS TARGET VARIABLE
sns.boxplot(x='loan_approved', y='years_employed', data=df)  
plt.title('years_employed vs Loan Approval')
plt.xlabel('Loan Approval Status')
plt.ylabel('years_employed')
plt.show()

# %%
sns.pairplot(df, hue='loan_approved', vars=['income', 'credit_score', 'loan_amount', 'years_employed', 'points', 'loan_approved'])
plt.suptitle('Pairplot of Numerical Features Colored by loan_approval')
plt.show()

# %%
# SPLITTING THE DATA INTO TRAIN /VALIDATION/ TEST-SPLIT
# We will split it into 60% 20% 20% distribution

# %%
# split the loan_approval dataset into train,val and test sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# %%
#check fo the length 
len(df_train), len(df_val), len(df_test)

# %%
df['loan_approved'].value_counts()

# %%
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.loan_approved.values
y_val = df_val.loan_approved.values
y_test = df_test.loan_approved.values

del df_train['loan_approved']
del df_val['loan_approved']
del df_test['loan_approved']

# %%
# PERFORMING ONE HOT ENCODING BEFORE WE TRAIN OUR DATA

# %%
train_dicts = df_train.to_dict(orient = 'records')
dv= DictVectorizer(sparse = False) 
X_train = dv.fit_transform(train_dicts)

# %%
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model

# %%
model.fit(X_train,y_train)

# %%
val_dicts = df_val.to_dict(orient = 'records')
X_val = dv.transform(val_dicts)

# %%
test_dicts = df_test.to_dict(orient = 'records')
X_test = dv.transform(test_dicts)

# %%
# TRAIN SEVERAL MODELS AND THEN DO FINE TUNING

# %%
# TRAINING A LOGISTIC REGRESSION MODEL WITH SCIKIT-LEARN

# %%
model.coef_

# %%
model.coef_[0].round(3)

# %%
model.intercept_

# %%
model.intercept_[0]

# %%
model.fit(X_train,y_train)

# %%
model.predict_proba(X_test).round(2)

# %%
y_pred = model.predict(X_test)
y_pred

# %%
reg_params = [0.01, 0.1, 1, 2, 10, 100]
reg_params_scores = []
for param in reg_params:
    model = LogisticRegression(solver = 'liblinear', C = param,max_iter = 1000, random_state = 42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)    
    param_score = 100 * (y_pred == y_val).mean()

    reg_params_scores += [round(param_score, 3)]
param_scores = pd.Series(reg_params_scores, index = reg_params, name = "parameters_scores")
param_scores

# %%
#LogisticRegression MODEL 

# %%
param_scores.plot(marker='o')
plt.xlabel("C (Regularization Strength)")
plt.ylabel("Validation Accuracy (%)")
plt.title("C parameter tuning for Logistic Regression")
plt.show()

# %%
X = df.drop('loan_approved', axis=1)  # features
y = df['loan_approved']               #target

# %%
print(X.shape)
print(y.value_counts())

# %%
# TRAINING A DECISION TREEE CLASSIFFIER

# %%
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# %%
dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, y_train)
 
y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print('train', auc)

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('val', auc)

# %%
print(export_text(dt, feature_names=dv.get_feature_names_out()))

# %%
depths = [1, 2, 3, 4, 5, 6, 10, 15, 20, None]
 
for depth in depths: 
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
     
    # remember we need the column with negative scores
    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
     
    print('%4s -> %.3f' % (depth, auc))

# %%
scores = []
 
for d in [4, 5, 6]:
    for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
        dt.fit(X_train, y_train)
 
        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
         
        scores.append((d, s, auc))
 
columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.head()

# %%
df_scores.sort_values(by='auc', ascending=False).head()

# %%
# index - rows
df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)
sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")

# %%
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

# %%
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
 
y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)
 
rf.predict_proba(X_val[[0]])

# %%
scores = []
 
for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
 
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
     
    scores.append((n, auc))
 
df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])
df_scores

# %%
plt.plot(df_scores.n_estimators, df_scores.auc)

# %%
scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))

# %%
columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.groupby("max_depth")["auc"].mean().round(4)

# %%
for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)

plt.legend()

# %%
max_depth = 15

# %%
scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))

# %%
columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

# %%
colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend()

# %%
min_samples_leaf = 1

# %%
f = RandomForestClassifier(n_estimators=200,
                            max_depth=max_depth,
                            min_samples_leaf= min_samples_leaf,
                            random_state=42)
rf.fit(X_train, y_train)

# %%
# XGBoost Classifier

# %%
import xgboost as xgb
print(xgb.__version__)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

watchlist = [(dtrain, "train"), (dval, "eval")]

# %%
xgb_params = {
    'eta' :0.3,
    'max_depth' :6,
    'min_child_weight' :1,
    'objective' : 'binary:logistic',
    'nthread' :8,

    'seed' :1,
    'verbosity' :1,
}

model = xgb.train(xgb_params,dtrain, num_boost_round=10)

# %%
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# %%
watchlist = [(dtrain, 'train'), (dval, 'val')]

# %%
xgb_params = {
    'eta' :0.3,
    'max_depth' :6,
    'min_child_weight' :1,
    'objective' : 'binary:logistic',
    'nthread' :8,
    'seed' :1,
    'verbosity' :1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200, evals=watchlist)

# %%
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# %%
# PICKING THE BEST MODEL
# We will pick  RANDOM FOREST CLASSIFIER 

# %%
# SAVING MODEL

# %%
import pickle

# %%
output_file = 'Random_Forest_Model.bin'
output_file

# %%
f_out = open(output_file, 'wb')
 
pickle.dump((dv, rf), f_out)
 
f_out.close()

# %%
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

# %%
input_file = 'Random_Forest_Model.bin'


with open(input_file,'rb') as f_in:
    
    dv, rf = pickle.load(f_in)

rf

# %%
import pandas as pd

# client dictionary
client = {'name':'Allison Hill',
          'city':'Mariastad',
          'income':33278,
          'credit_score':584, 
          'loan_amount':15446,
          'years_employed':13,
          'points':45.0,
          'loan_approved': False
         }

# Convert to DataFrame
client_df = pd.DataFrame([client])

# Transform features using the loaded encoder
X_client = dv.transform(client_df.to_dict(orient='records'))

# Make prediction
pred_class = rf.predict(X_client)[0]
pred_prob = rf.predict_proba(X_client)[0, 1]  # probability of positive class

# Display input and results
print("client information:")
print(client_df)

print(f"Predicted class: {pred_class}")
print(f"Predicted probability of loan_approved: {pred_prob:.2f}")

# Define approval if necessary
if pred_class == 1:
    print("Loan_approval should be considered for this client.")


# %%



