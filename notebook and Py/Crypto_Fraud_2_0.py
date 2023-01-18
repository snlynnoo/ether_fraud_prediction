# # Importing libraries and Dataset
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# from google.colab import drive
# drive.mount('/content/drive')

# Importing dataset
data = pd.read_csv("data/datasets/txn_dataset.csv")

# viewing features and types in dataset
data.info()

# Drop first two columns (All rows, col 3 to end)
df = data.iloc[:, 3:]
df.head()
print(df.columns)

# Renaming The Features Names To The Same Names Without Spaces
df = df.copy()
for i in df.columns:
    df.rename(columns=str.strip,inplace = True)

# Checking uniques values in the column
df['ERC20 most sent token type'].nunique()

# Checking uniques values in the column
df['ERC20_most_rec_token_type'].nunique()

# Dropping unuse conlumns since they have many levels and just the names
df.drop(['ERC20 most sent token type','ERC20_most_rec_token_type'],axis =1,inplace = True)

# df.shape
# df.describe()

# Listing features having zero varience
var_zero = list(df.var()[df.var() == 0].keys())
print(pd.DataFrame(var_zero))

#droping varience zero featues
df.drop(var_zero ,inplace = True, axis = 1)
# df.describe()
# df.shape


# # Data preprocessing

# Checking duplicates 
df[df.duplicated()== True].count()

# Dropping duplicates values
df.drop_duplicates(inplace = True)

# Verifying duplicates again
df[df.duplicated()== True].count()

# Checking null values
df.isnull().sum()

# Since the number of missing values are less than 1 %, the rows will be dropped
df1 = df.dropna()
df1.isnull().sum()

# # Feature Elimation using correlation matrix

# Observing correlations between features
corr_1 = df1.corr()
f, ax = plt.subplots(figsize = (40, 30))
sns.heatmap(corr_1, annot = True, cmap='PiYG', center = 0, fmt = '.1f', square = True)

# Dropping highly correlated features (cutoff correlation = 0.5 since there are 50 variables)
drop_feature_1 = ['total transactions (including tnx to create contract', 'total ether sent contracts', 'max val sent to contract', 'ERC20 avg val rec',
        'ERC20 avg val rec','ERC20 max val rec', 'ERC20 min val rec', 'ERC20 uniq rec contract addr', 'max val sent', 'ERC20 avg val sent',
        'ERC20 min val sent', 'ERC20 max val sent', 'Total ERC20 tnxs', 'avg value sent to contract', 'Unique Sent To Addresses',
        'Unique Received From Addresses', 'total ether received', 'ERC20 uniq sent token name', 'min value received', 'min val sent', 'ERC20 uniq rec addr' ]
df1.drop(drop_feature_1, axis = 1, inplace = True)

corr_2 = df1.corr()
f, ax = plt.subplots(figsize = (20, 10))
sns.heatmap(corr_2, annot = True, cmap='PiYG', center = 0, fmt = '.1f', square = True)

# Dropping highly correlated features again (cutoff correlation = 0.5)
drop_feature_2 = 'max value received'
df1.drop(drop_feature_2, axis = 1, inplace = True)

corr_3 = df1.corr()
f, ax = plt.subplots(figsize = (20, 10))
sns.heatmap(corr_3, annot = True, cmap='PiYG', center = 0, fmt = '.1f', square = True)

# saving column names
columns = df1.columns
columns

# Plotting boxplot for all feature to see their distribution 

fig, axes = plt.subplots(6, 3, figsize=(14, 14), constrained_layout =True)
plt.subplots_adjust(wspace = 0.7, hspace=0.8)
plt.suptitle("Distribution of features",y=0.95, size=18, weight='bold')

ax = sns.boxplot(ax = axes[0,0], data=df1, x=columns[1])
ax.set_title(f'Distribution of {columns[1]}')

ax1 = sns.boxplot(ax = axes[0,1], data=df1, x=columns[2])
ax1.set_title(f'Distribution of {columns[2]}')

ax2 = sns.boxplot(ax = axes[0,2], data=df1, x=columns[3])
ax2.set_title(f'Distribution of {columns[3]}')

ax3 = sns.boxplot(ax = axes[1,0], data=df1, x=columns[4])
ax3.set_title(f'Distribution of {columns[4]}')

ax4 = sns.boxplot(ax = axes[1,1], data=df1, x=columns[5])
ax4.set_title(f'Distribution of {columns[5]}')

ax5 = sns.boxplot(ax = axes[1,2], data=df1, x=columns[6])
ax5.set_title(f'Distribution of {columns[6]}')

ax6 = sns.boxplot(ax = axes[2,0], data=df1, x=columns[7])
ax6.set_title(f'Distribution of {columns[7]}')

ax7 = sns.boxplot(ax = axes[2,1], data=df1, x=columns[8])
ax7.set_title(f'Distribution of {columns[8]}')

ax8 = sns.boxplot(ax = axes[2,2], data=df1, x=columns[9])
ax8.set_title(f'Distribution of {columns[9]}')

ax9 = sns.boxplot(ax = axes[3,0], data=df1, x=columns[10])
ax9.set_title(f'Distribution of {columns[10]}')
 
ax10 = sns.boxplot(ax = axes[3,1], data=df1, x=columns[11])
ax10.set_title(f'Distribution of {columns[11]}')

ax11 = sns.boxplot(ax = axes[3,2], data=df1, x=columns[12])
ax11.set_title(f'Distribution of {columns[12]}')
 
ax12 = sns.boxplot(ax = axes[4,0], data=df1, x=columns[13])
ax12.set_title(f'Distribution of {columns[13]}')
 
ax13 = sns.boxplot(ax = axes[4,1], data=df1, x=columns[14])
ax13.set_title(f'Distribution of {columns[14]}')
 
ax14 = sns.boxplot(ax = axes[4,2], data=df1, x=columns[15])
ax14.set_title(f'Distribution of {columns[15]}')
 
ax15 = sns.boxplot(ax = axes[5,0], data=df1, x=columns[16])
ax15.set_title(f'Distribution of {columns[16]}')
 
ax16 = sns.boxplot(ax = axes[5,1], data=df1, x=columns[17])
ax16.set_title(f'Distribution of {columns[17]}')
plt.show()

# Dropping features that has almost 0 values
df1.drop(['min value sent to contract', 'ERC20 uniq sent addr.1'], axis=1, inplace=True)
# print(df1.shape)
# df1.head()

# Observing class for target variable
df1['FLAG'].value_counts()
#sns.countplot(x="FLAG", data=df1) 
pd.DataFrame(df1['FLAG'].value_counts()).plot(kind = 'bar', figsize = (6,6))

# Splitting dependent and independent variables
y = df1.iloc[:, 0]
X = df1.iloc[: , 1:]
print(y.shape, X.shape)

# Oversampling using SMOTE to balance the class
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(y_resampled.value_counts())

# Checking balanced class
pd.DataFrame(y_resampled.value_counts()).plot(kind = 'bar', figsize = (6,6))


# # Combine X_resampled and y_resampled for cleaned dataset
# df = pd.DataFrame(data=X_resampled, columns=X.columns)
# df['label'] = y_resampled
# df.to_csv('export_2.csv')

# Exploring feature importance uisng RF RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


rf = RandomForestClassifier()
rfe = RFE(estimator=rf, n_features_to_select=3)
rfe.fit(X_resampled, y_resampled)

# get feature importance
importance = rfe.estimator_.feature_importances_

# get column names
column_names = X.columns

# create a dictionary of feature importance and column names
feature_importance = dict(zip(column_names, importance))
ranking = dict(zip(column_names, rfe.ranking_))

ranking_df = pd.DataFrame.from_dict(ranking, orient='index', columns=['ranking'])
ranking_df = ranking_df.sort_values(by='ranking')
# print(ranking_df)

# Checking skewness of variables
df1.skew()

# Feature transformation using PowerTransformer
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="yeo-johnson", standardize=False)
X_transformed = pt.fit_transform(X_resampled)
X_transformed = pd.DataFrame(X_transformed, columns=X_resampled.columns)
# X_transformed

# Data partitioning 70:30 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_resampled, test_size = 0.3, random_state = 123)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# # Model building

#Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=123)
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
plot_confusion_matrix(LR, X_test, y_test)


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state=123)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)

print(classification_report(y_test, y_pred_RF))
print(confusion_matrix(y_test, y_pred_RF))
plot_confusion_matrix(RF, X_test, y_test)

# Feature importance
importance = RF.feature_importances_


#XGBoost classifer

from xgboost import XGBClassifier
XGB_C = XGBClassifier(random_state=123)
XGB_C.fit(X_train, y_train)
y_pred_XGB = XGB_C.predict(X_test)

print(classification_report(y_test, y_pred_XGB))
print(confusion_matrix(y_test, y_pred_XGB))
plot_confusion_matrix(XGB_C, X_test, y_test)

# plot feature importances
from xgboost import plot_importance
plt.rcParams["figure.figsize"] = [10, 10]
plot_importance(XGB_C)


# # Saving models

# Saving Logistic Regression model
import pickle
pickle_out = open('LR.pickle', 'wb')
pickle.dump(LR, pickle_out)
pickle_out.close()

# Saving Random Forest model
import pickle
pickle_out = open('RF.pickle', 'wb')
pickle.dump(RF, pickle_out)
pickle_out.close()

#Saving XGBooast model
XGB_C.save_model("XGB_C_Final.json")



# # Predicting unseen data 

# Test case
test_data  = pd.DataFrame(
              ({'Avg min between sent tnx': [789], 'Avg min between received tnx':[500], 'Time Diff between first and last (Mins)': [100], 
                'Sent tnx': [100], 'Received Tnx': [500], 'Number of Created Contracts': [2],  
                'avg val received': [756], 'avg val sent': [27], 'total Ether sent': [1000], 'total ether balance': [500],
                'ERC20 total Ether received': [0], 'ERC20 total ether sent': [5], 'ERC20 total Ether sent contract': [5],
                'ERC20 uniq sent addr': [2],'ERC20 uniq rec token name': [1]}))
test_data

# Testing data and result
# name - 1, time - 0.5 => 1
# name - 100, time - 0.5 => 1
# name - 100, time - 1 => 0
# name - 110, time - 1 => 0
# name - 55, time - 1 => 0
# name - 76, diff - 0.5, ERC20 ttl eth recd - 0 => 1
# name - 76, diff - 0.5, ERC20 ttl eth recd - 1 => 0
# name - 76, diff - 0.5, ERC20 ttl eth recd - 1 => 0
# name - 15, diff - 0, ERC20 ttl eth recd - 1 => 0
# name - 15, diff - 0, ERC20 ttl eth recd - 100000 => 0
# name - 1, diff - 0, ERC20 ttl eth recd - 0 => 1


# Transforming the test data
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson", standardize=False)
test_data_transformed = pt.fit_transform(test_data)
test_data_transformed = pd.DataFrame(test_data_transformed, columns=X_resampled.columns)
#test_data_transformed
y_pred = XGB_C.predict(test_data_transformed)
print(y_pred)

# ===== END =====