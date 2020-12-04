# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 
df = pd.read_csv(path)

X = df.drop(['customerID','Churn'],1)
y = df['Churn'].copy()
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Code starts here





# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
# X_train['TotalCharges'][X_train['TotalCharges'] == ' '] = np.NaN
# X_test['TotalCharges'][X_test['TotalCharges'] == ' '] = np.NaN

# X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)
# X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)

# X_mean = X_train['TotalCharges'].mean()
# X_train['TotalCharges'] = X_train['TotalCharges'].fillna(value = X_mean)

# X__test_mean = X_test['TotalCharges'].mean()
# X_test['TotalCharges'] = X_test['TotalCharges'].fillna(value = X__test_mean)

# print(X_mean)
# print(X__test_mean)

# X_train.isnull().sum()
# X_test.isnull().sum()

# X_train_cat = X_train.select_dtypes(include = object)
# X_test_cat = X_test.select_dtypes(include = object)

# le = LabelEncoder()
# X_train = X_train.apply(le.fit_transform)
# X_test = X_test.apply(le.fit_transform)

# y_train = y_train.replace(['No', 'Yes'],[0,1])
# y_test = y_test.replace(['No', 'Yes'],[0,1])

X_train['TotalCharges'][X_train['TotalCharges'] == ' '] = np.NaN
X_test['TotalCharges'][X_test['TotalCharges'] == ' '] = np.NaN
X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)
X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)
X_train.isna().sum()
X_test.isna().sum()
X_train['TotalCharges'][X_train['TotalCharges'].isna()] = X_train['TotalCharges'].mean()
X_test['TotalCharges'][X_test['TotalCharges'].isna()] = X_test['TotalCharges'].mean()
X_train.isna().sum()
X_test.isna().sum()
le = LabelEncoder()
#le.fit_transform(X_train.select_dtypes(include='object'))
 
X_train.dtypes
 
X_train.select_dtypes(include='object').columns
 
for i in X_train.select_dtypes(include='object').columns:
 X_train[i] = le.fit_transform(X_train[i])
 X_test[i] = le.transform(X_test[i])
 
#X = X.apply(LabelEncoder().fit_transform)
 
y_train = y_train.replace(['Yes','No'],[1,0])
y_test = y_test.replace(['Yes','No'],[1,0])


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
print(X_train)
print(X_test)
print(y_train)
print(y_test)

ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
ada_score = accuracy_score(y_test, y_pred)
ada_cm = confusion_matrix(y_test, y_pred)
ada_cr = classification_report(y_test,y_pred)
print(ada_score)
print(ada_cm)
print(ada_cr)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_test, y_pred)
xgb_cm = confusion_matrix(y_test, y_pred)
xgb_cr = classification_report(y_test, y_pred)

print(xgb_score)
print(xgb_cm)
print(xgb_cr)


clf_model = GridSearchCV(estimator=xgb_model, param_grid=parameters)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)
clf_score = accuracy_score(y_test, y_pred)
clf_cm = confusion_matrix(y_test, y_pred)
clf_cr = classification_report(y_test, y_pred)
print(clf_score)
print(clf_cm)
print(clf_cr)



