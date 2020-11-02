# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
data = pd.read_csv(path)
X = data.drop(['customer.id','paid.back.loan'], axis = 1)
y = data['paid.back.loan']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)




# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
fully_paid = y_train.value_counts()
fully_paid.plot(kind = 'bar')


# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
X_train['int.rate'] = X_train['int.rate'].str.replace('%', '').astype(float)
X_train['int.rate'] = X_train['int.rate']/100
X_test['int.rate'] = X_test['int.rate'].str.replace('%', '').astype(float)
X_test['int.rate'] = X_test['int.rate']/100
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_df = X_train.select_dtypes(include = numerics)
cat_df = X_train.select_dtypes(exclude = numerics)

# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
cols = list(num_df.columns.values)
fig,axes = plt.subplots(nrows = 9, ncols = 1)
plt.figure(figsize=(20,10))

for i in range (len(cols)):
    sns.boxplot(x=y_train, y=num_df[cols[i]], ax=axes[i])

plt.show()


# Code ends here


# --------------
# Code starts here
cols = list(cat_df.columns.values)
fig,axes = plt.subplots(nrows = 2, ncols = 2)
for i in range(0,2):
    for j in range(0,2):
        sns.countplot(x=X_train[cols[i*2+j]], hue=y_train, ax=axes[i,j])
plt.show()

# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code starts here
le = LabelEncoder()
for i in range(len(list(cat_df.columns.values))):
    X_train[list(cat_df.columns.values)[i]].fillna(value=np.NaN)
    X_test[list(cat_df.columns.values)[i]].fillna(value=np.NaN)
    X_train[list(cat_df.columns.values)[i]].replace(r'', np.NaN)
    X_train[list(cat_df.columns.values)[i]] = le.fit_transform(X_train[list(cat_df.columns.values)[i]])
    X_test[list(cat_df.columns.values)[i]] = le.transform(X_test[list(cat_df.columns.values)[i]])

y_train[y_train == 'Yes']= 1
y_train[y_train == 'No']= 0
y_train = y_train.astype('int')
y_test[y_test == 'Yes']= 1
y_test[y_test == 'No']= 0
y_test = y_test.astype('int')

model = DecisionTreeClassifier(random_state = 0)
model.fit(X_train,y_train)
acc = model.score(X_test,y_test)
print(acc)
# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
model_2 = DecisionTreeClassifier(random_state=0)
p_tree = GridSearchCV(estimator=model_2,param_grid=parameter_grid, cv= 5)
p_tree.fit(X_train, y_train)
acc_2 = p_tree.score(X_test,y_test)
print('Accuracy :', acc_2)
print(p_tree.best_params_)
print(p_tree.best_estimator_)

# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data = export_graphviz(decision_tree=p_tree.best_estimator_,out_file=None,feature_names=X.columns, filled = True, class_names= ['loan_paid_back_yes','loan_paid_back_no'])
graph_big = pydotplus.graph_from_dot_data(dot_data)

# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here



