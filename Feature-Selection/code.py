# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path
dataset = pd.read_csv(path)
# Code starts here


# read the dataset



# look at the first five columns
dataset.head()
dataset[dataset.isnull()].count()
dataset.drop(["Id"], axis = 1, inplace = True)

dataset.describe()


# Check if there's any column which is not useful and remove it like the column id


# check the statistical description



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns
size = len(cols) -1
y = dataset.iloc[:,:-1]
x = dataset.iloc[:,-1]

for i in range(0,size):
    sns.violinplot(x = y.iloc[:,i])
    plt.show()

#number of attributes (exclude target)


#x-axis has target attribute to distinguish between classes


#y-axis shows values of an attribute


#Plot violin for all attributes



# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
subset_train = dataset.iloc[:,:10]
data_corr = subset_train.corr()
sns.heatmap(data_corr, annot = True)
correlation = data_corr.unstack().sort_values(kind='quicksort')
print(correlation)
corr_var_list = correlation[((correlation>upper_threshold) | (correlation<lower_threshold)) & (correlation!=1)]
print(corr_var_list)

# Code ends here




# --------------
# #Import libraries 
# from sklearn import cross_validation
# from sklearn.preprocessing import StandardScaler

# # Identify the unnecessary columns and remove it 
# dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

# X = dataset.iloc[:,:-1]
# Y = dataset.iloc[:,-1]
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.2, random_state = 0)

# # Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
# scaler = StandardScaler()
# X_train_temp = scaler.fit_transform(X_train)
# X_train_temp = pd.DataFrame(X_train_temp)
# X_test_temp = scaler.transform(X_test)
# X_test_temp = pd.DataFrame(X_test_temp)
# X_train1 = pd.concat([X_train_temp,X_train.iloc[:,52:]], axis = 1)
# #X_train1 = np.concatenate((X_train_temp,X_train.iloc[:,52:]),axis=1)
# X_test1 = pd.concat([X_test_temp, X_test.iloc[:,52:]], axis = 1)

# scaled_features_train_df = pd.DataFrame(X_train1, index = X_train.index, columns = X_train.columns)
# scaled_features_test_df = pd.DataFrame(X_test1, index = X_test.index, columns = X_test.columns)
# #Standardized
# #Apply transform only for continuous data


# #Concatenate scaled continuous data and categorical
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

r,c = dataset.shape
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)


#Standardized

scaler = StandardScaler()

#Apply transform only for continuous data
X_train_temp = scaler.fit_transform(X_train.iloc[:,:10])
X_test_temp = scaler.transform(X_test.iloc[:,:10])

#Concatenate scaled continuous data and categorical
X_train1 = numpy.concatenate((X_train_temp,X_train.iloc[:,10:c-1]),axis=1)
X_test1 = numpy.concatenate((X_test_temp,X_test.iloc[:,10:c-1]),axis=1)

scaled_features_train_df = pd.DataFrame(X_train1, index=X_train.index, columns=X_train.columns)
scaled_features_test_df = pd.DataFrame(X_test1, index=X_test.index, columns=X_test.columns)


# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb = SelectPercentile(score_func=f_classif, percentile=90)
predictors = skb.fit_transform(X_train1,Y_train)
scores = skb.scores_

Features = X_train.columns
dataframe = pd.DataFrame(Features, scores)
temp_dict={'Features':Features,'scores':scores}
dataframe = pd.DataFrame(temp_dict,index=None)
dataframe.sort_values('scores',ascending = False,inplace = True)
# # top_k_predictors = dataframe['Features'][dataframe['scores'] > 90].to_list()
# top_k_predictors = dataframe['scores'].to_list()
top_k_predictors = list(dataframe[:predictors.shape[1]].Features)
print(top_k_predictors)


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf1 = LogisticRegression()
clf = OneVsRestClassifier(clf1)
model_fit_all_features = clf.fit(X_train, Y_train)
predictions_all_features = clf.predict(X_test)
score_all_features = accuracy_score(Y_test,predictions_all_features)
print(score_all_features)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)
predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(Y_test, predictions_top_features)
print(score_top_features)


