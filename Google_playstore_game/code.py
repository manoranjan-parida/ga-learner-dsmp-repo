# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

data = pd.read_csv(path)
plt.figure(figsize =(8,6))
data['Rating'].plot(kind='hist')
plt.show()
data = data[data["Rating"]<=5]
data['Rating'].plot(kind='hist')
# plt.show()
# data = data[data["Rating"]>5]
# data['Rating'].plot(kind='hist')





#Code starts here


#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())
#print(total_null)
#print(percent_null)
missing_data = pd.concat([total_null, percent_null], axis = 1, keys=['Total','Percent'])
print(missing_data)

data = data.dropna(how = 'any')

total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())

# print(total_null_1)
# print(percent_null_1)

missing_data_1 = pd.concat([total_null_1, percent_null_1], axis = 1, keys=['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------

#Code starts here




gg = sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)

gg.set_xticklabels(rotation=90)

gg.set_titles("Rating vs Category")


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data["Installs"].value_counts()
data["Installs"] = data["Installs"].str.replace(',','')
data["Installs"] = data["Installs"].str.replace('+','')
data["Installs"] = data["Installs"].astype(int)

data["Installs"].value_counts()
le = LabelEncoder()
data["Installs"] = le.fit_transform(data["Installs"])
sns.regplot(x="Installs", y="Rating", data=data)




#Code ends here



# --------------
#Code starts here
data["Price"].value_counts()

data["Price"] = data["Price"].str.replace("$",'').astype(float)
sns.regplot(x="Price", y="Rating", data=data)


#Code ends here


# --------------

#Code starts here
data["Genres"].unique()
gen_data = data["Genres"].str.split(";", n = 1, expand = True)

data["Genres"] = gen_data
gr_mean = data.groupby(["Genres"], as_index = False)["Rating"].mean()
gr_mean.describe()
gr_mean = gr_mean.sort_values("Rating")
#print("The lowest of average rating on genres (Dating) is: {}".format.gr_mean.head(0))
print(gr_mean.head(1))
print(gr_mean.tail(1))


#Code ends here


# --------------

#Code starts here
#print(data['Last Updated'])
#data.info()
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
print(max_date)
tmp = max_date - data["Last Updated"]
data['Last Updated Days'] = tmp.dt.days
sns.regplot(x="Last Updated Days", y="Rating", data=data)


#Code ends here


