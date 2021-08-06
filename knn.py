import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("PS_20174392719_1491204439457_log.csv")

#print(data.info())
print(data.head())

print(data.isFraud.value_counts())

print(data.isFraud.value_counts(normalize=True))

df_fraud = data[data.isFraud == 1]

df_nofraud = data[data.isFraud == 0].head(20000)

print(df_fraud.shape)

data_subset = pd.concat([df_fraud,df_nofraud],axis=0)

print(data_subset.shape)
print(data_subset.columns)

sns.barplot(x="type",y="amount",data=data_subset)
plt.show()

sns.barplot(x="type",y="amount",data=data_subset,hue='isFraud')
plt.show()

data_subset['type'] = data_subset['type'].astype('category')

data_subset = data_subset.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1)
print(data_subset.head())

type_dummy = pd.get_dummies(data_subset.type,prefix="type")
print(type_dummy.head())

data_subset = pd.concat([data_subset,type_dummy],axis=1)
print(data_subset.head())

x = data_subset.drop(['isFraud','type','step'],axis=1).values
y = data_subset.isFraud.values
x_train, x_test,y_test,y_train = train_test_split(x,y,test_size=0.3,random_state=5,stratify=y)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
knn.score(x_test,y_test)

from sklearn.model_selection import GridSearchCV
import numpy as np

qs = {'n_neighbors' :np.arange(1,25)}
knn_qs = KNeighborsClassifier()

knn_param_search = GridSearchCV(knn_qs,qs,cv=10)

knn_param_search.fit(x_train,y_train)

knn_param_search.best_params_

knn_param_search.best_score_

print(data_subset.head())






