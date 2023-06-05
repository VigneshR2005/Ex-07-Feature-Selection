# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file
# PROGRAM
```
NAME: R Vignesh
REGISTER NUMBER:212222230172
```
```
#importing library
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# data loading
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()

#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(data.isnull(),cbar=False)

#Data Cleaning and Data Drop Process
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

# Change to categoric column to numeric
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1

# instead of nan values
data['Embarked']=data['Embarked'].fillna('S')

# Change to categoric column to numeric
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2

#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)

data.head(11)

#heatmap for train dataset
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# Now, data is clean and read to a analyze
sns.heatmap(data.isnull(),cbar=False)

# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

#Age with survived
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

#Count the pessenger class
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...
data2.head(11)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
```
# OUPUT
DATASET:

![DS1](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/0cf92702-d26f-47e3-a026-e1bc0601aefd)

Null Values:

![DS2](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/d15b46ac-6c88-4597-8674-dda8b62597a9)

Describe:

![DS3](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/db0b8e21-ba73-4e9a-b046-f7cd36a2ad5e)

missing values:

![DS4](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/ed7863f6-4358-4a9a-9af2-326e22f02803)

Data after cleaning:

![DS5](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/77aa04e0-535b-408c-b96f-34b022ca75c2)

Median:

![DS6](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/63e4b3ce-8e8a-4062-a083-f19678fd66ac)

Data on Heatmap:

![DS7](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/907dc2d6-70a3-40d5-acb9-103ba84df06f)

Report of (people survived & Died):

![DS8](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/064fa622-b675-4e0d-8028-c801d9412403)

Cleaned Null values:

![DS9](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/ec09bf6b-af75-4270-86cd-13bf4ebf1764)

Report of Survived People's Age:

![DS10](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/4b232ce7-7862-4aea-b69c-197b3b3cdc7f)

Report of pessengers:

![DS11](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/86a26afd-00b2-4136-a5a2-3398e3cb78a1)

Report:

![DS12](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/44f48c51-b7ec-4e7c-b8d4-0844cdd93575)
![DS14](https://github.com/Praveenkumar2004-dev/Ex-07-Feature-Selection/assets/119559827/33e7863f-8f05-4025-af5c-cd405f309ee1)

# RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.




