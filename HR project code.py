
import pyodbc
import pandas as pd
import pandas_profiling


conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-A0U077F\SQLEXPRESS;'
                      'Database=hr data;'
                      'Trusted_Connection=yes;')

df = pd.read_sql_query('SELECT * FROM [dbo].[Sheet1$]', conn)
pd.set_option('display.max_columns',None)
print(df)
print(type(df))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from scipy import stats
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer

#To view first five rows
df.head()

#To view last five rows
df.tail()

#Shape of dataset
df.shape                  #28 features

#To view the categorical and numerical columns and its datatypes in the dataset
df.info()                                       # 7 numerical and 21 categorical features

#To see the column name 
df.columns

#To find the statistical property of data
df.describe() 
# describes statistics and shape of the data distribution 
#**Handling Missing values**
 #Using Simple Imputer for Mode 


# Importing the SimpleImputer class
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')

mode_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

"""1)Age"""

df["Age"] = pd.DataFrame(imputer.fit_transform(df[["Age"]]))#mean
df["Age"].isnull().sum()

#project compleation
df["ProjectCompletion"] = pd.DataFrame(imputer.fit_transform(df[["ProjectCompletion"]]))#mean
df["ProjectCompletion"].isnull().sum()

"""2)Marital Status"""

df["Marital_status"] = pd.DataFrame(mode_imputer.fit_transform(df[["Marital_status"]]))#mean imputer
df["Marital_status"].isnull().sum()

"""3)Job Involvement deleted that """


"""4)Annual Income"""

df["Annual_income"] = pd.DataFrame(imputer.fit_transform(df[["Annual_income"]])) #mean Imputer
df["Annual_income"].isnull().sum()

df.isnull().sum()

#Data Preprocessing
# Data Cleaning
#Dropping the columns not necessary for analysis
df=df.drop('Employee_ID',axis=1)

# checking any null values
df.isnull().sum()
#No missing values found

# Visualisation 'Graphical Representation'

#To check imbalance 
df['Performancerating'].value_counts() 
#3 good, average, poor - 3 classes in taget feature

plt.pie(df.Performancerating.value_counts(), labels = ['good', 'average', 'poor'], autopct='%1.f%%', pctdistance=0.5)
plt.title('Performance rating of employees')  # imbalanced dataset

sns.countplot(df['Performancerating'])
# given data is balanced dataset.

#Visualizing the Missing values
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df.isna().sum()                                        #no missing values

#boxplot - checking presence of outliers
f, ax = plt.subplots(2,4, figsize = (15,7))
plt.suptitle('Boxplot of numerical features')
sns.boxplot(x = df.Age, ax = ax[0][0])
sns.boxplot(x = df.Job_level, ax = ax[0][1])
sns.boxplot(x = df.Annual_income, ax = ax[0][2])
sns.boxplot(x = df.Experience, ax = ax[0][3])
sns.boxplot(x = df.Trainingtime, ax = ax[1][0])
sns.boxplot(x = df.ProjectCompletion, ax = ax[1][1])
sns.boxplot(x = df.PercentSalaryHike, ax = ax[1][2])
f.delaxes(ax[1,3]) 

# age and experience feature have outliers 
#creating winsorization techniques to handle outliers
from feature_engine.outliers import Winsorizer
iqr_winsor = Winsorizer(capping_method='iqr', tail='both',fold=1)

#handling outliers
df[['Age']] = iqr_winsor.fit_transform(df[['Age']])
df[['Experience']] = iqr_winsor.fit_transform(df[['Experience']])

# age & experience features have been winsorized using 'iqr' method.

# boxplot for checking outliers after winsorization
sns.boxplot(df.Age)

sns.boxplot(df.Experience)

#checking duplicates 
duplicates=df.duplicated()
duplicates
sum(duplicates)

#  objects(categoricals) into numeric - Label Encoding

from sklearn.preprocessing import LabelEncoder
categories = ['Gender', 'Marital_status', 'Education',
       'EnvironmentSatisfaction', 'JobInvolvement',
       'Job_satisfaction', 'RelationshipSatisfaction',
       'Working_hrs_per_Day', 'WorkLifeBalance',
       'BehaviouralCompetence', 'On_timeDelivery', 'TicketSolvingManagements',
       'WorkingFromHome', 'Psycho_social_indicators',
       'Over_time', 'Attendance', 'NetConnectivity',
       'Department', 'Position', 'Performancerating']

# Encode Categorical Columns
le = LabelEncoder()
df[categories] = df[categories].apply(le.fit_transform)
df 
#  EDA (Exploratory Data Analysis)
# --------------------------------
 
# Descreptive Analytics
# # Measure of central tendancy - 1st moment business decision 

df.mean()

# Observation: 1) average age of employee is 43.81
        #      2) average salary of employee is 15.18 in lakhs
        #      3) average experience of employee is 14.08 yrs
        #      4) average training time is 3.5 months
        #      5) average no.of projects completed is 10
        #      6) average salary hike of employee in percentages is 8.5
df.median()
# Observation:  1. average age of employee is 43
        #       2. average annual income of employee is 15 lakhs
        #       3. average work experience of employee is 16
        #       4. average training time is 4 months
        #       5. average no.of. projects completed is 12
        #       6. avereage percent salary hike is 8
        #       7. average job level is 3

# Observation: Mean and median values are not same due to outliers in the dataset.

#Finding mode for categorical data
df[["Gender","Marital_status","EnvironmentSatisfaction","Over_time"]].mode()

# most occuring value..
# gender - female, marital status - single, environment satisfaction- medium, overtime-no

# # measure of dispersion
df.var() # 2nd moment business decision -var(), std()
# variance - The variance measures the average degree to which each point differs from the mean.
# Variance Observation : age - 105.17
             
             #  annual income - 36.95
             # work experience - 41.9
             # training time  - 2.86
             # project evaluation - 11.25
             # percent salary hike - 20.79

df.std() # standard deviation - Standard deviation is the spread of a group of numbers from the mean.

# Standard Deviation Observation: 
     #  age  10.25
     # anual income - 6
     # work experience - 6.48
     # training time - 1.69
     # project evaluation -  3.35
     # percent salary hike 4.5
     
  # Note: While standard deviation is the square root of the variance, variance is the average of all data points within a group.
  
range = max(df.Age)-min(df.Age)
range #48

# # measure of skewess and kurtosis - 3rd & 4th business moment decisions

df.skew() #3rd moment business decision - skewness - a long tail
# Skewness refers to a distortion or asymmetry that deviates from the symmetrical bell curve, or normal distribution, in a set of data. If the curve is shifted to the left or to the right, it is said to be skewed

# Observations:
    # age 0.17 positively or right skewed
    # joblevel   0.025 - positively or right skewed
    # annual income -0.05 - positively or right skewed
    # work experience -0.58 negatively or left skewed
    # training time  -0.0040  - negatively or left skewed
    # project evaluation   -0.37 - negatively or left skewed
    # percent salary hike  0.01  - positively or right skewed

df.kurt() # 4th moment business decision- measure of tailedness of probability distribution
# Kurtosis is defined as the standardized fourth central moment of a distribution minus 3 (to make the kurtosis of the normal distribution equal to zero).
# standard normal distribution has kurtosis of 3 (Mesokurtic), 
# kurtosis >3 is  - leptokurtic, <3 is platykurtic

   # Observations:
       # age  is    -0.66- platykurtic
       # joblevel    -1.28  - leptokurtic
       # anual income  -1.01- platykurtic
       # work experience  -1.55 platykurtic
       # training time  -1.24 - leptokurtic
       # project evaluation   -1.25 - platykurtic
       # percent salary hike  -1.18 - platykurtic

#Fifth moment business decision - Graphical representation

# Univariate analysis
# --------------------------------------------------

# histogram
df.hist() # overall distribution of data

sns.histplot(data=df, x='Age', kde=True)
# observation: 
    # age is slightly skewed. (right skewed) 

sns.histplot(data=df, x='Annual_income', kde=True)
# observation:
    # salary is uniformly distributed.
        
sns.histplot(data=df, x='Experience', kde=True)
# observations:
    # experience feature is  left skewed.

sns.histplot(data=df, x='Trainingtime', kde=True)
# observations:
    # training time - data is uniformly distributed.

sns.countplot(x='Performancerating', hue="WorkingFromHome", data=df)
# observation:
    # both work form home & working from office employees performance is similar.

sns.countplot(y='Department', hue="Performancerating", data=df)

# observation:
    # employees in Finance & HR departments performing similiar in performance.

sns.countplot(y='Position', hue="Performancerating", data=df)
# observation:
    # budget analyst & hr positions performing good, managers are performing less than other positions.

sns.displot(df['Annual_income'],kde=True,color='blue')

# NO of projects completed when male and female employees work from home/office
sns.barplot(x = df["Experience"], y = df["Performancerating"], data= df)
plt.show()
#Performance rating increases with experience

# NO of projects completed when male and female employees work from home/office
sns.barplot(x = df["Performancerating"], y = df["ProjectCompletion"], hue = "WorkingFromHome", data= df).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()
#performance rating is good for employees completing more number of projects and it is similar for work from home and office

# NO of projects completed when male and female employees work from home/office
sns.barplot(x = df["Psycho_social_indicators"], y = df["ProjectCompletion"], hue = "WorkingFromHome", data= df).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()

#satisfactory employee working from office completes more projects

# NO of projects completed when male and female employees work from home/office
sns.barplot(x = df["Gender"], y = df["ProjectCompletion"], hue = "WorkingFromHome", data= df).legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()

#project completion is same for employee work from home or office, female employees slightly complete more projects than male


# In[122]:


#Correlation 
corrMatrix = df.corr()
corrMatrix

# Correlation between different variables
corr = df.corr()
# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(30, 10))
# Generate a mask for upper traingle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)
#Attendance and working hrs per day have strong correlation
#netconnectivity and workfromhome has strong correlation
#performance rating has positive and negative correlation with the features

#Automatic EDA : Pandas profiling
from pandas_profiling import ProfileReport
profile = ProfileReport(df)
profile
# Random forest classifier
#Import train_test_split function and randomforest classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#separating dependent and independent columns 
X = df.iloc[:,0:26]  #independent columns
y = df.iloc[:,-1]    #target column
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=30) # 70% training and 30% test
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100,max_depth=3)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train,clf.predict(X_train) ))
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# SVM support vector machine
#separating dependent and independent columns 
X = df.iloc[:,0:26]  #independent columns
y = df.iloc[:,-1]    #target column
# dividing X, y into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
 
# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'rbf', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
 
# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test)
print(accuracy)
# creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, svm_predictions) 
cm

#DEsicion tree

#Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score,confusion_matrix

#separating dependent and independent columns 
X = df.iloc[:,0:26]  #independent columns
y = df.iloc[:,-1]    #target column

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 0:26]))
df[df.columns[0:26]] = df_scaled[df_scaled.columns[0:26]]


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30) # 70% training and 30% test

     
# Model building for Decision Tree

clf = DecisionTreeClassifier(criterion="gini", max_depth=3)

clf.fit(X_train,y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}') # 98.62
print(f'Test score {accuracy_score(y_test_pred,y_test)}')  #  98.36

# confusion matrix for performance metrics
cm = confusion_matrix(y_test, y_test_pred)
cm
# or
pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predictions'])

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))

# =============================================================
# KNN
#separating dependent and independent columns 
X = df.iloc[:,0:26]  #independent columns
y = df.iloc[:,-1]    #target column
# dividing X, y into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
 
# accuracy on X_test
accuracy = knn.score(X_test, y_test)
print(accuracy)
 
# creating a confusion matrix
knn_predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, knn_predictions)
cm
import pickle

# open the pickle file in writebyte mode
file = open("model.pkl",'wb')
#dump information to that file
pickle.dump(clf, file)
file.close()

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[27, 1,1,0,2,1,2,6,6,3,0,2, 6,1,2,0,1,2,1,0,1,0,8,1, 3,2]]))




