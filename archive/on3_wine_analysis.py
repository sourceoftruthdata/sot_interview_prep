#########################
#
# Author: Daniel Covarrubias
# Date: 2018-09-02
# Version: 1.0
# Description: A brief Data Science template
# Revision History: 2018-09-02 Program creation
# Purpose: this code is meant to be compiled using Spyder for analysis of wine data          
#########################

######################
# Read in data       #
######################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from urllib.request import urlretrieve
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


link = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
urlretrieve(link, 'winequality-red.csv')
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())

#########################
# ETL                   #       
#########################

# show column names
list(df)
# examine top 10 rows
df.head()
# number or rows and columns
df.shape
# simple summary table
df.pivot_table(values=["fixed acidity"], index=["quality"], aggfunc=np.mean)


# extract inputs
df_inputs = df.iloc[:,0:11]
# extract target/output
df_target = df['quality']
# summarize categories
df_target = df_target.astype('category')
# count the number of categories in quality
pd.crosstab(df["quality"],df["pH"],margins=True)

# remove spaces in column names

df.rename(columns={'fixed acidity': 'fixed_acidity','volatile acidity':'volatile_acidity','citric acid':'citric_acid','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)

#########################
# EDA                   #       
#########################


df['quality'].hist(bins=12)
plt.title("Quality Distribution")
plt.ylabel("Number")
plt.show()

# create histograms
df.hist(bins=10)
plt.show()

pd.plotting.scatter_matrix(df, alpha=0.2)
plt.tight_layout()
# plt.savefig('scatter_matrix.png')

plt.scatter(df['density'], df['residual_sugar'])
plt.title("Density v. Residual Sugar")
plt.xlabel("Density")
plt.ylabel("Residual Sugar")
plt.show()

#########################
# summarize             #       
#########################

# extract summary stats #

list(np.transpose(df.describe()))

np.transpose(df.describe())

#########################
# correlation           #       
#########################

np.corrcoef(df).round(3)

wine_corr = df.corr()

abs(wine_corr) > .5

# whats correlated with quality
wine_corr['quality']

#########################
# heatmap               #       
#########################

# generate heatmap

sns.heatmap(wine_corr, xticklabels=wine_corr.columns, yticklabels=wine_corr.columns)



#########################
# Classifier 1          #       
#########################

train, test = train_test_split(df, test_size=0.3, random_state=3)

result = smf.ols('quality ~ volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH  + alcohol + sulphates ', data=df).fit()

print(result.summary())

#########################

y = train["quality"]
cols = ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","alcohol","sulphates"]

x = train[cols]

regr = linear_model.LinearRegression()
regr.fit(x,y)

ytrain_pred = regr.predict(x)
print("Training MSE: %.3f" % mean_squared_error(y, ytrain_pred))

errors = ytrain_pred.round() - y
errors.hist(bins=10)
plt.xlabel("Predicted - Actual")
plt.show()

#########################
# now test on Test data #
#########################

y = test["quality"]
cols = ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","alcohol","sulphates"]

x = test[cols]

ytest_pred = regr.predict(x)
print("Test MSE: %.3f" % mean_squared_error(y, ytest_pred))


# object is to decrease MSE across models #

# to measure performance of this model need ROC, classification matrix #
# lets plot observed -  expected for this exercise #

plt.scatter(y, ytest_pred)
plt.title("Observerd v. Predicted")
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.show()

plt.scatter(y, ytest_pred.round() - y, alpha=0.5)
plt.title("Observerd v. Expected")
plt.xlabel("Observed")
plt.ylabel("Rounding Difference")
plt.show()

errors = ytest_pred.round() - y
errors.hist(bins=10)
plt.show()

#########################
# princomp              #       
#########################

# Separating out the features
x = df_inputs.values
# Separating out the target
y = df_target.values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
prinComp = pca.fit_transform(x)
prinDf = pd.DataFrame(data = prinComp , columns = ['principal component 1', 'principal component 2'])

prinDfFinal = pd.concat([prinDf, df[['quality']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [3,4,5,6,7,8]
colors = ['r', 'g', 'b','y','o']
for targets, color in zip(targets,colors):
    indicesToKeep = prinDfFinal['quality'] == targets
    ax.scatter(prinDfFinal.loc[indicesToKeep, 'principal component 1']
               , prinDfFinal.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    
pca.explained_variance_ratio_

# I know there is formatting issue with the code, but it works here
# will address in future
# I believe it is number -> character in the labels/legend

##########################################
# would like to compare to decision tree #  
# perhaps in the future                  #   
##########################################

