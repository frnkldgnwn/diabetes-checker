from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# % matplotlib inline

diabetesDF = pd.read_csv('diabetes.csv')
print(diabetesDF.head())

corr = diabetesDF.corr()
print(corr)
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)

dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]

trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome', 1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome', 1))

print(pd.DataFrame(trainLabel))
print(pd.DataFrame(trainData))
print(pd.DataFrame(testLabel))
print(pd.DataFrame(testData))

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds
testData = (testData - means)/stds
# check that new means equal 0
np.mean(trainData, axis=0)
# check that new stds equal 1
np.std(trainData, axis=0)

diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)

accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")
