import numpy as np
import pandas as pd

from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask import jsonify

from math import sqrt
from math import pi
from math import exp

app = Flask(__name__)

@app.route('/api/diabetes', methods = ['GET'])
def diabetes():
	pregnancies = request.args.get("pregnancies")
	glucose = request.args.get("glucose")
	bloodpressure = request.args.get("bloodpressure")
	skinthickness = request.args.get("skinthickness")
	insulin = request.args.get("insulin")
	bmi = request.args.get("bmi")
	age = request.args.get("age")
	
	res_str = RUN([pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, age])
	return jsonify(result=res_str)

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

def summarize_dataset(dataset):
	num_per_feature = [[row[i] for index, row in dataset.iterrows()] for i in range(len(dataset.columns))]
	
	mean_per_feature = [[mean(np.asarray([float(str(num).replace(",", ".")) for num in elem]))] for elem in num_per_feature]
	mean_per_feature = pd.DataFrame(mean_per_feature).transpose()
	mean_per_feature.columns = dataset.columns
	mean_per_feature = mean_per_feature.drop('Outcome', axis = 1)
	
	stdev_per_feature = [[stdev(np.asarray([float(str(num).replace(",", ".")) for num in elem]))] for elem in num_per_feature]
	stdev_per_feature = pd.DataFrame(stdev_per_feature).transpose()
	stdev_per_feature.columns = dataset.columns
	stdev_per_feature = stdev_per_feature.drop('Outcome', axis = 1)
	
	index = pd.Series(['Mean', 'Stdev'])
	summarize_res = pd.concat([mean_per_feature, stdev_per_feature], ignore_index = True).set_index(index)
	return summarize_res

def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

def RUN(data):
	diabetes_df = pd.read_csv('export_diabetes.csv', sep =';')
	print("Orignal:\n", diabetes_df, "\n")
	
	#step 1 separate by class
	df_negative = pd.DataFrame([row for index, row in diabetes_df.iterrows() if row['Outcome'] == 0]).reset_index().drop('index', axis = 1)
	df_positive = pd.DataFrame([row for index, row in diabetes_df.iterrows() if row['Outcome'] == 1]).reset_index().drop('index', axis = 1)
	print("DataFrame Negative:\n", df_negative, "\n")	
	print("DataFrame Positive:\n", df_positive, "\n")

	#step 2 summarize dataset
	dataset_summaries = summarize_dataset(diabetes_df)
	print("Summeries:\n", dataset_summaries, "\n")
	
	#step 3 summarize data by class
	negative_summaries = summarize_dataset(df_negative)
	positive_summaries = summarize_dataset(df_positive)
	print("Summeries Negative:\n", negative_summaries, "\n")
	print("Summeries Positive:\n", positive_summaries, "\n")
	
	#step 4 gaussian probability density function
	data = pd.DataFrame(data).transpose()
	data.columns = dataset_summaries.columns
	
	likelihoods_negative = [calculate_probability(float(str(data.iloc[0]['Pregnancies']).replace(",", ".")), float(str(negative_summaries.iloc[0]['Pregnancies']).replace(",", ".")), float(str(negative_summaries.iloc[1]['Pregnancies']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['Glucose']).replace(",", ".")), float(str(negative_summaries.iloc[0]['Glucose']).replace(",", ".")), float(str(negative_summaries.iloc[1]['Glucose']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['BloodPressure']).replace(",", ".")), float(str(negative_summaries.iloc[0]['BloodPressure']).replace(",", ".")), float(str(negative_summaries.iloc[1]['BloodPressure']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['SkinThickness']).replace(",", ".")), float(str(negative_summaries.iloc[0]['SkinThickness']).replace(",", ".")), float(str(negative_summaries.iloc[1]['SkinThickness']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['Insulin']).replace(",", ".")), float(str(negative_summaries.iloc[0]['Insulin']).replace(",", ".")), float(str(negative_summaries.iloc[1]['Insulin']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['BMI']).replace(",", ".")), float(str(negative_summaries.iloc[0]['BMI']).replace(",", ".")), float(str(negative_summaries.iloc[1]['BMI']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['Age']).replace(",", ".")), float(str(negative_summaries.iloc[0]['Age']).replace(",", ".")), float(str(negative_summaries.iloc[1]['Age']).replace(",", ".")))]
	likelihoods_negative = pd.DataFrame(likelihoods_negative).transpose()
	likelihoods_negative.columns = data.columns
	
	likelihoods_positive = [calculate_probability(float(str(data.iloc[0]['Pregnancies']).replace(",", ".")), float(str(positive_summaries.iloc[0]['Pregnancies']).replace(",", ".")), float(str(positive_summaries.iloc[1]['Pregnancies']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['Glucose']).replace(",", ".")), float(str(positive_summaries.iloc[0]['Glucose']).replace(",", ".")), float(str(positive_summaries.iloc[1]['Glucose']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['BloodPressure']).replace(",", ".")), float(str(positive_summaries.iloc[0]['BloodPressure']).replace(",", ".")), float(str(positive_summaries.iloc[1]['BloodPressure']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['SkinThickness']).replace(",", ".")), float(str(positive_summaries.iloc[0]['SkinThickness']).replace(",", ".")), float(str(positive_summaries.iloc[1]['SkinThickness']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['Insulin']).replace(",", ".")), float(str(positive_summaries.iloc[0]['Insulin']).replace(",", ".")), float(str(positive_summaries.iloc[1]['Insulin']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['BMI']).replace(",", ".")), float(str(positive_summaries.iloc[0]['BMI']).replace(",", ".")), float(str(positive_summaries.iloc[1]['BMI']).replace(",", "."))),
		calculate_probability(float(str(data.iloc[0]['Age']).replace(",", ".")), float(str(positive_summaries.iloc[0]['Age']).replace(",", ".")), float(str(positive_summaries.iloc[1]['Age']).replace(",", ".")))]
	likelihoods_positive = pd.DataFrame(likelihoods_positive).transpose()
	likelihoods_positive.columns = data.columns
	
	print("Likelihoods Negative:\n", likelihoods_negative, "\n")
	print("Likelihoods Positive:\n", likelihoods_positive, "\n")
	
	prior_negative = df_negative.shape[0] / diabetes_df.shape[0]
	prior_positive = df_positive.shape[0] / diabetes_df.shape[0]
	
	print("Prior Negative:\n", prior_negative, "\n")
	print("Prior Positive:\n", prior_positive, "\n")
	
	#step 5 class probabilities
	probability = [1, 1]
	
	for feature in likelihoods_negative.columns:
		probability[0] *= likelihoods_negative.iloc[0][feature]
	probability[0] *= prior_negative
	
	for feature in likelihoods_positive.columns:
		probability[1] *= likelihoods_positive.iloc[0][feature]
	probability[1] *= prior_positive
	
	print("Probability Negative: ", probability[0], "\n")
	print("Probability Positive: ", probability[1], "\n")
	
	if probability[0] > probability[1]:
		return "NORMAL"
	else:
		return 'DIABETES'

if __name__ == '__main__':
	app.run(port='5002')
