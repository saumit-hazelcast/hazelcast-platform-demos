import pandas as pd
import sklearn
import os
import pickle
import datetime
import json
import yaml
import sys
import util

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


target_column = 'yearly-income'

def feature_selection(df, target_column):
	# Use data science to select the best features
	# here we simply return static ones
	return ['age', 'workclass', 'education', 'occupation', 'hours-per-week', 'native-country', 'mean_age']


def train_model_simple(df, inputs_folder, outputs_folder):
	if os.path.exists(outputs_folder):
		raise ValueError("Output path exists.")

	os.mkdir(outputs_folder)

	# Separate the target column
	df_train = df.drop(target_column, axis = 1)
	util.write_coll(list(df_train.columns), outputs_folder + '/input_column_names.json')
	labels = df[target_column]
	labels_list = list(labels.unique())
	labels_dict = {labels_list[0]: 0, labels_list[1]: 1}
	labels = labels.apply(lambda x: labels_dict[x])
	util.write_coll(labels_dict, outputs_folder + '/labels_dict.json')

	# An example transform. Maybe some other data is needed
	# to "enrich" the dataset. This could come from a database
	# table but we assume a CSV for simplicty. If it came
	# from a dataset for example it could be a complex query
	# occupation_stats = pd.read_sql(some uber complex query)
	occupation_stats = util.read_csv(inputs_folder + '/occupation_stats.csv')

	# join with the training data
	df = df_train.merge(occupation_stats, how = 'inner', on = 'occupation')

	print("After merging with occupation.")
	df.head()

	# Maybe a feature selection step is performed here. In ML
	# feature selection typically tries to prune columns that
	# show little evidence early on as adding value to the model
	# and worse could add noise to it. Many strategies exist
	# from the simplest (constant columns), to highly involved
	# one's that build ML models themselves.
	feature_selected_columns = feature_selection(df, target_column)

	# we store the list of selected features because we will need
	# this when we do testing (we will select the same features)
	util.write_coll(feature_selected_columns, outputs_folder + '/selected_features.txt')

	# subset the features based on feature selection
	df = df[feature_selected_columns]

	# Now let's do some 'feature engineering' i.e. the data science
	# process of adding new features to the data (usually based on the
	# data scientists intuition about the problem).
	#
	# In this case we will add some aggregate statistics as a column
	# in the training data. Since aggregate statistics are NOT available
	# during testing time (testing assumes 1 row at a time for example)
	# we must store these.
	inter_1 = df[['education']]
	inter_1[target_column] = labels
	inter_1 = pd.DataFrame(inter_1.groupby('education').mean()[target_column]).reset_index()
	inter_1.columns = ['education', '{}_mean'.format(target_column)]
	inter_1.to_csv(outputs_folder + '/education_mean.csv', index = False)
	df = df.merge(inter_1, on = 'education')

	# now we may read some control variables from a yaml file
	with open(inputs_folder + '/ai-config.yml', 'r') as stream:
	    config = yaml.load(stream)
	df['age'] = config['age_multiplier'] * df['age']

	hours_info = df['hours-per-week'].describe().to_dict()
	util.write_coll(hours_info, outputs_folder + '/hours_info.json')

	df['hours-per-week'] = df['hours-per-week'] - hours_info['min']
	df['hours-per-week'] = df['hours-per-week'] / (hours_info['max'] - hours_info['min'])

	# OK end of featurizations - now preparing data for algorithm
	util.write_coll(list(df.columns), outputs_folder + '/column_names.json')

	# create the encoder object (this is an example of an ML specific transform for categorical data)
	enc_obj = OneHotEncoder(sparse = False, handle_unknown = 'ignore')

	encoder = enc_obj.fit(df)
	x_train_encoded = encoder.transform(df)

	print('Training the model')
	# we are not going to tune any hyper-parameters
	rf = RandomForestClassifier(n_estimators = 30, n_jobs = -1)  # -1 uses all cores...
	model = rf.fit(x_train_encoded, labels)

	# get training accuracy
	training_predictions = model.predict_proba(x_train_encoded).transpose()[1] # get predicstions for label 1
	print("Auc on training data is ", util.auc(labels, training_predictions, model.classes_[1]))

	print("Dumping models as pickle file.")
	with open( outputs_folder + '/rf.model', 'wb') as f:
		pickle.dump(model, f)

	with open(outputs_folder + '/onehot.model', 'wb') as f:
		pickle.dump(encoder, f)

	print("Done. Models and other files output are in location: ", outputs_folder)


if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("Usage: <training file> <input folder> <output folder>")
		exit(1)

	filename = sys.argv[1]
	inputs_folder = sys.argv[2]
	outputs_folder = sys.argv[3]

	df = util.read_csv(filename)
	train_model_simple(df, inputs_folder, outputs_folder)

