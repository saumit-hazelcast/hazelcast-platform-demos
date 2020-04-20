import datetime
import sklearn
import pandas as pd
import json

def auc(true_labels, predicted_probabilities, positive_label):
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_labels, predicted_probabilities, pos_label = positive_label)
	auc = sklearn.metrics.auc(fpr, tpr)
	return auc

def read_csv(filename, has_header = True, sep = ','):
	df = pd.read_csv(filename, header = 0 if has_header else None)
	df.columns = [col.strip() for col in df.columns]
	return df

def write_coll(collection, filename):
	with open(filename, 'w') as f:
		f.write(json.dumps(collection, indent = 4))

def read_coll(filename):
	with open(filename, 'r') as f:
		return json.load(f)

def get_timestamp():
	return datetime.datetime.now().strftime('%Y%m%d_%H%M')

