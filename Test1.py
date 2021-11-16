# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:07:57 2021

@author: USER
"""
import warnings
import pandas as pd

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from pprint import pprint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC, accuracy_score as accuracy



from config import input_file, features_file
from config import train_file, test_file



d = pd.read_csv( "Data1.csv", header = None )

columns = list( d.columns )
columns.pop()
columns.append( 'target' )
d.columns = columns

y = d.target
d.drop( 'target', axis = 1, inplace = True )

d = d.stack()
d.index.rename([ 'id', 'time' ], inplace = True )
d = d.reset_index()
	

	
	# doesn't work too well
with warnings.catch_warnings():
	warnings.simplefilter( "ignore" )
	f = extract_features( d, column_id = "id", column_sort = "time" )

#c:\usr\anaconda\lib\site-packages\scipy\signal\spectral.py:1633: 
# UserWarning: nperseg = 256 is greater than input length  = 152, using nperseg = 152

# Feature Extraction: 20it [22:33, 67.67s/it]
	
impute( f )
assert f.isnull().sum().sum() == 0
	
f['y'] = y
f.to_csv( features_file, index = None )
    

print ("loading {}".format( features_file ))
features = pd.read_csv( features_file )

train_x = features.iloc[:validation_split_i].drop( 'y', axis = 1 )
test_x = features.iloc[validation_split_i:].drop( 'y', axis = 1 )

train_y = features.iloc[:validation_split_i].y
test_y = features.iloc[validation_split_i:].y

print ("selecting features...")
train_features_selected = select_features( train_x, train_y, fdr_level = fdr_level )
	
print ("selected {} features.".format( len( train_features_selected.columns )))

train = train_features_selected.copy()
train['y'] = train_y

test = test_x[ train_features_selected.columns ].copy()
test['y'] = test_y
	


print ("saving {}".format( train_file ))
train.to_csv( train_file, index = None )

print ("saving {}".format( test_file ))
test.to_csv( test_file, index = None )
    
    
train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

x_train = train.drop( 'y', axis = 1 ).values
y_train = train.y.values

x_test = test.drop( 'y', axis = 1 ).values
y_test = test.y.values

classifiers = [
	#LR( C = 10 ),
	#LR( C = 1 ),													
	#LR( C = 0.1 ),									
									
	make_pipeline( StandardScaler(), LR()),	
	#make_pipeline( StandardScaler(), LR( C = 10 )),
	#make_pipeline( StandardScaler(), LR( C = 30 )),

	make_pipeline( MinMaxScaler(), LR()),					
	#make_pipeline( MinMaxScaler(), LR( C = 10 )),	
	#make_pipeline( MinMaxScaler(), LR( C = 30 )),

	#LDA(),										
	RF( n_estimators = 100, min_samples_leaf = 5 )
]

for clf in classifiers:

	clf.fit( x_train, y_train )
	p = clf.predict_proba( x_test )[:,1]
	p_bin = clf.predict( x_test )

	auc = AUC( y_test, p )
	acc = accuracy( y_test, p_bin )
	print( "AUC: {:.2%}, accuracy: {:.2%} \n\n{}\n\n".format( auc, acc, clf ))