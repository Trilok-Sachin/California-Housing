import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('housing.csv')

data1 = data[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]

x = np.array(data1)
y = np.array(data['median_house_value'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

data["income_cat"] = pd.cut(data["median_income"], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["income_cat"]):
	strat_train_set = data.loc[train_index]
	strat_test_set = data.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_cat", axis=1, inplace=True)

data = strat_train_set.copy()

data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
	s=data["population"]/100, label="population", figsize=(10,7),
	c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
	)
plt.legend()
plt.show()

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(data[attributes], figsize=(12, 8))
plt.show()

data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

data["rooms_per_household"] = data["total_rooms"]/data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["population_per_household"]=data["population"]/data["households"]

corr_matrix = data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

data = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#Data Cleaning
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = data.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

X = imputer.transform(housing_num)

#OneHot Representation
from sklearn.preprocessing import OrdinalEncoder

housing_cat = data[["ocean_proximity"]]

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housin_cat_1hot = cat_encoder.fit_transform(housing_cat)
housin_cat_1hot.toarray()

#Class For Custom Transformation
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
 	
 	def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
 		self.add_bedrooms_per_room = add_bedrooms_per_room
 	
 	def fit(self, X, y=None):
 		return self # nothing else to do
 	
 	def transform(self, X, y=None):
 		rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
 		population_per_household = X[:, population_ix] / X[:, households_ix]
 		if self.add_bedrooms_per_room:
 			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
 			return np.c_[X, rooms_per_household, population_per_household,
 						bedrooms_per_room]
 		else:
 			return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(data.values)

#Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
	           ('attribs_adder', CombinedAttributesAdder()),
	           ('std_scaler', StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
	            ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
                ])

housing_prepared = full_pipeline.fit_transform(data)

#Training the Model

#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = data.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression Error: ',lin_rmse)

#Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print('Decision Tree Error: ',tree_rmse)

#Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
	     scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)	

#Function to display scores
def display_scores(scores,algorithm):
	print('Algorithm used: ',algorithm)
	print("Scores:", scores)
	print("Mean:", scores.mean())
	print("Standard deviation:",scores.std())

display_scores(tree_rmse_scores, 'Decision Tree')

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
	     scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)	

display_scores(lin_rmse_scores, 'Linear Regressor')

#Random Forests
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print('Forest Error: ',forest_rmse)


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
	     scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)	

display_scores(forest_rmse_scores, 'Random Forest')

#Grid Search
from sklearn.model_selection import GridSearchCV

param_grid_forest = [
			{'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]},
			{'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]}]

forest_reg = RandomForestRegressor()
grid_search_forest = GridSearchCV(forest_reg, param_grid_forest, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search_forest.fit(housing_prepared, housing_labels)

print('Best Forest Params: ',grid_search_forest.best_params_)
print('Best Forest Estimator: ',grid_search_forest.best_estimator_)

cvres_forests_grid = grid_search_forest.cv_results_
for mean_score, params in zip(cvres_forests_grid["mean_test_score"], cvres_forests_grid["params"]):
	print('Grid Search Scores for Random Forests: ',np.sqrt(-mean_score), params)

#RandomSearch CV 
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

params_distribs_forest = {
			'n_estimators': randint(low=1, high=200),
			'max_features': randint(low =1, high = 8),
		}
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg,	param_distributions=params_distribs_forest,
		     n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

rnd_search.fit(housing_prepared, housing_labels)

cvres_forests_rand = rnd_search.cv_results_
for mean_score, params in zip(cvres_forests_rand['mean_test_score'], cvres_forests_rand['params']):
	print(np.sqrt(-mean_score), params)


#Support Vector Machines 
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(housing_prepared, housing_labels)

svr_scores = cross_val_score(svm_reg, housing_prepared, housing_labels,
	     scoring="neg_mean_squared_error", cv=10)
svr_rmse_scores = np.sqrt(-svr_scores)	

display_scores(svr_rmse_scores, 'Support Vector Regressor')

#Generalized Support Vector Regressor
#Grid Search CV
from sklearn.svm import SVR
param_grid_svr = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svm_reg = SVR()
grid_search_svr = GridSearchCV(svm_reg, param_grid_svr, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search_svr.fit(housing_prepared, housing_labels)

print('Best SVR parameters: ', grid_search_svr.best_params_)
print('Best SVR Estimator: ', grid_search_svr.best_estimator_)

cvres_svr_grid = grid_search_svr.cv_results_
for mean_score, params in zip(cvres_svr_grid["mean_test_score"], cvres_svr_grid["params"]):
	print('Grid Search Scores for SVR: ',np.sqrt(-mean_score), params)

#Random Search CV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import  expon, reciprocal

params_distribs_svr = {
					'kernel':['linear', 'rbf'],
					'C':reciprocal(20, 200000),
					'gamma': expon(scale=1.0),
					}
svm_reg_random = SVR()
rnd_search_svr = RandomizedSearchCV(svm_reg_random, param_distributions=params_distribs_svr,
		     n_iter=50, cv=5, scoring='neg_mean_squared_error',
		     verbose=2, random_state=42)
rnd_search_svr.fit(housing_prepared, housing_labels)

negative_mse_svr = rnd_search_svr.best_score_
rmse_svr_rand = np.sqrt(-negative_mse_svr)
print('SVR rmse randomCV error', rmse_svr_rand)
print('Random Search SVR Best Parameters', rnd_search.best_params_)

#Analyze the best Models
feature_importances = grid_search_forest.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
print('Feature Importance: ',feature_importances)

final_model = grid_search_forest.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse_forest = mean_squared_error(y_test, final_predictions)
final_rmse_forest = np.sqrt(final_mse_forest)

print('Final Forest Scores: ',final_rmse_forest)