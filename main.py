from create_dataset import create_dataset, create_multi_shot_dataset, read_test_data
from plot_data import plot_results
from models import models, multi_models

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

x_data, y_data = create_dataset()

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.15)
x_test = read_test_data()

'''
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
'''
# Linear regression
linear_regression_model = models('Linear', LinearRegression(), x_train y_train, x_val, y_val)

# Ridge model with regularization 0.1
ridge_regression_model = models('Ridge', Ridge(alpha=0.1), x_train, y_train, x_val, y_val)

# Lasso model with regularization 0.1
lasso_regression_model = models('Lasso', Lasso(alpha=0.1), x_train, y_train, x_val, y_val)

# Elastic net with regularization 1 and ratio = 0.5
elasticnet_regression_model = models('Elastic net', ElasticNet(alpha=1, l1_ratio=0.5), x_train, y_train, x_val, y_val)

# XGB regression model with depth=4, 250=estimators, learning rate=0.05
xgb_regression_model = models('XGB', xgb.XGBRegressor(max_depth = 4, n_estimators = 250, learning_rate = 0.05), x_train, y_train, x_val, y_val)


linear_results = linear_regression_model.evaluate_model()
ridge_results = ridge_regression_model.evaluate_model()
lasso_results = lasso_regression_model.evaluate_model()
elasticnet_results = elasticnet_regression_model.evaluate_model()
xgb_results = xgb_regression_model.evaluate_model()

plot_results([linear_results, ridge_results, lasso_results, elasticnet_results, xgb_results])

'''
first_train_x, first_train_y, second_train_x, second_train_y, first_val_x, first_val_y, second_val_x, second_val_y = create_multi_shot_dataset()

xgb_regression_model_multi = multi_models('XGB regression model', xgb.XGBRegressor(max_depth = 4, n_estimators = 250, learning_rate = 0.05),xgb.XGBRegressor(max_depth = 4, n_estimators = 250, learning_rate = 0.05), first_train_x, \
                                    first_train_y, second_train_x, second_train_y, first_val_x, first_val_y, second_val_x, second_val_y)

xgb_regression_model_multi.evaluate_model()'''