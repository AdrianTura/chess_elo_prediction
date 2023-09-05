from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import numpy as np

class models:
    def __init__(self, name, model, x_train, y_train, x_val, y_val):
        self.name = name
        self.model = model

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        

        self.no_features = len(x_train[0])
        self.train_size = len(x_train)
        self.val_size = len(y_train)

        self.train()

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_val)

        mae = mean_absolute_error(self.y_val, y_pred)
        mse = mean_squared_error(self.y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_val, y_pred)
        adjusted_r2 = 1 - ((1-r2) * (self.val_size-1)/(self.val_size - self.no_features -1))

        print('Evaluating the {} model:'.format(self.name))
        print('Mean absolute error:', mae)
        print('Mean squared error:', mse)
        print('Root mean squared error:', rmse)
        print('r2 score:', r2)
        print('adjusted r2 score:', adjusted_r2)
        print('\n')
        return self.name,mae,mse,rmse,r2,adjusted_r2
    
    def predict(self, x_features):
        y_pred = self.model.predict([x_features, self.x_val[0]])

        return y_pred

class multi_models:
    def __init__(self, name, model, model2,first_train_x, first_train_y, second_train_x, second_train_y, first_val_x, first_val_y, second_val_x, second_val_y):
        self.name = name

        self.first_model = model
        self.second_model = model2

        self.first_x_train = first_train_x
        self.first_y_train = first_train_y
        self.second_x_train = second_train_x
        self.second_y_train = second_train_y
        
        
        self.first_x_val = first_val_x
        self.first_y_val = first_val_y
        self.second_x_val = second_val_x
        self.second_y_val = second_val_y

        self.train()

    def train(self):
        self.first_model.fit(self.first_x_train, self.first_y_train)
        self.second_model.fit(self.second_x_train, self.second_y_train)

    def evaluate_model(self):
        first_y_pred = self.first_model.predict(self.first_x_val)
        
        second_shot_x = self.second_x_val
        for i in range(0, len(first_y_pred)):
            second_shot_x[i][-1] = first_y_pred[i]

        final_pred = self.second_model.predict(second_shot_x)

        mae = mean_absolute_error(self.second_y_val, final_pred)
        mse = mean_squared_error(self.second_y_val, final_pred)
        rmse = np.sqrt(mse)

        print('Evaluating the {} model:'.format(self.name))
        print('Mean absolute error:', mae)
        print('Mean squared error:', mse)
        print('Root mean squared error:', rmse)
        print('\n')
        return mae,mse,rmse