#%% 
# Import necessary libraries
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import StackingRegressor



data = pd.read_csv('/home/star_planet/swatas/csv_data/data.csv')

features = data.iloc[:, :4999]
labels = data.iloc[:, 4999:]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)


encoding_dim = 1000  


input_layer = Input(shape=(4999,))


encoded = Dense(encoding_dim, activation='relu')(input_layer)

decoded = Dense(4999, activation='sigmoid')(encoded)


autoencoder = Model(input_layer, decoded)

encoder = Model(input_layer, encoded)


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))


X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)


base_models = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=0)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, random_state=0)),
    ('knn', KNeighborsRegressor(n_neighbors=3))
]

meta_model = Ridge()


model = MultiOutputRegressor(StackingRegressor(estimators=base_models, final_estimator=meta_model))


model.fit(X_train_encoded, y_train)


y_pred = model.predict(X_test_encoded)


r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
mse_scores = mean_squared_error(y_test, y_pred, multioutput='raw_values')


r2_scores_list = []
for i, label in enumerate(y.columns):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    r2_scores_list.append(r2)
    print(f'R^2 score for {label}: {r2}')

print("Model R2 Score:", r2_score(y_test, y_pred, multioutput='variance_weighted'))


plt.figure(figsize=(10, 6))
plt.bar(y.columns, r2_scores_list)
plt.xlabel('Labels')
plt.ylabel('R^2 Score')
plt.title('R^2 Scores for Individual Labels')
plt.xticks(rotation=45)
plt.show()