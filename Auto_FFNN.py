#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor
from tensorflow.keras.layers import Dense
import os
import torch    


data = pd.read_csv('/home/star_planet/swatas/csv_data/data.csv')

features = data.iloc[:, :4999]
labels = data.iloc[:, 4999:]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

autoencoder = Sequential()
autoencoder.add(Dense(256, activation='relu', input_shape=(4999,)))
autoencoder.add(Dense(128, activation='relu'))
autoencoder.add(Dense(30, activation='relu'))  
autoencoder.add(Dense(128, activation='relu'))
autoencoder.add(Dense(256, activation='relu'))
autoencoder.add(Dense(4999, activation='sigmoid'))
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=10, batch_size=100, validation_data=(X_test, X_test))


encoder = Sequential(autoencoder.layers[:3])


reduced_features = encoder.predict(scaled_features)


class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.dropout = nn.Dropout(0.2)  
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.fc3(x)
        return x

output_dim = y_train.shape[1]

hidden_dim = 500 
model = FFNN(encoding_dim, hidden_dim, output_dim).to(device)  

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



for epoch in range(num_epochs):
    running_loss = 0.0
    for data in dataloader:
        inputs, labels = data
        inputs_encoded, _ = autoencoder(inputs)  # Use the autoencoder to reduce the dimensionality of the inputs
        optimizer.zero_grad()
        outputs = model(inputs_encoded)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')



with torch.no_grad():
    encoded_test_data, _ = autoencoder(X_test_tensor)


model.eval()  
with torch.no_grad():
    y_pred = model(encoded_test_data)

y_pred = y_pred.cpu().numpy()
y_test = y_test_tensor.cpu().numpy()

r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]


for i, score in enumerate(r2_scores):
    print(f"R² score for label {i}: {score:.4f}")
