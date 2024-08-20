import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

#Prepare Plot
def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs. Predicted for {model_name}')
    plt.show()


#Read in data
data = pd.read_csv('nvidia_clean.csv')
X = data.drop(columns=['Close'])
y = data['Close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data for models that have the requirement
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print("Linear Regression RMSE:", mean_squared_error(y_test, lr_pred, squared=False))
plot_actual_vs_predicted(y_test, lr_pred, 'Linear Regression')

# K Nearest Neighbors
knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
print("K Nearest Neighbors RMSE:", mean_squared_error(y_test, knn_pred, squared=False))
plot_actual_vs_predicted(y_test, knn_pred, 'K Nearest Neighbors')

# Gradient Boosting
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print("Gradient Boosting RMSE:", mean_squared_error(y_test, gb_pred, squared=False))
plot_actual_vs_predicted(y_test, gb_pred, 'Gradient Boosting'
                         
# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest RMSE:", mean_squared_error(y_test, rf_pred, squared=False))
plot_actual_vs_predicted(y_test, rf_pred, 'Random Forest')

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree RMSE:", mean_squared_error(y_test, dt_pred, squared=False))
plot_actual_vs_predicted(y_test, dt_pred, 'Decision Tree')


# Support Vector Regression
svr = SVR()
svr.fit(X_train_scaled, y_train)
svr_pred = svr.predict(X_test_scaled)
print("Support Vector Regression RMSE:", mean_squared_error(y_test, svr_pred, squared=False))
plot_actual_vs_predicted(y_test, svr_pred, 'Support Vector Regressio

# LSTM Model
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
plot_actual_vs_predicted(y_test, lstm_pred, 'LSTM')

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.fit(X_train_lstm, y_train, epochs=200, verbose=0)

lstm_pred = lstm_model.predict(X_test_lstm)
print("LSTM RMSE:", mean_squared_error(y_test, lstm_pred, squared=False))

