from simforest import SimilarityTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression, load_svmlight_file, load_wine, make_friedman1, load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt


#y = y + np.abs(np.min(y))

X, y = load_svmlight_file('../data/space_ga')
X = X.toarray()



'''df = pd.read_csv('../data/AirQualityUCI.csv', sep=',')
df.drop(columns=['Date', 'Time', 'AH', 'val1', 'val2', 'val3', 'val4', 'val5'], inplace=True)
df.dropna(inplace=True)
print(df.head())

y, X = df.pop('RH'), df'''

#X, y = load_boston(return_X_y=True)


#X = SelectKBest(f_regression, k=8).fit_transform(X, y)
y = y + np.abs(np.min(y))
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Compare regressors' accuracy
print(f'Decision Tree R2 score: {r2_score(y_test, dt_pred)}')
print(f'Decision Tree MSE: {mean_squared_error(y_test, dt_pred)}')
print(f'Decision Tree depth: {dt.get_depth()}')

# Fit predict for both classifiers
st = SimilarityTreeRegressor(random_state=42, criterion='variance', discriminative_sampling=False)
st.fit(X_train, y_train)
st_pred = st.predict(X_test)
print(f'Similarity Tree R2 score: {r2_score(y_test, st_pred)}')
print(f'Similarity Tree MSE: {mean_squared_error(y_test, st_pred)}')
print(f'SF average tree depth: {st.get_depth()}')

'''# Scale predictions for plotting
sf_pred = (sf_pred - np.min(sf_pred))/np.ptp(sf_pred)
rf_pred = (rf_pred - np.min(rf_pred))/np.ptp(rf_pred)

# Plot classifiers' predictions
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=sf_pred,
            s=50, edgecolor='k', alpha=1.0, cmap='BuPu', lw=0, facecolor='0.5')
plt.title("Similarity Tree")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=rf_pred,
            s=50, edgecolor='k', alpha=1.0, cmap='BuPu', lw=0, facecolor='0.5')
plt.title("Decision Tree")
plt.show()
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test,
            s=50, edgecolor='k', alpha=1.0, cmap='BuPu', lw=0, facecolor='0.5')
plt.title("True")
plt.show()'''
