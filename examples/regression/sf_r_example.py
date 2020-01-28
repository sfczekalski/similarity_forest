from simforest import SimilarityForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression, load_svmlight_file, load_wine, make_friedman1, load_boston
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
from scipy.spatial import distance
import pandas as pd
import time
import builtins

def get_forest_fires_dataset():
    df = pd.read_csv('../data/forestfires.csv')
    df['month'] = LabelEncoder().fit_transform(df['month'])
    df['day'] = LabelEncoder().fit_transform(df['day'])
    y, X = df.pop('area'), df

    return y, X


'''df = pd.read_csv('../data/AirQualityUCI.csv', sep=',')
df.drop(columns=['Date', 'Time', 'AH', 'val1', 'val2', 'val3', 'val4', 'val5'], inplace=True)
df.dropna(inplace=True)
print(df.head())

y, X = df.pop('RH'), df'''

'''df = pd.read_csv('../data/winequality-white.csv', sep=';')
df.dropna(inplace=True)
print(df.head())

y, X = df.pop('quality'), df'''


def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def get_who_dataset():
    df = pd.read_csv('../data/Life Expectancy Data.csv')
    '''df['Country'] = LabelEncoder().fit_transform(df['Country'])
    df['Status'] = LabelEncoder().fit_transform(df['Status'])'''
    df = pd.concat([df, pd.get_dummies(df['Country'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Status'])], axis=1)
    df.drop(columns=['Country', 'Status'], inplace=True)
    df.dropna(inplace=True)
    df = downcast_dtypes(df)
    y, X = df.pop('Life expectancy '), df

    return y, X


X, y = load_svmlight_file('../data/mpg')
X = X.toarray()
#X, y = load_boston(return_X_y=True)
'''X, y = load_svmlight_file('../data/abalone')
X = X.toarray()'''
#X, y = make_friedman1(n_samples=1000, random_state=42)

#X = SelectKBest(f_regression, k=8).fit_transform(X, y)
y = y + np.abs(np.min(y))
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestRegressor(random_state=1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(f'Random Forest R2 score: {r2_score(y_test, rf_pred)}')
print(f'Random Forest MSE: {mean_squared_error(y_test, rf_pred)}')
print(f'RF average tree depth: {np.mean([t.get_depth() for t in rf.estimators_])}')

start = time.time()
# Fit predict for both classifiers
sf = SimilarityForestRegressor(criterion='variance', n_estimators=100)
sf.fit(X_train, y_train)
print(f'Fit time: {time.time() - start} s')
start = time.time()
sf_pred = sf.predict(X_test)
print(f'Predict time: {time.time() - start} s')
print(f'Similarity Forest R2 score: {r2_score(y_test, sf_pred)}')
print(f'Similarity Forest MSE: {mean_squared_error(y_test, sf_pred)}')
print(f'SF average tree depth: {np.mean([t.get_depth() for t in sf.estimators_])}')

# Scale predictions for plotting
'''sf_pred = (sf_pred - np.min(sf_pred))/np.ptp(sf_pred)
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
