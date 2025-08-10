# Rental Bike Prediction System - Complete ML Model
# System Approach: Predicting bike rental demand using machine learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Plotly Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BikeRentalPredictionSystem:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_performance = {}

    def generate_sample_data(self, n_samples=10000):
        np.random.seed(42)
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(hours=x) for x in range(n_samples)]
        data = {
            'datetime': dates,
            'season': np.random.choice([1, 2, 3, 4], n_samples),
            'holiday': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
            'workingday': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'weather': np.random.choice([1, 2, 3, 4], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
            'temp': np.random.normal(20, 10, n_samples),
            'atemp': np.random.normal(20, 8, n_samples),
            'humidity': np.random.uniform(0, 100, n_samples),
            'windspeed': np.random.exponential(2, n_samples),
        }
        df = pd.DataFrame(data)
        base_demand = 100
        season_effect = df['season'].map({1: 0.8, 2: 1.2, 3: 1.3, 4: 0.9})
        df['hour'] = df['datetime'].dt.hour
        hour_effect = df['hour'].apply(lambda x: 1.5 if 7 <= x <= 9 or 17 <= x <= 19 else 1.2 if 10 <= x <= 16 else 0.3 if 22 <= x or x <= 5 else 0.8)
        weather_effect = df['weather'].map({1: 1.0, 2: 0.8, 3: 0.6, 4: 0.3})
        temp_effect = 1 - 0.02 * np.abs(df['temp'] - 22.5)
        temp_effect = np.clip(temp_effect, 0.1, 1.3)
        workingday_effect = df['workingday'] * 1.2 + (1 - df['workingday']) * 0.9
        df['count'] = (base_demand * season_effect * hour_effect * weather_effect * temp_effect * workingday_effect * np.random.uniform(0.8, 1.2, n_samples))
        df['count'] = np.round(df['count']).astype(int)
        df['count'] = np.clip(df['count'], 0, 1000)
        return df

    def preprocess_data(self, df):
        df = df.copy()
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['dayofyear'] = df['datetime'].dt.dayofyear
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        df['is_night'] = ((df['hour'] <= 5) | (df['hour'] >= 22)).astype(int)
        df['temp_humidity'] = df['temp'] * df['humidity'] / 100
        df['temp_windspeed'] = df['temp'] * df['windspeed']
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        for col in ['temp', 'atemp', 'humidity', 'windspeed']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower, upper)
        return df

    def prepare_features(self, df):
        feature_cols = [
            'season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'hour', 'dayofweek', 'month', 'year',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'dayofweek_sin', 'dayofweek_cos', 'is_weekend',
            'is_rush_hour', 'is_night', 'temp_humidity', 'temp_windspeed']
        self.feature_names = feature_cols
        X = df[feature_cols]
        y = df['count'] if 'count' in df.columns else None
        return X, y

    def train_models(self, X_train, y_train):
        configs = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Tree': DecisionTreeRegressor(max_depth=10),
            'RF': RandomForestRegressor(n_estimators=100, n_jobs=-1),
            'GB': GradientBoostingRegressor(n_estimators=100)
        }
        if XGBOOST_AVAILABLE:
            configs['XGB'] = xgb.XGBRegressor(n_estimators=100, n_jobs=-1)
        for name, model in configs.items():
            model.fit(X_train, y_train)
            self.models[name] = model

    def evaluate_models(self, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
        self.model_performance = results
        self.best_model = self.models[max(results, key=lambda x: results[x]['R2'])]
        return results

    def plot_results(self, X_test, y_test):
        fig = make_subplots(rows=2, cols=2, subplot_titles=['R2 Scores', 'Actual vs Predicted', 'Residuals', 'Feature Importance'])
        if self.model_performance:
            names = list(self.model_performance.keys())
            r2s = [v['R2'] for v in self.model_performance.values()]
            fig.add_trace(go.Bar(x=names, y=r2s), row=1, col=1)
        if self.best_model:
            y_pred = self.best_model.predict(X_test)
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'), row=1, col=2)
            residuals = y_test - y_pred
            fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers'), row=2, col=1)
            if hasattr(self.best_model, 'feature_importances_'):
                fi = self.best_model.feature_importances_
                sorted_idx = np.argsort(fi)[-10:]
                fig.add_trace(go.Bar(x=fi[sorted_idx], y=[self.feature_names[i] for i in sorted_idx], orientation='h'), row=2, col=2)
        fig.update_layout(height=800)
        fig.show()

    def predict(self, X):
        if not self.best_model:
            raise ValueError("No model trained.")
        return self.best_model.predict(X)


def main():
    system = BikeRentalPredictionSystem()
    df = system.generate_sample_data(8760)
    df = system.preprocess_data(df)
    X, y = system.prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    system.train_models(X_train, y_train)
    results = system.evaluate_models(X_test, y_test)
    for model, metrics in results.items():
        print(f"\nModel: {model}")
        for metric, val in metrics.items():
            print(f"{metric}: {val:.4f}")
    system.plot_results(X_test, y_test)
    preds = system.predict(X_test[:5])
    print("\nSample Predictions:", preds)

if __name__ == '__main__':
    main()
