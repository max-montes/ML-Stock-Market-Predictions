from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from Attributes import *
import yfinance as yf

def get_attributes(ticker, period):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period)
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    non_target_attribs = pd.DataFrame()
    non_target_attribs['10 Day Simple Moving Average'] = get_SMA(data)
    non_target_attribs['10 Day Momentum'] = get_momentum(data)
    non_target_attribs['14 Day Stochastic Oscillator K%'] = get_stochastic_oscillator_k_percent(data, 14)
    non_target_attribs['14 Day Stochastic D%'] = get_stochastic_oscillator_k_percent_moving_average(data, 14)
    non_target_attribs['14 Day Relative Strength Index'] = get_relative_strength_index(data, 14)
    non_target_attribs['Moving Average Convergence Divergence'] = get_moving_average_convergence_divergence(data)
    non_target_attribs['14 Day Williams % R'] = get_williams_percent_range(data, 14)
    non_target_attribs['A/D Index'] = get_a_d_index(data)
    non_target_attribs['20 Day Commodity Channel Index'] = get_commodity_channel_index(data, 20)
    non_target_attribs = non_target_attribs.dropna()
    data = data.loc[non_target_attribs.index]
    target_attribs = (data['Close'] > data['Open'])
    return non_target_attribs, target_attribs

def print_accuracy_metrics(actuals, predicts):
    accuracy = accuracy_score(actuals, predicts)
    balanced_accuracy = balanced_accuracy_score(actuals, predicts)
    precision = precision_score(actuals, predicts)
    recall = recall_score(actuals, predicts)
    f1 = f1_score(actuals, predicts)
    roc_auc = roc_auc_score(actuals, predicts)
    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")

def model_ticker(ticker, period = 182):
    X, y = get_attributes(ticker, period)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)
    hyper_grid = {
        'n_estimators': [50, 75, 100, 125, 150],
        'learning_rate': [0.5, 0.75, 1.0, 1.25, 1.5]
    }
    splitter = TimeSeriesSplit(n_splits=5, max_train_size=730)
    precision_scorer = make_scorer(precision_score, zero_division=1)
    grid_search = GridSearchCV(AdaBoostClassifier(algorithm='SAMME', random_state=1), hyper_grid, scoring=precision_scorer, cv=splitter)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    model = grid_search.best_estimator_
    y_preds = model.predict(X_test)
    print_accuracy_metrics(y_test, y_preds)

#Note -- PERIOD:
#29 gets 1st value with all non_target attributes not nan
#48 is first with enough data for cross-validation
model_ticker('MSFT', 1461)