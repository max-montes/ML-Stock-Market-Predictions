from Top3Attributes import *
from Bottom3Attributes import *
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, make_scorer, recall_score, f1_score, roc_auc_score
import yfinance as yf

STOCK = 'MSFT'
PERIOD = 21

ticker = yf.Ticker(STOCK)
hist = ticker.history(period='max')

non_target_attribs = pd.DataFrame()
non_target_attribs['10 Day SMA'] = get_SMA(hist)
non_target_attribs['10 Day Momentum'] = get_momentum(hist)
non_target_attribs['Stochastic Oscillator K%'] = get_stochastic_oscillator_k_percent(hist, 14)
#non_target_attribs[''] = get_
#non_target_attribs[''] = get_
#non_target_attribs[''] = get_
non_target_attribs['Williams % R'] = get_williams_percent_range(hist, PERIOD)
non_target_attribs['A/D Index'] = get_a_d_index(hist)
non_target_attribs['CCI'] = get_commodity_channel_index(hist, PERIOD)
non_target_attribs = non_target_attribs.dropna()

X = non_target_attribs
y = (hist['Close'] > hist['Open'])[PERIOD - 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

hyper_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.5, 1.0, 1.5]
}

grid_search = GridSearchCV(AdaBoostClassifier(algorithm='SAMME'), hyper_grid, cv=5, scoring='precision')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

model = grid_search.best_estimator_

y_preds = model.predict(X_test)

accuracy = accuracy_score(y_test, y_preds)
balanced_accuracy = balanced_accuracy_score(y_test, y_preds)
precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)
roc_auc = roc_auc_score(y_test, y_preds)

print(f"Accuracy: {accuracy}")
print(f"Balanced Accuracy: {balanced_accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")