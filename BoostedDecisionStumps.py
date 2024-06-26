from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from Attributes import *
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


def get_attributes(ticker, period, days_to_target=1):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period + days_to_target + 39)
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

    # Z-score normalization
    scaler = StandardScaler()
    non_target_attribs_normalized = scaler.fit_transform(non_target_attribs)
    non_target_attribs_normalized = pd.DataFrame(non_target_attribs_normalized, columns=non_target_attribs.columns, index=non_target_attribs.index)

    target_attribs = (data['Close'] > data['Close'].shift(days_to_target))
    target_attribs = target_attribs.loc[non_target_attribs.index]
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

    # Calculate confusion matrix
    cm = confusion_matrix(actuals, predicts)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def model_ticker(ticker, period=182):
    X, y = get_attributes(ticker, period)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

    hyper_grid = {
        'n_estimators': [1, 2, 4, 6, 10],
        'learning_rate': [0.25, 0.5, .75, 1, 1.25]
    }

    splitter = TimeSeriesSplit(n_splits=5)
    estimator = DecisionTreeClassifier(max_depth=3)  # Regularization by limiting max depth
    grid_search = GridSearchCV(AdaBoostClassifier(estimator=estimator, algorithm='SAMME', random_state=1), hyper_grid, scoring='accuracy', cv=splitter)
    grid_search.fit(X_train, y_train)

    # Extracting results from grid search
    results = grid_search.cv_results_
    params = results['params']
    scores = results['mean_test_score'].reshape(len(hyper_grid['learning_rate']), len(hyper_grid['n_estimators']))

    # Plotting hyperparameter tuning graphs
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting n_estimators vs. mean test score
    for i, lr in enumerate(hyper_grid['learning_rate']):
        ax[0].plot(hyper_grid['n_estimators'], scores[i, :], label=f"Learning Rate: {lr}", marker='o', linestyle=':')
    ax[0].plot(hyper_grid['n_estimators'], scores.mean(0), label=f"Average Score", lw=3)
    ax[0].set_xlabel('Number of Estimators')
    ax[0].set_ylabel('Mean Test Score (Accuracy)')
    ax[0].set_title('Hyperparameter Tuning: Number of Estimators')
    ax[0].set_xticks(hyper_grid['n_estimators'])
    ax[0].legend()
    ax[0].grid(True)

    # Plotting learning_rate vs. mean test score
    for i, n_est in enumerate(hyper_grid['n_estimators']):
        ax[1].plot(hyper_grid['learning_rate'], scores[:, i], label=f"# of Estimators: {n_est}", marker='o', linestyle=':')
    ax[1].plot(hyper_grid['learning_rate'], scores.mean(1), label=f"Average Score", lw=3)
    ax[1].set_xlabel('Learning Rate')
    ax[1].set_ylabel('Mean Test Score (Accuracy)')
    ax[1].set_title('Hyperparameter Tuning: Learning Rate')
    ax[1].set_xticks(hyper_grid['learning_rate'])
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    model = grid_search.best_estimator_
    y_preds = model.predict(X_test)
    print_accuracy_metrics(y_test, y_preds)

    # Plot feature importance
    feature_importance = model.feature_importances_
    feature_names = X.columns
    sorted_idx = feature_importance.argsort()
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()

# Note -- PERIOD:
# 29 gets 1st value with all non_target attributes not nan
# 48 is first with enough data for cross-validation

model_ticker('GOOG', 1461)