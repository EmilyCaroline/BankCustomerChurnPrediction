import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.process_data import preporcess_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import f1_score
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

plt.style.use('seaborn-muted')

ds = pd.read_csv('data/BankChurners.csv')
ds = ds.iloc[:, :-2]  # discard two last columns
ds = ds.iloc[:, 1:]  # remove IdClient

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

dataset1_csv_file = "data/Dataset2.csv"
dataset2_csv_file = "data/Credit Card Customer Churn.csv"
merged_df = preporcess_data(dataset1_csv_file, dataset2_csv_file)

columns_obj = []
count = 0
for x in merged_df.columns:
    if merged_df.dtypes[count] == object:
        columns_obj.append(x)
        count = count+1
    else:
        count = count+1

# convert into numeric values
le = LabelEncoder()
for col in columns_obj:
    merged_df[col+'_dummy'] = le.fit_transform(merged_df[col])

# select object column name
object_columns = list(merged_df.select_dtypes(include=['object']).columns)

# drop column with data type object
df = merged_df.drop(columns=object_columns)


def random_forest():
    y = df['Exited'].values
    X = df.drop(columns=['Exited'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101)

    RF = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1,
                                random_state=50, max_features="auto",
                                max_leaf_nodes=30)
    RF.fit(X_train, y_train)
    # Make predictions
    predictiont = RF.predict(X_test)

    # predict labels for test set
    y_pred = RF.predict(X_test)
    accuracy_score_val = metrics.accuracy_score(y_test, predictiont)
    # Evaluate the test set RMSE
    rmse_test = MSE(y_test, predictiont)**(1/2)
    rf_val = float("{:.2f}".format(rmse_test))

    # compute F1-score
    f1 = f1_score(y_test, y_pred)

    important_feature = RF.feature_importances_
    evaluate = pd.Series(important_feature, index=X.columns.values)
    chart_data = evaluate.sort_values()[-10:]
    plot = chart_data.plot(
        kind='barh', title="Feature Importances: Random Forest", ylabel="Feature", xlabel="Relative Importance", color="#5B8BAB")
    fig = plot.get_figure()

    y_pred = RF.predict(X_test)
    predictions_prob = RF.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    cm_plot1 = plot_confusion_matrix(
        cm, target_names=['Not Exited', 'Exited'], normalize=False)
    cm_fig1 = cm_plot1[0]
    cm_fig1_cap = cm_plot1[1]

    cm_plot2 = plot_confusion_matrix(cm, target_names=[
        'Not Exited', 'Exited'], normalize=True, title='Confusion Matrix (Normalized)')

    cm_fig2 = cm_plot2[0]
    cm_fig2_cap = cm_plot2[1]

    table_of_models = classification_report_to_dataframe(
        y_test, y_pred, predictions_prob, model_name='Random Forest')

    return accuracy_score_val, rf_val, f1, fig, cm_fig1_cap, cm_fig1, cm_fig2_cap, cm_fig2, table_of_models


def classification_report_to_dataframe(true, predictions, predictions_proba, model_name, balanced='no'):
    a = classification_report(true, predictions, output_dict=True)
    zeros = pd.DataFrame(data=a['0'], index=[0]).iloc[:, 0:3].add_suffix('_0')
    ones = pd.DataFrame(data=a['1'], index=[0]).iloc[:, 0:3].add_suffix('_1')
    df = pd.concat([zeros, ones], axis=1)
    temp = list(df)
    df['Model'] = model_name
    df['Balanced'] = balanced
    df['Accuracy'] = accuracy_score(true, predictions)
    df['Balanced_Accuracy'] = balanced_accuracy_score(true, predictions)
    df['AUC'] = roc_auc_score(true, predictions_proba, average='macro')
    df = df[['Model', 'Balanced', 'Accuracy', 'Balanced_Accuracy', 'AUC'] + temp]
    return df


def plot_confusion_matrix(cm, target_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        caption = "Normalized confusion matrix"
    else:
        caption = 'Confusion matrix, without normalization'

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig, caption


def split_train_test():
    # split the data set into 80% and 20%, train and test sets respectively.

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Check the shape of the training and testing sets
    print('Number of clients in the dataset: {}'.format(len(df)))
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Find the number of clients that have exited the program in the training set
    train_exited = y_train.sum()
    train_total = len(y_train)
    train_exited_percent = train_exited / train_total * 100
    print("Number of clients that have exited in the training set:",
          train_exited, f"({train_exited_percent:.2f}%)")

    # Find the number of clients that haven't exited the program in the testing set
    test_not_exited = (y_test == 0).sum()
    test_total = len(y_test)
    test_not_exited_percent = test_not_exited / test_total * 100
    print("Number of clients that haven't exited in the testing set:",
          test_not_exited, f"({test_not_exited_percent:.2f}%)")

    features = list(df.drop('Exited', axis=1))
    target = 'Exited'

    # Instantiate StandardScaler
    scaler = StandardScaler()

    # Fit scaler on training features
    scaler.fit(X_train)

    # Transform training and test features
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {'penalty': ['l1', 'l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100]}

    # Instantiate a logistic regression model
    lr_model = LogisticRegression()

    # Perform grid search cross-validation
    grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='f1')

    # Fit the grid search object on the training data
    grid_search.fit(X_train, y_train)

    # Extract the best estimator and best score
    best_lr = grid_search.best_params_
    best_score_lr = grid_search.best_score_

    return X_train, X_test, y_train, y_test, features, target, grid_search


def logistic_regression():
    # Create a logistic regression model object with best parameters

    best_lr = LogisticRegression(C=1, penalty='l2')

    data_split = split_train_test()
    X_train = data_split[0]
    X_test = data_split[1]
    y_train = data_split[2]
    y_test = data_split[3]
    features = data_split[4]
    target = data_split[5]
    grid_search = data_split[6]

    # Create a logistic regression model object with best parameters
    best_lr = LogisticRegression(C=1, penalty='l2')

    # Fit the model on training data
    best_lr.fit(X_train, y_train)

    # Calculate feature importances
    importances = abs(best_lr.coef_[0])
    importances = 100.0 * (importances / importances.max())
    indices = np.argsort(importances)

    # print(importances[indices])
    # Define size
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))

    plt.title('Feature Importances: Logistic Regression', fontsize=20)

    # Create a horizontal bar chart of the feature importances
    plt.barh(range(len(indices)),
             importances[indices], align='center', color='#5B8BAB')
    plt.yticks(range(len(indices)), [features[i]
               for i in indices], fontsize=14)
    plt.xlabel('Relative Importance', fontsize=16)
    plt.ylabel('Feature', fontsize=16)

    # Remove the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.grid(axis='x', alpha=0.5)

    # Make predictions on test data
    predictions = best_lr.predict(X_test)
    predictions_prob = grid_search.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions)

    cm_plot1 = plot_confusion_matrix(
        cm, target_names=['Not Exited', 'Exited'], normalize=False)
    cm_fig1 = cm_plot1[0]
    cm_fig1_cap = cm_plot1[1]

    cm_plot2 = plot_confusion_matrix(cm, target_names=[
        'Not Exited', 'Exited'], normalize=True, title='Confusion Matrix (Normalized)')
    cm_fig2 = cm_plot1[0]
    cm_fig2_cap = cm_plot1[1]

    classification_report_df = classification_report_to_dataframe(
        y_test, predictions, predictions_prob, model_name='Logistic Regression')

    return fig, cm_fig1_cap, cm_fig1, cm_fig2_cap, cm_fig2, classification_report_df

def gradient_boosting_sklearn():
    data_split = split_train_test()
    X_train = data_split[0]
    X_test = data_split[1]
    y_train = data_split[2]
    y_test = data_split[3]
    features = data_split[4]
    target = data_split[5]
    grid_search = data_split[6]

    # Define the parameter grid to search over
    param_grid = {'max_depth': [2, 3, 4, 6, 10, 15],
                'n_estimators': [50, 100, 300, 500]}

    # Fit Gradient Boosting model with best hyperparameters
    model = GradientBoostingClassifier(max_depth=3, n_estimators=100)
    model.fit(X_train, y_train)

    # Calculate feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)

    # Print feature importances
    # print('Feature Importances:')
    # print(importances[indices])
    # print([features[i] for i in indices])
    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Feature Importances: Complete Gradient Boosting', fontsize=16)
    ax.barh(range(len(indices)), importances[indices], align='center', color='#5B8BAB')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices], fontsize=12)
    ax.set_xlabel('Relative Importance', fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.01, i, f'{v:.2f}', fontsize=12)
    plt.tight_layout()

     # Make predictions on test data
    y_pred = grid_search.predict(X_test)
    y_pred_prob = grid_search.predict_proba(X_test)[:,1]

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    

    cm_plot1 = plot_confusion_matrix(cm, target_names=['Not Churned', 'Churned'], normalize=False)
    cm_fig1 = cm_plot1[0]
    cm_fig1_cap = cm_plot1[1]

    cm_plot2 = plot_confusion_matrix(cm, target_names=['Not Churned', 'Churned'], normalize=True, title='Confusion Matrix (Normalized)')
    cm_fig2 = cm_plot2[0]
    cm_fig2_cap = cm_plot2[1]

    classification_report_df = classification_report_to_dataframe(y_test, y_pred, y_pred_prob, model_name='Gradient Boosting (Sklearn)')

    return fig, cm_fig1_cap, cm_fig1, cm_fig2_cap, cm_fig2, classification_report_df

    data_split = split_train_test()
    X_train = data_split[0]
    X_test = data_split[1]
    y_train = data_split[2]
    y_test = data_split[3]
    features = data_split[4]
    target = data_split[5]
    grid_search = data_split[6]

    
    # Define hyperparameters to search over
    param_grid = {'max_depth': [2, 3, 4, 6, 10, 15],
                'n_estimators': [50, 100, 300, 500],
                'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5]}

    # Create an XGBoost model object
    xgb = XGBClassifier()

    # Create a grid search object with cross-validation
    model_XGB = GridSearchCV(xgb, param_grid, cv=5, n_jobs=10)

    # Fit the grid search object on the training data
    model_XGB.fit(X_train, y_train)

    # Print the best hyperparameters
    best_max_depth = model_XGB.best_params_['max_depth']
    best_n_estimators = model_XGB.best_params_['n_estimators']
    best_learning_rate = model_XGB.best_params_['learning_rate']
    print("Best max depth:", best_max_depth)
    print("Best n estimators:", best_n_estimators)
    print("Best learning rate:", best_learning_rate)

    # Use the best hyperparameters to fit the model on the training data
    xgb = XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.2)
    xgb.fit(X_train, y_train)

    print(xgb)

    # # Predict on the test data
    # y_pred = xgb.predict(X_test)
    # y_pred_prob = xgb.predict_proba(X_test)[:,1]

    # importances = xgb.feature_importances_
    # indices = np.argsort(importances)

    # # Plot feature importances
    # fig, ax = plt.subplots(figsize=(12, 8))
    # plt.figure(figsize=(12, 8))
    # plt.title('Feature Importances: Complete Extreme Gradient Boosting (XGBoost)', fontsize=18)
    # plt.barh(range(len(indices)), importances[indices], align='center', color='orange')
    # plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=14)
    # plt.xlabel('Relative Importance', fontsize=14)
    # plt.grid(axis='x', linestyle='--', alpha=0.7)
    # plt.tight_layout()

    return fig