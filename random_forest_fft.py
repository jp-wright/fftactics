import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
# from src.roc import plot_roc

def load_data():
    df = pd.read_csv('/Users/jpw/Dropbox/Data_Science/jp_projects/2017-01-25_fft_class_stats/data/fft_class_stats_jp.csv')

    # y_classes = df['Association'].unique()
    # y = df.pop('Association').values
    # X = df.iloc[:, 6:].values
    # df = df.iloc[:, 6:]
    # Data cols start at col7
    y = df['Association']
    X = df.iloc[:, 7:]

    return df, X, y

def plot_trees(X_train, X_test, y_train, y_test):
    num_trees_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 750, 1000]
    scores = []
    for num in num_trees_list:
        rf = RandomForestClassifier(n_estimators=num, oob_score=True)
        rf.fit(X_train, y_train)
        scores.append(rf.oob_score_)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(num_trees_list, scores, lw=2, marker='o', alpha=0.6, c='g')
    ax.grid(True)
    ax.set_ylim(min(scores) - 0.1, max(scores) + 0.1)
    ax.set_title('Accuracy vs. Number of Trees', fontsize=20)
    ax.set_xlabel('Number of Trees', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)

    plt.show()

def plot_feats(X_train, X_test, y_train, y_test):
    max_feats = X.shape[1]
    scores = []
    for num in xrange(1, max_feats+1):
        rf = RandomForestClassifier(max_features=num, oob_score=True)
        rf.fit(X_train, y_train)
        scores.append(rf.oob_score_)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, max_feats+1), scores, lw=2, marker='o', alpha=0.6, c='g')
    ax.grid(True)
    ax.set_ylim(min(scores) - 0.1, max(scores) + 0.1)
    ax.set_title('Accuracy vs. Number of Features', fontsize=20)
    ax.set_xlabel('Number of Features', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)

    plt.show()

def comp_classifiers(X_train, X_test, y_train, y_test, avg=None):
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()

    clfs = [lr, knn, dt]

    scores_lst = []

    for classifier in clfs:
        classifier.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, classifier.predict(X_test))
        precision = precision_score(y_test, classifier.predict(X_test), average=avg)
        recall = recall_score(y_test, classifier.predict(X_test), average=avg)
        scores_lst.append([accuracy, precision, recall])

    return clfs, scores_lst

def plot_rocs():
    pass
    cl_list = [LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

    for cl in cl_list:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        plot_roc(X_test, y_test, cl)
    plt.show()

def feature_importance(rf, col_labels=None, importances=None, err=True):
    # plt.style.use('ggplot')
    if isinstance(importances, type(None)):
        importances = np.round(rf.feature_importances_, 2)
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    idxs = np.argsort(importances)[::-1]

    if isinstance(col_labels, type(None)):
        col_labels = {}
    else:
        col_labels = {idx: label for idx, label in enumerate(col_labels)}

    # Print the ranking
    print('Feature ranking:')
    for feat in range(importances.shape[0]):
        print("{}. {} ({})".format(feat+1, col_labels.get(idxs[feat], idxs[feat]), importances[idxs[feat]]))

    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    clr = 'orange'
    if err:
        plt.bar(range(importances.shape[0]), importances[idxs], yerr=std[idxs], align='center', color=clr)
    else:
        plt.bar(range(importances.shape[0]), importances[idxs], align='center', color=clr)

    xticks = [col_labels.get(idx, idx) for idx in idxs]
    plt.xticks(range(importances.shape[0]), xticks, rotation=-45)
    plt.xlim([-1, importances.shape[0]])
    plt.tight_layout()

def grid_search(X_train, y_train):
    rf = RandomForestClassifier()
    cv_folds, cv_jobs = 3, 3
    all_clfs = list()
    avg = 'weighted'
    recall_scorer = make_scorer(recall_score, average=avg)
    metrics = ['accuracy', 'precision', 'recall', 'f1_weighted', 'roc_auc', 'average_precision']
    parameters = {
    'n_estimators': [300, 400],
    'max_features': ['sqrt'],
    'max_depth': [3, 5, 10, 17, 25, int(.25 * X_train.shape[1]), None],
    'max_leaf_nodes': [3, 5, 10, 17, 25, int(.25 * X_train.shape[1]), None],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3],
    'min_weight_fraction_leaf': [0., .01, .05, .1],
    'class_weight': ['balanced', None],
    'criterion': ['gini'],
    'oob_score': [True],
    'random_state': [462]}

    [make_scorer(met, average=avg) for met in ['precision', 'recall']]
    metrics = [recall_scorer]
    metric = metrics[0]
    duration_est_secs = .75 - (.15 * (cv_jobs - 1))
    fits = len(metrics) * np.prod([len(v) for v in parameters.values()]) * cv_folds
    print ("\nNow running GridSearchCV. A rough max estimate on duration is: {d:.2f} mins. \nStart time is: {t1}".format(d=(fits * duration_est_secs)/60, t1=time.strftime('%H:%M')))
    for idx, metric in enumerate(metrics):
        t1 = time.time()
        print ("Now Grid Searching {f} fits on {i} out of {l}.".format(f=fits, i=metric, l=metrics))
        clf = GridSearchCV(rf, parameters, scoring=metric, cv=cv_folds, verbose=1, n_jobs=cv_jobs, pre_dispatch='2*n_jobs').fit(X_train, y_train)
        # accuracy_score(y_test, clf.predict(X_test))
        print ('{m:.2f} minutes for GridSearchCV, at {spf:.2f} seconds per fit.'.format(m=(time.time() - t1)/60, spf=((time.time() - t1)/fits)))
        all_clfs.append(clf)
        best_results = create_best_results(clf)
        print_best_results(best_results, clf)
        create_param_df(best_results)
        dfp = pd.DataFrame(clf.cv_results_)
    return clf, all_clfs, best_results, dfp

def create_best_results(clf):
    best_results = {
    'Param_Grid': clf.param_grid,
    'Best_Parameters': clf.best_params_,
    'Test_Score {s}'.format(s=clf.scoring.capitalize()): clf.best_score_,
    'Train_Score {s}'.format(s=clf.scoring.capitalize()): clf.cv_results_['mean_train_score'][clf.best_index_],
    'Test_StDev': clf.cv_results_['std_test_score'][clf.best_index_],
    'Train_StDev': clf.cv_results_['std_train_score'][clf.best_index_],
    'Test_Std_Error_Mean': clf.cv_results_['std_test_score'][clf.best_index_] / clf.cv,
    'Train_Std_Error_Mean': clf.cv_results_['std_train_score'][clf.best_index_] / clf.cv,
    'Test_Score_Ranks': clf.cv_results_['rank_test_score'],
    'Best_Rank_Indices': [idx for idx,val in enumerate(clf.cv_results_['rank_test_score']) if val == 1],
    'Fit_Time(s)': clf.cv_results_['mean_fit_time'][clf.best_index_],
    'Score_Time(s)': clf.cv_results_['mean_score_time'][clf.best_index_],
    'Gain': clf.best_score_ - np.min(clf.cv_results_['mean_test_score'])}
    return best_results

def print_best_results(best_results, clf):
    for k, v in best_results.items():
        if k == 'Best_Parameters':
            continue
        if isinstance(v, dict):
            print ('\n'+k+':')
            print ('\n'.join('{k}:  {v}'.format(k=k, v=str(v)) for k,v in best_results[k].items()))
            print ('')
        else:
            if k == 'Best_Rank_Indices':
                for idx, num in enumerate(v):
                    print('\nBest_Params #{i}:'.format(i=idx+1))
                    print('\n'.join('{k}:  {v}'.format(k=k, v=str(v)) for k,v in clf.cv_results_['params'][num].items()))
                print('')
            try:
                print ('{k}:  {v:.3f}'.format(k=k, v=v))
            except (TypeError, ValueError):
                print ('{k}:  {v}'.format(k=k, v=str(v)))
    print ("")

def examine_prediction_confidence(df_test, y_pred, y_proba, y_test):
    ## confidence values
    max_proba = 100 * np.max(y_proba, axis=1)
    rnd_proba = [float(str(x)[:5]) for x in max_proba]
    df_test['Pred_Assoc'] = y_pred
    df_test['Pred_Proba'] = rnd_proba
    df_test['Pred_Correct?'] = ['Yes' if df_test.loc[idx, 'Association'] == df_test.loc[idx, 'Pred_Assoc'] else 'No' for idx in df_test.index]

    # return sort in desc order
    proba_sort = np.argsort(y_proba, axis=1)[:, ::-1]
    second_pred = proba_sort[:, 1]
    second_proba = [100 * y_proba[row, idx] for row, idx in zip(range(y_proba.shape[0]), second_pred)]
    rnd_second_proba = [float(str(x)[:5]) for x in second_proba]
    df_test['2nd_Pred'] = [rf.classes_[idx] for idx in second_pred]
    df_test['2nd_Proba'] = rnd_second_proba
    df_test['Pred_Delta'] = df_test['Pred_Proba'] - df_test['2nd_Proba']
    df_test['2nd_Pred_Correct?'] = ['Yes' if df_test.loc[idx, 'Association'] == df_test.loc[idx, '2nd_Pred'] else 'No' for idx in df_test.index]

    dft = df_test[['Identity', 'Association', 'Pred_Assoc', 'Pred_Proba', 'Pred_Correct?', '2nd_Pred', '2nd_Proba', 'Pred_Delta', '2nd_Pred_Correct?']]


    total = np.sum([1 if cell == 'Yes' else 0 for cell in np.ravel([df_test['Pred_Correct?'], df_test['2nd_Pred_Correct?']])])

    print("Accuracy on first prediction: {:.2f}".format(100* accuracy_score(y_test, y_pred)))
    print("Accuracy on first two predictions: {:.2f}".format(100 * total/df_test.shape[0]))

    print(dft)
    print(dft.groupby('Pred_Correct?').mean())
    print(dft[dft['Pred_Correct?'] == 'No'])
    return dft


if __name__ == '__main__':
    df, X, y = load_data()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.75, stratify=y)
    X_train, X_test, y_train, y_test , df_train, df_test = train_test_split(X, y, df, train_size=.75, stratify=y)
    clf, all_regs, best_results, dfp = grid_search(X_train, y_train)
    best_results = create_best_results(clf)
    print_best_results(best_results, clf)

    # rf = RandomForestClassifier(n_estimators=300, n_jobs=3, oob_score=True, verbose=False, class_weight='balanced', max_leaf_nodes=25, max_depth=5).fit(X_train, y_train)
    # y_pred = rf.predict(X_test)
    # y_proba = rf.predict_proba(X_test)
    #
    # dft = examine_prediction_confidence(df_test, y_pred, y_proba, y_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print ('Accuracy: ', accuracy)
    # print ('OOB Score: ', rf.oob_score_)
    # print 'Important Features: ', rf.feature_importances_
    # columns = df.columns
    # feature_list = [(feature, fi) for feature, fi in zip(columns, rf.feature_importances_)]
    # sorted_list = sorted(feature_list, key=lambda x: x[1], reverse=True)

    # plot_trees(X_train, X_test, y_train, y_test)
    # plot_feats(X_train, X_test, y_train, y_test)
    '''
    avg = None
    # accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg)
    recall = recall_score(y_test, y_pred, average=avg)

    if avg == None:
        scores = [precision, recall, 'Precision', 'Recall']
        for idx, metric in enumerate(scores[:2]):
            print ("\n{0}".format(scores[idx+2]))
            for idy, assoc in enumerate(y_classes):
                print ("{0} - {1}".format(assoc, metric[idy]))
    else:
        print ('\nrf \nAccuracy: {}, \nPrecision: {}, \nRecall: {}'.format(accuracy, precision, recall))


    print ('Classification Report: ')
    # print (classification_report(y_test, y_pred))#, target_names=[x for x in range(9)]))
    '''


    # print ('Confusion_matrix: ')
    # mat = confusion_matrix(y_test, y_pred)
    # # mat = confusion_matrix(y, y)
    # y_name = sorted(y_test.unique())
    # ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=y_name, yticklabels=y_name, robust=True)
    #
    # plt.xlabel('True Label')
    # plt.ylabel('Predicted Label')
    # plt.xticks(rotation='vertical')
    # plt.yticks(rotation='horizontal')
    #
    # f = plt.gcf()
    # f.subplots_adjust(bottom=0.16, left=0.12, right=.99, top=0.95)
    # feature_importance(rf, X.columns.tolist())
    # plt.show()
