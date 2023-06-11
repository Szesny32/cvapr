import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import umap
import json
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import time
import itertools
import seaborn as sns
#import addcopyfighandler # enables ctrl + c -> save matplotlib figure to clipboard

plt.rcParams.update({'font.size': 20})
#plt.style.use('seaborn')

class KickstartedPredict():
    def __init__(self, data_folder_path: str, num_of_files_to_load: int = 1) -> None:

        pd.options.display.max_columns = 9999
        self.data_folder_path: str = data_folder_path
        self.num_of_files_to_load: int = num_of_files_to_load
        self.df: pd.Dataframe = pd.DataFrame()
        self.columns_to_use: List[str] = ["backers_count", "blurb", "category", "country",
                               "created_at", "deadline", "goal", "launched_at",
                               "name", "staff_pick", "state", "usd_pledged" ]

    def run(self) -> None:
        self.load_data()
        self.prepare_data()
        #self.PCA()
        self.UMAP()
        #self.SISO()
        #self.prepare_plotsPCA()

    def load_data(self) -> None:
        """Load data to self.df dataframe. Param use_columns==None means all columns are used."""

        for i, filename in enumerate(os.scandir(Path(self.data_folder_path))):
            if filename.name.endswith('.csv'):
                i += 1
                if i > self.num_of_files_to_load:
                    break

                if filename.is_file():
                    if i == 1:
                        #Initial dataframe for mergeing
                        self.df = pd.read_csv(filename, usecols=self.columns_to_use, )
                        print(self.df.columns)
                        continue

                    #Load new df and merge it to the main one
                    new_df = pd.read_csv(filename, usecols=self.columns_to_use)
                    self.df = pd.concat([self.df, new_df])

    def prepare_data(self) -> None:
        # drop other states than ['successful', 'failed']
        self.df = self.df[self.df['state'].isin(['successful', 'failed'])]
        self.df['state'] = self.df['state'].replace({'failed': 0, 'successful': 1})

        self.df['staff_pick'] = self.df['staff_pick'].replace({False: 0, True: 1})
        # Prepare datetime data
        self.df['deadline'] = pd.to_datetime(self.df['deadline'], unit='s')

        self.df['launched_at'] = pd.to_datetime(self.df['launched_at'], unit='s')
        self.df['launched_month'] = pd.to_datetime(self.df['launched_at'], unit='s').dt.month

        self.df['launch_duration'] = (self.df['deadline'] - self.df['launched_at']).dt.days

        self.df['created_at'] = pd.to_datetime(self.df['created_at'], unit='s')
        self.df['created_duration'] = (self.df['deadline'] - self.df['created_at']).dt.days
        #dropping unusefull columns
        self.df.drop('deadline', axis=1, inplace=True)
        self.df.drop('launched_at', axis=1, inplace=True)
        self.df.drop('created_at', axis=1, inplace=True)
        
        self.df['name_word_len'] = self.df['name'].str.split().str.len()
        #self.df['name_char_len'] = self.df['name'].str.len()
        self.df.drop('name', axis=1, inplace=True)

        self.df['blurb_word_len'] = self.df['blurb'].str.split().str.len()
        #self.df['blurb_char_len'] = self.df['blurb'].str.len()
        self.df.drop('blurb', axis=1, inplace=True)

        #self.df['pledge_per_backer'] = round(self.df['usd_pledged'] / self.df['backers_count'], 2)

        # Extracting the relevant category section from the string, and replacing the original category variable
        f = lambda x: x['category'].split('"slug":"')[1].split('/')[0]
        self.df['category'] = self.df.apply(f, axis=1)
        f = lambda x: x['category'].split('","position"')[
            0]  # Some categories do not have a sub-category, so do not have a '/' to split with
        self.df['category'] = self.df.apply(f, axis=1)

        # Convert categorical data
        self.df_prepared: pd.DataFrame = pd.get_dummies(self.df, drop_first=True, dtype="float")
        # print(self.df.loc[1])
        # print(self.df_prepared.loc[1])
        # Convert NaN to 0
        #print(self.df_prepared.isna().sum())  # two blurb_word_len is NaN
        self.df_prepared.blurb_word_len.fillna(0, inplace=True)

        # Dropping columns and creating new dataframe
        self.df_transformed = self.df_prepared.drop(['backers_count', 'usd_pledged'], axis=1)
        self.df_transformed.head()
        ## Dropping columns beginning with 'country'
        self.df_transformed = self.df_transformed[[c for c in self.df_transformed if c[:7] != 'country']]
        self.df_transformed.head()

        self.df_transformed['state'] = self.df_transformed['state'].replace({'failed': 0, 'successful': 1})
        self.df_transformed['staff_pick'] = self.df_transformed['staff_pick'].astype(str)
        self.df_transformed = pd.get_dummies(self.df_transformed)

        self.X_unscaled = self.df_transformed.drop('state', axis=1)
        self.y = self.df_transformed.state

        # Transforming the data
        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X_unscaled), columns=list(self.X_unscaled.columns))
        self.X.head()
        #
        # print("\nBefore get_dummies:")
        # print(self.df.head())
        #
        # print("\nAfter get_dummies:")
        # print(self.df_prepared)

    def plot_cf(y_true, y_pred, class_names=None, model_name=None):
        """Plots a confusion matrix"""
        cf = confusion_matrix(y_true, y_pred)
        plt.imshow(cf, cmap=plt.cm.Blues)
        plt.grid(b=None)
        if model_name:
            plt.title("Confusion Matrix: {}".format(model_name))
        else:
            plt.title("Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
            
        class_names = set(y_true)
        tick_marks = np.arange(len(class_names))
        if class_names:
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
            
        thresh = cf.max() / 2.
        
        for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
            plt.text(j, i, cf[i, j], horizontalalignment='center', color='white' if cf[i, j] > thresh else 'black')
        plt.colorbar()

    def PCA(self) -> None:
        # Splitting into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, test_size=0.3, random_state=123)
        
        pca = PCA()
        principal_comps = pca.fit_transform(self.X)
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        
        # Creating a list of PCA column names
        pca_columns = []
        for i in range(1,23):
            pca_columns.append("PC"+str(i))

        #PCA graph
        principal_comps_df = pd.DataFrame(principal_comps, columns=pca_columns)

        # Adding target (success/fail) to the principal components dataframe
        principal_comps_df = pd.concat([principal_comps_df, self.y.reset_index()], axis=1)
        #principal_comps_df.drop('id', inplace=True, axis=1)
        principal_comps_df.head()

        # Plotting the first two principal components, coloured by target
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=principal_comps_df.PC1, y=principal_comps_df.PC2, data=principal_comps_df, hue='state')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.show()

        loadings = pca.components_[1]  # Współczynniki zaangażowania dla PCA1
        sorted_loadings = sorted(zip(self.X.columns, loadings), key=lambda x: abs(x[1]), reverse=True)

        for column, loading in sorted_loadings:
            print(f"Cecha: {column}, Współczynnik zaangażowania: {abs(loading)}")
        

        pca_loadings = pca.components_[1]  # Coefficients for PCA1
        loadings_df = pd.DataFrame({'Variable': self.X.columns, 'Loading': pca_loadings})
        loadings_df = loadings_df.reindex(loadings_df['Loading'].abs().sort_values(ascending=False).index)  # Sort by absolute loading values
        top_variables = loadings_df.head(10)
        print("Top variables contributing to PCA2:")
        print(top_variables)

        # Plotting the amount of variation explained by PCA with different numbers of components
        plt.plot(list(range(1, len(explained_var)+1)), explained_var)
        plt.title('Amount of variation explained by PCA', fontsize=14)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.show()



        n_comps = [20]
        for n in n_comps:
            pipe = Pipeline([('pca', PCA(n_components=n)), ('clf', LogisticRegression())])
            pipe.fit(X_train, y_train)
            print("\nNumber of components:", n)
            print("Score:", round(pipe.score(X_test, y_test),5))

        #logistic regresion best params
        # Using GridSearchCV to test multiple different parameters
        logreg_start = time.time()

        pipe_logreg = Pipeline([('pca', PCA(n_components=20)),
                            ('clf', LogisticRegression())])

        params_logreg = [
            {'clf__penalty': ['l1', 'l2'],
            'clf__fit_intercept': [True, False],
                'clf__C': [0.001, 0.01, 1, 10]
            }
        ]

        grid_logreg = GridSearchCV(estimator=pipe_logreg,
                        param_grid=params_logreg,
                        cv=5)

        grid_logreg.fit(X_train, y_train)

        logreg_end = time.time()

        logreg_best_score = grid_logreg.best_score_
        logreg_best_params = grid_logreg.best_params_

        print(f"Time taken to run: {round((logreg_end - logreg_start)/60,1)} minutes")
        print("Best accuracy:", round(logreg_best_score,2))
        print("Best parameters:", logreg_best_params)
        
        
        #best logistic reg model
        pipe_best_logreg = Pipeline([('pca', PCA(n_components=20)),
                    ('clf', LogisticRegression(C=10, fit_intercept=False, penalty='l2'))])

        clf = pipe_best_logreg.fit(X_train, y_train)

        lr_y_hat_train = pipe_best_logreg.predict(X_train)
        lr_y_hat_test = pipe_best_logreg.predict(X_test)

        print("Logistic regression score for training set:", round(pipe_best_logreg.score(X_train, y_train),5))
        print("Logistic regression score for test set:", round(pipe_best_logreg.score(X_test, y_test),5))
        print("\nClassification report:")
        print(classification_report(y_test, lr_y_hat_test))
        print(self.X.columns)

        self.score_dfPCA: pd.DataFrame = pd.DataFrame()

        for col in self.X.columns[:]:
            self.PCAindicators(np.array(self.X[col]).reshape(-1, 1), np.array(self.y), col_label = col)

    def UMAP(self) -> None:
        #X = self.df_prepared
        reducer = umap.UMAP(n_components=2)
        X_umap = reducer.fit_transform(self.X)
        umap_df = pd.DataFrame(data=X_umap, columns=['UMAP1', 'UMAP2'])
        plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], hue='state')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title('UMAP Projection')
        plt.show()

        
        #X_train, X_test, y_train, y_test = train_test_split(umap_df, self.y, test_size=0.3, random_state=123)
        
        #X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, test_size=0.3, random_state=123)
        #model = LogisticRegression()
        #model.fit(X_train, y_train)
        #y_pred = model.predict(X_test)

    def PCAindicators(self, X: np.ndarray, y: np.ndarray, use_scaler: bool = True, col_label: str = "", plot_conf_matrix: bool = False) -> None:
        def sensitivity(clf, X, y) -> float:
            # [trueNegative,falsePositive, falseNegative, truePositive]
            y_pred = clf.predict(X)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            # Used to remove dividing by 0 warnings
            if (tp+fn) == 0:
                return np.nan

            return tp / (tp+fn)

        def specificity(clf, X, y) -> float:
            # [trueNegative,falsePositive, falseNegative, truePositive]
            y_pred = clf.predict(X)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            # Used to remove dividing by 0 warnings
            if (tn + fp) == 0:
                return np.nan

            return tn / (tn + fp)

        def PPV(clf, X, y) -> float:
            # [trueNegative,falsePositive, falseNegative, truePositive]
            y_pred = clf.predict(X)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            # Used to remove dividing by 0 warnings
            if (tp + fp) == 0:
                return np.nan

            return tp / (tp + fp)

        def NPV(clf, X, y) -> float:
            # [trueNegative,falsePositive, falseNegative, truePositive]
            y_pred = clf.predict(X)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            # Used to remove dividing by 0 warnings
            if (tn + fn) == 0:
                return np.nan

            return tn / (tn + fn)

        def calculate_bic2(clf, X, y, num_params = 2) -> float:
            # Sources:
            # https://stackoverflow.com/questions/48185090/how-to-get-the-log-likelihood-for-a-logistic-regression-model-in-sklearn
            # https://en.wikipedia.org/wiki/Bayesian_information_criterion

            n = len(y)
            y_pred = clf.predict(X)
            log_likelihood = -log_loss(y, y_pred)

            bic = -2 * log_likelihood+ num_params * np.log(n)
            return bic
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=123)

        scaler = StandardScaler()
        logreg = LogisticRegression(C=10, fit_intercept=True, penalty='l2', class_weight = "balanced")
        if use_scaler:
            pipeline:Pipeline = Pipeline([('scaler', scaler), ('logreg', logreg )])
        else:
            pipeline:Pipeline = Pipeline([('logreg', logreg)])


        # use optional scaler and fit train data
        clf = logreg.fit(X_train, y_train)
        print(clf.score(X_test, y_test))
        #Plot conf matrix
        if plot_conf_matrix:
            ConfusionMatrixDisplay.from_estimator(
                clf,
                X_test,
                y_test,
                normalize="true"
            )
            plt.title("PCA: " + col_label)
            plt.show()

        self.scores_for_cross_validation = {
            "BIC": calculate_bic2,
            "sensitivity": sensitivity, #(also called the true positive rate/recall),
            "specificity": specificity, #(also called the true negative rate),
            "NPV": NPV, # Negative predictive value
            "PPV": PPV, # Precision or positive predictive value
            "balanced_accuracy": "balanced_accuracy",
            'f1': 'f1',
        }
        scores = cross_validate(clf, X, y, cv=15, scoring = self.scores_for_cross_validation, return_train_score = False, return_estimator=False)


        # Calculate mean from k-validations and add to score_df
        col_score = {}
        for score, k_val_arr in scores.items():
            col_score[score] = k_val_arr.mean()
            col_score[score+"_std"] = 2*k_val_arr.std()

        #print(col_score)

        if self.score_dfPCA.empty:
            self.score_dfPCA = pd.DataFrame(data = col_score, index = [col_label])
        else:
            self.score_dfPCA.loc[col_label] = col_score

    def SISO(self) -> None:
        y: pd.DataFrame = self.df_prepared['state']

        X_all: pd.DataFrame = self.df_prepared.drop('state', axis = 1)

        print(self.X.columns)

        self.score_df: pd.DataFrame = pd.DataFrame()

        for col in self.X.columns[:]:
            self.logistic_regression_SISO(np.array(self.X[col]).reshape(-1, 1), np.array(self.y), col_label = col)

    def logistic_regression_SISO(self, X: np.ndarray, y: np.ndarray, use_scaler: bool = True, col_label: str = "", plot_conf_matrix: bool = False) -> None:
        def sensitivity(clf, X, y) -> float:
            # [trueNegative,falsePositive, falseNegative, truePositive]
            y_pred = clf.predict(X)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            # Used to remove dividing by 0 warnings
            if (tp+fn) == 0:
                return np.nan

            return tp / (tp+fn)

        def specificity(clf, X, y) -> float:
            # [trueNegative,falsePositive, falseNegative, truePositive]
            y_pred = clf.predict(X)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            # Used to remove dividing by 0 warnings
            if (tn + fp) == 0:
                return np.nan

            return tn / (tn + fp)

        def PPV(clf, X, y) -> float:
            # [trueNegative,falsePositive, falseNegative, truePositive]
            y_pred = clf.predict(X)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            # Used to remove dividing by 0 warnings
            if (tp + fp) == 0:
                return np.nan

            return tp / (tp + fp)

        def NPV(clf, X, y) -> float:
            # [trueNegative,falsePositive, falseNegative, truePositive]
            y_pred = clf.predict(X)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            # Used to remove dividing by 0 warnings
            if (tn + fn) == 0:
                return np.nan

            return tn / (tn + fn)

        def calculate_bic2(clf, X, y, num_params = 2) -> float:
            # Sources:
            # https://stackoverflow.com/questions/48185090/how-to-get-the-log-likelihood-for-a-logistic-regression-model-in-sklearn
            # https://en.wikipedia.org/wiki/Bayesian_information_criterion

            n = len(y)
            y_pred = clf.predict(X)
            log_likelihood = -log_loss(y, y_pred)

            bic = -2 * log_likelihood+ num_params * np.log(n)
            return bic

        # X_test, y_test -> hold-out set
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=123)

        scaler = StandardScaler()
        logreg = LogisticRegression(C=10, fit_intercept=True, penalty='l2', class_weight = "balanced")
        if use_scaler:
            pipeline:Pipeline = Pipeline([('scaler', scaler), ('logreg', logreg )])
        else:
            pipeline:Pipeline = Pipeline([('logreg', logreg)])
        
        pipe_best_logreg = Pipeline([('pca', PCA(n_components=1)),
                    ('clf', LogisticRegression(C=10, fit_intercept=False, penalty='l2'))])

        pipe_best_logreg.fit(X_train, y_train)

        lr_y_hat_train = pipe_best_logreg.predict(X_train)
        lr_y_hat_test = pipe_best_logreg.predict(X_test)


        # use optional scaler and fit train data
        clf = pipe_best_logreg.fit(X_train, y_train)
        print(clf.score(X_test, y_test))
        #Plot conf matrix
        if plot_conf_matrix:
            ConfusionMatrixDisplay.from_estimator(
                clf,
                X_test,
                y_test,
                normalize="true"
            )
            plt.title("PCA: " + col_label)
            plt.show()

        self.scores_for_cross_validation = {
            "BIC": calculate_bic2,
            "sensitivity": sensitivity, #(also called the true positive rate/recall),
            "specificity": specificity, #(also called the true negative rate),
            "NPV": NPV, # Negative predictive value
            "PPV": PPV, # Precision or positive predictive value
            "balanced_accuracy": "balanced_accuracy",
            'f1': 'f1',
        }
        scores = cross_validate(clf, X, y, cv=15, scoring = self.scores_for_cross_validation, return_train_score = False, return_estimator=False)


        # Calculate mean from k-validations and add to score_df
        col_score = {}
        for score, k_val_arr in scores.items():
            col_score[score] = k_val_arr.mean()
            col_score[score+"_std"] = 2*k_val_arr.std()

        #print(col_score)

        if self.score_df.empty:
            self.score_df = pd.DataFrame(data = col_score, index = [col_label])
        else:
            self.score_df.loc[col_label] = col_score


        #print(self.score_df)
        # print(scores)

    def prepare_plots(self) -> None:

        print("\nScore DataFrame columns: ")
        print(self.score_df.columns)

        params_to_plot = ["test_BIC", "test_sensitivity", "test_specificity", "test_PPV", "test_NPV", "test_balanced_accuracy", "test_f1"]
        for param in params_to_plot:
            asc_bool = False
            if param == "test_BIC":
                asc_bool = True

            plt.figure(figsize=(16, 11))
            self.score_df[param].sort_values(ascending=asc_bool).plot(kind="bar")
            bar_x = self.score_df[param].sort_values(ascending=asc_bool).index
            bar_y = self.score_df[param].loc[bar_x]
            bar_err = self.score_df[param+"_std"].loc[bar_x]

            if param != "test_BIC":
                bar_err[(bar_y+bar_err)>1.0] = np.clip(bar_y+bar_err, 0.0, 1.0) - bar_y
                bar_err[(bar_y-bar_err)<0.0] = np.clip(bar_y-bar_err, 0.0, 1.0) + bar_y

            plt.errorbar(bar_x, bar_y, yerr=bar_err, fmt = 'o',color = 'black',
            ecolor = 'black', elinewidth = 2, capsize=10, capthick = 2)
            plt.title(param+" Balanced")
            plt.tight_layout()
            plt.show()

    def prepare_plotsPCA(self) -> None:

        print("\nScore DataFrame columns: ")
        print(self.score_dfPCA.columns)

        params_to_plot = ["test_BIC", "test_sensitivity", "test_specificity", "test_PPV", "test_NPV", "test_balanced_accuracy", "test_f1"]
        for param in params_to_plot:
            asc_bool = False
            if param == "test_BIC":
                asc_bool = True

            plt.figure(figsize=(16, 11))
            self.score_dfPCA[param].sort_values(ascending=asc_bool).plot(kind="bar")
            bar_x = self.score_dfPCA[param].sort_values(ascending=asc_bool).index
            bar_y = self.score_dfPCA[param].loc[bar_x]
            bar_err = self.score_dfPCA[param+"_std"].loc[bar_x]

            if param != "test_BIC":
                bar_err[(bar_y+bar_err)>1.0] = np.clip(bar_y+bar_err, 0.0, 1.0) - bar_y
                bar_err[(bar_y-bar_err)<0.0] = np.clip(bar_y-bar_err, 0.0, 1.0) + bar_y

            plt.errorbar(bar_x, bar_y, yerr=bar_err, fmt = 'o',color = 'black',
            ecolor = 'black', elinewidth = 2, capsize=10, capthick = 2)
            plt.title(param+" Balanced")
            plt.tight_layout()
            plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predictor = KickstartedPredict(
        data_folder_path=r"C:\Users\kbklo\Desktop\Studia\_INFS2\CVaPR\Projekt\Data",
        num_of_files_to_load = 5,

    )
    predictor.run()