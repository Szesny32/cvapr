import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import addcopyfighandler # enables ctrl + c -> save matplotlib figure to clipboard


# plt.rcParams.update({'font.size': 22})
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
        self.SISO()
        self.prepare_plots()

    def load_data(self) -> None:
        """Load data to self.df dataframe. Param use_columns==None means all columns are used."""

        for i, filename in enumerate(os.scandir(Path(self.data_folder_path))):
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

        self.df.drop('deadline', axis=1, inplace=True)
        self.df.drop('launched_at', axis=1, inplace=True)
        self.df.drop('created_at', axis=1, inplace=True)

        # Prepare string data
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
        self.df_prepared: pd.DataFrame = pd.get_dummies(self.df, drop_first=True)

        # Convert NaN to 0
        #print(self.df_prepared.isna().sum())  # two blurb_word_len is NaN
        self.df_prepared.blurb_word_len.fillna(0, inplace=True)

        #
        # print("\nBefore get_dummies:")
        # print(self.df.head())
        #
        # print("\nAfter get_dummies:")
        # print(self.df_prepared)
    
    def SISO(self) -> None:
        y: pd.DataFrame = self.df_prepared['state']

        X_all: pd.DataFrame = self.df_prepared.drop('state', axis = 1)

        print(X_all.columns)

        self.score_df: pd.DataFrame = pd.DataFrame()

        for col in X_all.columns[:]:
            self.logistic_regression_SISO(np.array(X_all[col]).reshape(-1, 1), np.array(y), col_label = col)


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
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=123)

        scaler = StandardScaler()
        logreg = LogisticRegression(C=10, fit_intercept=True, penalty='l2')
        if use_scaler:
            pipeline:Pipeline = Pipeline([('scaler', scaler), ('logreg', logreg )])
        else:
            pipeline:Pipeline = Pipeline([('logreg', logreg)])


        # use optional scaler and fit train data
        clf = pipeline.fit(X_train, y_train)

        #Plot conf matrix
        if plot_conf_matrix:
            plot_confusion_matrix(clf, X, y)
            plt.title("SISO: " + col_label)
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
        scores = cross_validate(clf, X, y, cv=5, scoring = self.scores_for_cross_validation, return_train_score = False)

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

            plt.figure(figsize=(16, 9))
            self.score_df[param].sort_values(ascending=asc_bool).plot(kind="bar")
            bar_x = self.score_df[param].sort_values(ascending=asc_bool).index
            bar_y = self.score_df[param].loc[bar_x]
            bar_err = self.score_df[param+"_std"].loc[bar_x]

            if param != "test_BIC":
                bar_err[(bar_y+bar_err)>1.0] = np.clip(bar_y+bar_err, 0.0, 1.0) - bar_y
                bar_err[(bar_y-bar_err)<0.0] = np.clip(bar_y-bar_err, 0.0, 1.0) + bar_y
                
            plt.errorbar(bar_x, bar_y, yerr=bar_err, fmt = 'o',color = 'black',
            ecolor = 'black', elinewidth = 2, capsize=10, capthick = 2)
            plt.title(param)
            plt.tight_layout()
            plt.show()


#
# def main():
#     pd.options.display.max_columns = 9999
#     pd.options.display.max_rows = 10
#     # df = pd.read_csv(r"C:\Users\kbklo\Desktop\Studia\_INFS2\CVaPR\Projekt\Data\Kickstarter010.csv",
#     #                  )
#     # # print(read_df.columns)
#     # # df = read_df[columns_to_read]
#     # print(df.columns)
#
#     init_filename = r"C:\Users\kbklo\Desktop\Studia\_INFS2\CVaPR\Projekt\Data\Kickstarter.csv"
#     # used_columns = ["goal", "state", "usd_pledged", "launched_at", "spotlight","deadline"]
#     used_columns = ["state","launched_at", "deadline"]
#
#     # Init dataframe for merge
#     df = pd.read_csv(init_filename, usecols=used_columns,)
#
#
#     n_of_files = 100
#     for i, filename in enumerate(os.scandir(r"C:\Users\kbklo\Desktop\Studia\_INFS2\CVaPR\Projekt\Data")):
#         if filename.is_file():
#
#             if filename != init_filename:
#                 new_df = pd.read_csv(filename, usecols=used_columns,)
#                 df = pd.concat([df, new_df])
#
#                 i +=1
#                 if i > n_of_files:
#                     break
#
#     fig, ax = plt.subplots(1,1)
#
#
#     df['launched_at'] = pd.to_datetime(df['launched_at'], unit ='s')
#     df['deadline'] = pd.to_datetime(df['deadline'], unit ='s')
#
#     df['duration'] = (df['deadline'] - df['launched_at']).dt.days
#
#     print(df)
#     print(df.groupby('duration'))
#     success_rate = df.groupby('duration').state.count().plot(kind = 'bar', rot = 0)
#     print(df['duration'].max())
#
#     xtick = np.char.mod('%d', range(0 , 105))
#     xtick[:] = ""
#     xtick[::20] = np.char.mod('%d', range(0 ,105, 20))
#
#     plt.xticks(range(0 , 105),xtick )
#     plt.ylabel("Amount of Projects")
#     plt.xlabel("Campaign Duration (Days)")
#     plt.title("Amount of Projects by Duration")
#     plt.show()
#     print(success_rate)
#     # fraction_rate = success_rate/df.groupby('duration').count()
#     # fraction_rate.plot(kind = 'bar')
#     plt.title("Median usd_pledged value by state")
#     plt.ylabel("Median usd_pledged value")
#     plt.show()
#     df = df[df['state'].isin(['successful', 'failed'])] # drop other states than ['successful', 'failed']
#     # df.groupby('state').goal.median().plot(kind='bar', rot=0) #, color=['darkred', 'green']
#     # annotate
#     print(df.groupby('state').describe().astype(int))
#
#     print(np.array(df['goal']))
#
#     hist_max_goal = 1E5
#     s_hist_data = np.array(df['goal'][df['state']=="successful"])
#     s_hist_data = np.clip(s_hist_data, 0.0, hist_max_goal)
#     f_hist_data = np.array(df['goal'][df['state'] == "failed"])
#     f_hist_data = np.clip(f_hist_data, 0.0, hist_max_goal)
#     s_w = np.empty(s_hist_data.shape)
#     s_w.fill(1 / s_hist_data.shape[0])
#     f_w = np.empty(f_hist_data.shape)
#     f_w.fill(1 / f_hist_data.shape[0])
#
#
#     hist_data = ax.hist([s_hist_data, f_hist_data], bins =10, weights = [s_w, f_w], range = (0,hist_max_goal), align = 'mid', color = ["green", "darkred"])
#     bin_edges = hist_data[1]
#     bin_edges_labels = np.copy(bin_edges)
#     bin_edges_labels = np.char.mod('%d', bin_edges_labels)
#     for bin_i in range(len(bin_edges_labels)):
#         bin_edges_labels[bin_i] = bin_edges_labels[bin_i][:2] + " " + bin_edges_labels[bin_i][2:]
#     bin_edges_labels[-1] = ""
#     plt.xticks(bin_edges, bin_edges_labels)
#     plt.xlabel("Goal ($)")
#     plt.ylabel("Fraction of projects")
#
#     print(1E6)
#     # plt.yscale("log")
#     # plt.bar_label(ax.containers[0], label_type='edge')
#     plt.legend(["successful", "failure"])
#     plt.title("Project goal distribution")
#     plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predictor = KickstartedPredict(
        data_folder_path=r"C:\Users\kbklo\Desktop\Studia\_INFS2\CVaPR\Projekt\Data",
        num_of_files_to_load = 50,
    )
    predictor.run()
