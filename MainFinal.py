import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys
from umap import UMAP
import json
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from collections import OrderedDict
# import seaborn as sns

import Scores


from Searchers import *


# import addcopyfighandler # enables ctrl + c -> save matplotlib figure to clipboard

plt.rcParams.update({'font.size': 20})
"""
ETAP I:     https://docs.google.com/document/d/13UgWdYbAnr3SVn1Ssj6Y_KiKmPix4tuc-oK1JD2tehs/edit?usp=sharing
ETAP II:    https://docs.google.com/document/d/1yyEoZOCA6FFDqls-rtYKk19PXqpdUIFHM7cWjWJgR18/edit?usp=sharing
ETAP III:   https://docs.google.com/document/d/1oJiQtZmKK2RHHHmcHS1ksXMz2Zn4dayVmIStPRM2MtU/edit?usp=sharing
"""



# plt.style.use('seaborn')

class KickstartedPredict():
    def __init__(self, data_folder_path: str, num_of_files_to_load: int = 1) -> None:
        self.random_state = 123
        pd.options.display.max_columns = 9999
        self.data_folder_path: str = data_folder_path
        self.num_of_files_to_load: int = num_of_files_to_load
        self.df: pd.Dataframe = pd.DataFrame()
        self.columns_to_use: List[str] = ["backers_count", "blurb", "category", "country",
                                          "created_at", "deadline", "goal", "launched_at",
                                          "name", "staff_pick", "state", "usd_pledged"]

    def run(self) -> None:
        self.load_data()
        self.prepare_data()
        # self.PCA()
        # self.SISO()
        # self.prepare_plotsPCA()
        self.hyper_paramether_tuning()
        # y: pd.DataFrame = self.df_prepared['state']
        # X_all: pd.DataFrame = self.df_prepared.drop('state', axis=1)
        # self.plot_umap_data_transform(X_all, y)
        # self.score_df = pd.read_pickle(r"C:\Users\kbklo\Documents\GitHub\cvapr\Outputs\DIFF_EVO_12_06_2023_20_21_44.pkl")
        # print(self.score_df)
        # self.score_df = pd.read_csv(".\GRID.csv", sep = ";")
        # print(self.score_df.test_balanced_accuracy.max())


    def load_data(self) -> None:
        """Load data to self.df dataframe. Param use_columns==None means all columns are used."""

        for i, filename in enumerate(os.scandir(Path(self.data_folder_path))):
            if filename.name.endswith('.csv'):
                i += 1
                if i > self.num_of_files_to_load:
                    break

                if filename.is_file():
                    if i == 1:
                        # Initial dataframe for mergeing
                        self.df = pd.read_csv(filename, usecols=self.columns_to_use, )
                        print(self.df.columns)
                        continue

                    # Load new df and merge it to the main one
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
        # dropping unusefull columns
        self.df.drop('deadline', axis=1, inplace=True)
        self.df.drop('launched_at', axis=1, inplace=True)
        self.df.drop('created_at', axis=1, inplace=True)
        self.df.drop('backers_count', axis=1, inplace=True)
        self.df.drop('usd_pledged', axis=1, inplace=True)

        self.df['name_word_len'] = self.df['name'].str.split().str.len()
        # self.df['name_char_len'] = self.df['name'].str.len()
        self.df.drop('name', axis=1, inplace=True)

        self.df['blurb_word_len'] = self.df['blurb'].str.split().str.len()
        # self.df['blurb_char_len'] = self.df['blurb'].str.len()
        self.df.drop('blurb', axis=1, inplace=True)

        # self.df['pledge_per_backer'] = round(self.df['usd_pledged'] / self.df['backers_count'], 2)

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
        # print(self.df_prepared.isna().sum())  # two blurb_word_len is NaN
        self.df_prepared.blurb_word_len.fillna(0, inplace=True)




        print(self.df_prepared.columns)
        #
        # print("\nBefore get_dummies:")
        # print(self.df.head())
        #
        # print("\nAfter get_dummies:")
        # print(self.df_prepared)

    def hyper_paramether_tuning(self, normalize_data=True):

        print("init paramethers")
        y: pd.DataFrame = self.df_prepared['state']
        X_all: pd.DataFrame = self.df_prepared.drop('state', axis=1)

        if normalize_data:
            for col in self.df_prepared.columns:
                X_all = MinMaxScaler().fit_transform(X_all)

        # Example for CustomGridSearch
        params = OrderedDict({
            "scalers": ["QuantileTransformer()"],
            "PCA": {
                "n_components": [2, 10, 20, 40]
            },
            "UMAP": {
                "n_components": [2, 5, 10],
                "n_neighbors": [5, 20, 50],
                "min_dist": [0.1, 0.5, 1.0]
            },
            "LogisticRegression": {
                "class_weight": [None, 'balanced'],
                "C": [0.1, 1.0, 100, 1000]
            }
        })
        searcher = CustomGridSearch(random_state = self.random_state)
        self.score: pd.DataFrame = searcher.evaluate_umap_transform_all_cv(X_all, y, params, cross_validations = 5)

        date_string = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.score.to_csv(".\Outputs\GRID_umap_%s.csv"%date_string, sep=";")
        self.score.to_pickle(".\Outputs\RID_umap_%s.pkl" % date_string)
        return

        # params = OrderedDict({
        #     "scalers": ["StandardScaler()"],
        #     "PCA": {
        #         "n_components": ["int", 2, 10]
        #     },
        #     "UMAP": {
        #         # "n_components": ["int", 2, 10]
        #         "n_neighbors": ["int", 2, 5]
        #     },
        #     "LogisticRegressionCV": {
        #         # "class_weight": ["categorical", None, 'balanced'],
        #         "C": ["float", 0.1, 1000]
        #     }
        # })
        #
        # searcher = DifferentialEvolution(random_state = self.random_state, score_metric="BIC")
        # self.score: pd.DataFrame = searcher.evaluate(X_all, y, params,
        #                                              cross_validations=2,
        #                                              popsize=3,
        #                                              max_iters=1)
        # #Dump score to file
        # date_string = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        # self.score.to_pickle(".\Outputs\DIFF_EVO_%s.pkl"%date_string)

        # with open(r"%s\Outputs\DIFF_EVO_%s.pickle"%(os.getcwd(), date_string), "wb") as output_file:
        #     pickle.dump(self.score, output_file)

    def plot_umap_data_transform(self, X, y, n_neighbors=15, min_dist=0.1, n_components=3, metric='euclidean'):
        fig = plt.figure()
        fit = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state = self.random_state
        )
        u = fit.fit_transform(X, y)

        if n_components == 1:
            ax = fig.add_subplot(111)
            ax.scatter(u[:, 0], range(len(u)))
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(u[:, 0], u[:, 1])
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=100)

        print(cross_validate(LogisticRegression(random_state = self.random_state), u, y, scoring=Scores.scores, cv = 5))
        plt.show()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predictor = KickstartedPredict(
        data_folder_path=r"%s\Data"%os.getcwd(),
        num_of_files_to_load=10,
    )
    predictor.run()
