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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import time
from datetime import timedelta
import itertools
from collections import OrderedDict
import seaborn as sns


import Scores
# import addcopyfighandler # enables ctrl + c -> save matplotlib figure to clipboard

plt.rcParams.update({'font.size': 20})
"""
ETAP I:     https://docs.google.com/document/d/13UgWdYbAnr3SVn1Ssj6Y_KiKmPix4tuc-oK1JD2tehs/edit?usp=sharing
ETAP II:    https://docs.google.com/document/d/1yyEoZOCA6FFDqls-rtYKk19PXqpdUIFHM7cWjWJgR18/edit?usp=sharing
ETAP III:   https://docs.google.com/document/d/1oJiQtZmKK2RHHHmcHS1ksXMz2Zn4dayVmIStPRM2MtU/edit?usp=sharing
"""


def print_with_border(string:str, space:int = 3):
    print("#"*(len(string)+(2+2*space)))
    for i in range(space-1):
        print("#%s#"%(" "*(len(string)+2*space)))

    print("#%s%s%s#"%(" "*space ,string, " "*space))

    for i in range(space-1):
        print("#%s#"%(" "*(len(string)+2*space)))
    print("#"*(len(string)+(2+2*space)))

# plt.style.use('seaborn')

class KickstartedPredict():
    def __init__(self, data_folder_path: str, num_of_files_to_load: int = 1) -> None:

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

    def hyper_paramether_tuning(self):

        print("init paramethers")
        y: pd.DataFrame = self.df_prepared['state']
        X_all: pd.DataFrame = self.df_prepared.drop('state', axis=1)

        # params = {
        #     "scalers": ["StandardScaler()"],
        #     "PCAs": ["PCA(n_components=5)", "PCA(n_components=10)"],
        #     "UMAPs": ["umap.UMAP(n_components=2)"],
        #     "LogRegs": ["LogisticRegression()", "LogisticRegression(class_weight='balanced')"]
        # }
        x_t =  [None, 'balanced'],
        y_t = [0.1, 1, 100, 1000]
        print(np.meshgrid(x_t, y_t))

        params = OrderedDict({
            "scalers": ["StandardScaler()"],
            "PCA": {
                "n_components": [5, 10, 15]
            },
            "UMAP": {
                "n_components": [2, 5]
            },
            "LogisticRegression": {
                "class_weight": [None, 'balanced'],
                "C": [0.1, 1, 100, 1000]
            }
        })

        self.score = self.custom_grid_search(X_all, y, params)

    def custom_grid_search(self, X, y, params: Dict, cross_validations = 3, max_iter = 1000) -> pd.DataFrame:
        def create_lists(paramethers: Dict, return_strings: bool = False) -> Dict:
            """
            Creates list of classes specified in pramethers.\n
            intput: Parameters = { "a": [1, 2, 3]}\n
            will give output: [a(1), a(2), a(3)]

            :param paramethers: Keys can be given in two ways:
                1. custom_name: [Class(param=1), Class(param=2)]
                2. Class: {param: [1, 2]}
            :param return_strings
            :return: Dictionary with key name same as in paramethers.
            Values are the list classes
            """
            ret_list = {}
            ret_list_strings = {}
            for key in paramethers:
                # eval(str) - creates object from string
                obj = paramethers[key]
                if isinstance(obj, List):
                    ret_list[key] = [eval(x) for x in obj]
                    ret_list_strings[key] = obj
                elif isinstance(obj, Dict):
                    param_grid = list(itertools.product(*obj.values()))
                    obj_list = []
                    obj_list_strings = []
                    for cord in param_grid:
                        temp = "%s("%key
                        for p_i, param in enumerate(obj):
                            value = cord[p_i]
                            if isinstance(value, str):
                                value = "'"+value+"'"

                            temp += "%s=%s," %(str(param), value)
                        temp += ")"
                        print(temp)
                        obj_list.append(eval(temp))
                        obj_list_strings.append(temp)
                    ret_list[key] = obj_list
                    ret_list_strings[key] = obj_list_strings

            print(ret_list)
            if return_strings:
                return [ret_list, ret_list_strings]
            else:
                return ret_list

        start_time = time.monotonic()
        current_iter = 1

        param_grid, param_grid_strings = create_lists(params, return_strings=True)
        params_keys = list(params.keys())


        scalers = param_grid[params_keys[0]]
        PCAs = param_grid[params_keys[1]]
        UMAPs = param_grid[params_keys[2]]
        LogRegs = param_grid[params_keys[3]]

        score_df = pd.DataFrame()

        all_iters = len(scalers) * len(PCAs) * len(UMAPs) * len(LogRegs)
        print_with_border("Creating grid search with %d iterations" % all_iters)
        for s_i, scaler in enumerate(scalers):

            X_scaled = scaler.fit_transform(X, y)

            for p_i, pca in enumerate(PCAs):

                X_scaled_pca = pca.fit_transform(X_scaled, y)

                for u_i, UMAP in enumerate(UMAPs):

                    X_scaled_pca_umap = UMAP.fit_transform(X_scaled_pca, y)

                    # Splitting into train and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled_pca_umap, y, test_size=0.3,
                                                                        random_state=123)

                    for lr_i, logReg in enumerate(LogRegs):
                        reg_time = time.monotonic() - start_time
                        time_left = (all_iters - current_iter) * reg_time / current_iter
                        sys.stdout.write(f"\r %.2f %% done | Elapsed time: %s | Estimated time left: %s" % (
                            current_iter / all_iters * 100,
                            str(timedelta(seconds=reg_time)).split('.', 2)[0],  # Split is used to remove ms
                            str(timedelta(seconds=time_left)).split('.', 2)[0]  # Split is used to remove ms
                        ))
                        sys.stdout.flush()

                        sys.stdout.write(f"\r %.2f %% done | Elapsed time: %s | Estimated time left: %s" % (
                            current_iter / all_iters * 100,
                            str(timedelta(seconds=reg_time)).split('.', 2)[0],  # Split is used to remove ms
                            str(timedelta(seconds=time_left)).split('.', 2)[0]  # Split is used to remove ms
                        ))
                        sys.stdout.flush()

                        logReg.max_iter = max_iter

                        scores = cross_validate(logReg, X_scaled_pca_umap, y, cv=cross_validations, scoring = Scores.scores,
                                                return_train_score=False, return_estimator=False)



                        iter_score = {
                            "scaler": param_grid_strings[params_keys[0]][s_i],
                            "pca": param_grid_strings[params_keys[1]][p_i],
                            "umap": param_grid_strings[params_keys[2]][u_i],
                            "log": param_grid_strings[params_keys[3]][lr_i],
                        }

                        # Calculate mean from k-validations
                        for score, k_val_arr in scores.items():
                            iter_score[score] = k_val_arr.mean()
                            iter_score[score + "_std"] = 2 * k_val_arr.std()

                        if score_df.empty:
                            score_df = pd.DataFrame(data=iter_score, index=[current_iter])
                        else:
                            score_df.loc[current_iter] = iter_score
                        # print("iter %d: %.3f" %(current_iter, clf.score(X_test, y_test)))

                        current_iter += 1

        print(score_df)
        return score_df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predictor = KickstartedPredict(
        data_folder_path=r"C:\Users\kbklo\Desktop\Studia\_INFS2\CVaPR\Projekt\Data",
        num_of_files_to_load=2,

    )
    predictor.run()
