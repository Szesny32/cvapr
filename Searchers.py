import pandas as pd
import numpy as np
import time
import sys
import itertools

from collections import OrderedDict
from datetime import timedelta
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from scipy.optimize import differential_evolution, Bounds
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.decomposition import *
from umap import UMAP
import pickle
import Scores


def print_with_border(string: str, space: int = 3):
    print("#" * (len(string) + (2 + 2 * space)))
    for i in range(space - 1):
        print("#%s#" % (" " * (len(string) + 2 * space)))

    print("#%s%s%s#" % (" " * space, string, " " * space))

    for i in range(space - 1):
        print("#%s#" % (" " * (len(string) + 2 * space)))
    print("#" * (len(string) + (2 + 2 * space)))


def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))


class CustomGridSearch():
    def __init__(self, random_state=123):
        self.random_state = random_state

    def create_lists(self, paramethers: Dict, return_strings: bool = False) -> Dict:
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
        ret_dict = {}
        ret_dict_strings = {}
        for key in paramethers:
            # eval(str) - creates object from string
            obj = paramethers[key]
            if isinstance(obj, List):
                ret_dict[key] = [eval(x) for x in obj]
                ret_dict_strings[key] = obj
            elif isinstance(obj, Dict):
                param_grid = list(itertools.product(*obj.values()))
                obj_list = []
                obj_list_strings = []
                for cord in param_grid:
                    temp = "%s(" % key
                    for p_i, param in enumerate(obj):
                        value = cord[p_i]
                        if isinstance(value, str):
                            value = "'" + value + "'"

                        temp += "%s=%s," % (str(param), value)
                    temp += ")"
                    print(temp)
                    obj_list.append(eval(temp))
                    obj_list_strings.append(temp)
                ret_dict[key] = obj_list
                ret_dict_strings[key] = obj_list_strings

        print(ret_dict)
        if return_strings:
            return [ret_dict, ret_dict_strings]
        else:
            return ret_dict

    def evaluate(self, X, y, params: Dict, cross_validations=3, max_iter=1000, kfold_on_all=False) -> pd.DataFrame:
        if kfold_on_all:
            return self.evaluate_with_transformers_cv(X, y, params, cross_validations, max_iter)
        else:
            return self.evaluate_without_transformers_cv(X, y, params, cross_validations, max_iter)

    def evaluate_with_transformers_cv(self, X, y, params: Dict, cross_validations=1, max_iter=1000):
        start_time = time.monotonic()
        current_iter = 1

        param_grid, param_grid_strings = self.create_lists(params, return_strings=True)
        params_keys = list(params.keys())

        scalers = param_grid[params_keys[0]]
        PCAs = param_grid[params_keys[1]]
        UMAPs = param_grid[params_keys[2]]
        LogRegs = param_grid[params_keys[3]]

        score_df = pd.DataFrame()

        all_iters = len(scalers) * len(PCAs) * len(UMAPs) * len(LogRegs)
        print_with_border("Creating grid search with %d iterations" % all_iters)
        for indices, objects in enumerated_product(scalers, PCAs, UMAPs, LogRegs):
            s_i, p_i, u_i, lr_i = indices
            scaler, pca, UMAP, logReg = objects

            pca.random_state = self.random_state
            UMAP.random_state = self.random_state
            logReg.random_state = self.random_state

            reg_time = time.monotonic() - start_time
            time_left = (all_iters - current_iter) * reg_time / current_iter
            sys.stdout.write(f"\r %.2f %% done | Elapsed time: %s | Estimated time left: %s" % (
                (current_iter - 1) / all_iters * 100,
                str(timedelta(seconds=reg_time)).split('.', 2)[0],  # Split is used to remove ms
                str(timedelta(seconds=time_left)).split('.', 2)[0]  # Split is used to remove ms
            ))
            sys.stdout.flush()

            pipeline = Pipeline([
                ("scaler", scaler),
                ("pca", pca),
                ("umap", UMAP),
                ("logreg", logReg)
            ])

            scores = cross_validate(pipeline, X, y, cv=cross_validations,
                                    scoring=Scores.scores,
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

    def evaluate_umap_transform_all_cv(self, X, y, params: Dict, cross_validations=5, max_iter=1000):
        start_time = time.monotonic()
        current_iter = 1

        param_grid, param_grid_strings = self.create_lists(params, return_strings=True)
        params_keys = list(params.keys())

        scalers = param_grid[params_keys[0]]
        PCAs = param_grid[params_keys[1]]
        UMAPs = param_grid[params_keys[2]]
        LogRegs = param_grid[params_keys[3]]

        score_df = pd.DataFrame()

        all_iters = len(scalers) * len(PCAs) * len(UMAPs) * len(LogRegs)
        print_with_border("Creating grid search with %d iterations" % all_iters)

        # Splitting into train and test sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
        #                                                     random_state=self.random_state)

        for s_i, scaler in enumerate(scalers):
            X_scaled = scaler.fit_transform(X, y)

            for p_i, pca in enumerate(PCAs):
                pca.random_state = self.random_state
                X_scaled_pca = pca.fit_transform(X_scaled, y)

                for u_i, UMAP in enumerate(UMAPs):
                    UMAP.random_state = self.random_state
                    X_scaled_pca_umap = UMAP.fit_transform(X_scaled_pca, y)

                    for lr_i, logReg in enumerate(LogRegs):
                        reg_time = time.monotonic() - start_time
                        time_left = (all_iters - current_iter) * reg_time / current_iter
                        sys.stdout.write(f"\r %.2f %% done | Elapsed time: %s | Estimated time left: %s" % (
                            (current_iter - 1) / all_iters * 100,
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
                        logReg.random_state = self.random_state

                        scores = cross_validate(logReg, X_scaled_pca_umap, y, cv=cross_validations,
                                                scoring=Scores.scores)

                        iter_score = {
                            "scaler": param_grid_strings[params_keys[0]][s_i],
                            "scaler_object": scaler,
                            "pca": param_grid_strings[params_keys[1]][p_i],
                            "pca_object": pca,
                            "umap": param_grid_strings[params_keys[2]][u_i],
                            "umap_object": UMAP,
                            "log": param_grid_strings[params_keys[3]][lr_i],
                            "log_object": logReg,
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

    def evaluate_without_transformers_cv(self, X, y, params: Dict, cross_validations=3, max_iter=1000):
        start_time = time.monotonic()
        current_iter = 1

        param_grid, param_grid_strings = self.create_lists(params, return_strings=True)
        params_keys = list(params.keys())

        scalers = param_grid[params_keys[0]]
        PCAs = param_grid[params_keys[1]]
        UMAPs = param_grid[params_keys[2]]
        LogRegs = param_grid[params_keys[3]]

        score_df = pd.DataFrame()

        all_iters = len(scalers) * len(PCAs) * len(UMAPs) * len(LogRegs)
        print_with_border("Creating grid search with %d iterations" % all_iters)

        # Splitting into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=self.random_state)

        for s_i, scaler in enumerate(scalers):
            X_scaled = scaler.fit_transform(X_train, y_train)
            X_test_scaler = scaler.transform(X_test)
            for p_i, pca in enumerate(PCAs):
                pca.random_state = self.random_state
                X_scaled_pca = pca.fit_transform(X_scaled, y_train)
                X_test_pca = pca.transform(X_test_scaler)
                for u_i, UMAP in enumerate(UMAPs):
                    UMAP.random_state = self.random_state
                    X_scaled_pca_umap = UMAP.fit_transform(X_scaled_pca, y_train)
                    X_test_umap = UMAP.transform(X_test_pca)
                    for lr_i, logReg in enumerate(LogRegs):
                        logReg.random_state = 123
                        reg_time = time.monotonic() - start_time
                        time_left = (all_iters - current_iter) * reg_time / current_iter
                        sys.stdout.write(f"\r %.2f %% done | Elapsed time: %s | Estimated time left: %s" % (
                            (current_iter - 1) / all_iters * 100,
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
                        logReg.fit(X_scaled_pca_umap, y_train)

                        scores = Scores.get_scores(logReg, X_test_umap, y_test)

                        iter_score = {
                            "scaler": param_grid_strings[params_keys[0]][s_i],
                            "scaler_object": scaler,
                            "pca": param_grid_strings[params_keys[1]][p_i],
                            "pca_object": pca,
                            "umap": param_grid_strings[params_keys[2]][u_i],
                            "umap_object": UMAP,
                            "log": param_grid_strings[params_keys[3]][lr_i],
                            "log_object": logReg,
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


class DifferentialEvolution():
    def __init__(self, random_state=123, score_metric="BIC"):
        if not score_metric in Scores.scores.keys():
            raise Exception("Given score metric does not exists in Scores")

        if score_metric == "BIC":
            self.convert_to_minimum = False
        else:
            self.convert_to_minimum = True
        self.random_state = random_state
        self.score_metric = score_metric
        self.score_df = pd.DataFrame()
        self.categorical_values = OrderedDict()

    def create_lists(self, paramethers: Dict) -> Dict:
        """
        Creates dict of classes' paramethers.\n
        intput: Parameters = { "a": [type, min_bound, max_bound]}\n
        """
        ret_dict = {}
        for key in paramethers:
            # eval(str) - creates object from string
            obj = paramethers[key]
            if isinstance(obj, List):
                ret_dict[key] = [eval(x) for x in obj]
            elif isinstance(obj, Dict):
                obj_dict = {}
                for p in obj:
                    values = obj[p]
                    type = values[0]
                    obj_dict[p] = {
                        "param_name": p,
                        "param_type": type,
                    }
                    if type == "categorical":
                        obj_dict[p]["min_bound"] = 0.0
                        obj_dict[p]["max_bound"] = len(values[1:]) - 1
                        self.categorical_values[key] = {p: values[1:]}
                        print(self.categorical_values)
                    elif type in ["int", "float"]:
                        obj_dict[p]["min_bound"] = min(values[1:])
                        obj_dict[p]["max_bound"] = max(values[1:])
                    else:
                        raise Exception("%s is unsupported data type" % type)

                ret_dict[key] = obj_dict

        print(ret_dict)
        return ret_dict

    def evaluate(self,
                 X,
                 y,
                 params: Dict,
                 cross_validations=3,
                 max_iters=1,
                 popsize=5,
                 cross_validate_transformers=False,
                 fit_transform_all_data=False,
                 threads=1) -> pd.DataFrame:
        param_grid = self.create_lists(params)

        # Unpack boundaries
        lower_boundries = []
        upper_boundries = []
        for obj in ["PCA", "UMAP", "LogisticRegression"]:
            for pca_p in param_grid[obj]:
                min_b = param_grid[obj][pca_p]["min_bound"]
                max_b = param_grid[obj][pca_p]["max_bound"]
                lower_boundries.append(min_b)
                upper_boundries.append(max_b)
        boundries = Bounds(lower_boundries, upper_boundries)
        print(boundries)

        self.evaluation_i = 1
        self.score_df = pd.DataFrame()

        self.all_evaluations = (max_iters + 1) * popsize * len(lower_boundries)
        self.evolution_start_time = time.monotonic()
        print_with_border("Perform evolution with %d evaluations." % (self.all_evaluations))

        results = differential_evolution(
            self.func,
            boundries,
            args=(
            param_grid, X, y, cross_validations, threads != 1, cross_validate_transformers, fit_transform_all_data),
            popsize=popsize,
            maxiter=max_iters,
            polish=False,
            workers=threads,
            seed=self.random_state,
            disp=True,
        )
        print(results)
        return self.score_df

    def func(self, hyperparameters, param_grid, X, y, cross_validations, multithread=False,
             cross_validate_transformers=False, fit_transform_all_data=False):
        start_time = time.monotonic()

        scaler = param_grid['scalers'][0]
        pca, UMAP, logReg = self.create_objects_from_param_grid(hyperparameters, param_grid)
        if pca.n_components < UMAP.n_components:
            print("Drop")
            return np.inf
        pca.random_state = 123
        UMAP.random_state = 123
        logReg.random_state = 123
        # print_with_border("Creating differential evolution")

        # X_scaled = scaler.fit_transform(X, y)
        # X_scaled_pca = pca.fit_transform(X_scaled, y)
        # X_scaled_pca_umap = UMAP.fit_transform(X_scaled_pca, y)

        # Splitting into train and test sets


        if cross_validate_transformers:
            pipeline = Pipeline([
                ("scaler", scaler),
                ("pca", pca),
                ("umap", UMAP),
                ("logreg", logReg)
            ])
            scores = cross_validate(pipeline, X, y, cv=cross_validations, scoring=Scores.scores,
                                    return_train_score=False, return_estimator=False)
        else:
            if fit_transform_all_data:
                pipeline = Pipeline([
                ("scaler", scaler),
                ("pca", pca),
                ("umap", UMAP),
                ])

                X_transformed = pipeline.fit_transform(X, y)

                scores = cross_validate(logReg, X, y, cv=cross_validations, scoring=Scores.scores,
                                        return_train_score=False, return_estimator=False)
            else:
                pipeline = Pipeline([
                    ("scaler", scaler),
                    ("pca", pca),
                    ("umap", UMAP),
                ])

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                    random_state=self.random_state)

                X_train_transformed = pipeline.fit_transform(X_train, y_train)
                X_test_transformed = pipeline.transform(X_test)

                scores = Scores.get_scores(logReg, X_test_transformed, y_test)


        iter_score = {
            "scaler": param_grid["scalers"][0],
            "pca": pca,
            "umap": UMAP,
            "log": logReg,
        }

        # Calculate mean from k-validations
        for score, k_val_arr in scores.items():
            iter_score[score] = k_val_arr.mean()
            iter_score[score + "_std"] = 2 * k_val_arr.std()
        if not multithread:
            if self.score_df.empty:
                score_df = pd.DataFrame(data=iter_score, index=[self.evaluation_i])
            else:
                self.score_df.loc[self.evaluation_i] = iter_score

        if self.convert_to_minimum:
            scr = 1 - iter_score["test_" + self.score_metric]
        else:
            scr = iter_score["test_" + self.score_metric]
        reg_time = time.monotonic() - start_time
        reg_time_all = time.monotonic() - self.evolution_start_time
        print(hyperparameters)
        print("evaluation: %d: %.2f\t| computation time: %s" % (
        self.evaluation_i, scr, str(timedelta(seconds=reg_time)).split('.', 2)[0]))

        if not multithread:
            time_left = (self.all_evaluations - self.evaluation_i) * reg_time_all / self.evaluation_i
            sys.stdout.write(f"\r %.2f %% done| Elapsed time: %s | Estimated time left: %s\n" % (
                (self.evaluation_i - 1) / self.all_evaluations * 100,
                str(timedelta(seconds=reg_time_all)).split('.', 2)[0],  # Split is used to remove ms
                str(timedelta(seconds=time_left)).split('.', 2)[0]  # Split is used to remove ms
            ))
            self.evaluation_i += 1
        return scr

    def create_objects_from_param_grid(self, hyperparameters, param_grid):
        ret_obj = []
        hyper_par_i = 0
        for obj_name in ["PCA", "UMAP", "LogisticRegression"]:
            temp = "%s(" % obj_name
            obj_params = param_grid[obj_name]
            for p_i, param in enumerate(obj_params):
                value = hyperparameters[hyper_par_i]
                value_type = obj_params[param]['param_type']
                if value_type == "categorical":
                    value = self.categorical_values[obj_name][param][int(round(value))]
                    if isinstance(value, str):
                        value = "'" + value + "'"
                elif value_type == "int":
                    value = int(round(value))

                temp += "%s=%s," % (str(param), value)
                hyper_par_i += 1
            temp += ")"
            print(temp)
            ret_obj.append(eval(temp))

        return ret_obj
