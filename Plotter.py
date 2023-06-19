import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Dict
from matplotlib.lines import Line2D
import addcopyfighandler # enables ctrl + c -> save matplotlib figure to clipboard



class Plotter():
    """
    
    :param fig_size: 
    :param labels_from_cols: Column names with object properties to show in labels in method plot_scores
    :param label_shortening_start: Character length needed to start param_name shortening
    :param label_shortening_amount: Character length to which  param_name will be shortened
    """
    def __init__(self,
                 fig_size = (16, 11),
                 labels_from_cols = ("pca", "umap", "log"),
                 label_shortening_start = 7,
                 label_shortening_amount = 6,
                 banned_label_params = ("solver")
                 ):
    

        self.score_data = []
        self.colors = ["blue", "orange", "green", "red", "purple", "yellow", "cyan", "magenta", "lime", "pink", "teal", "gold",
          "black", "darkgray", "gray", "lightgray", "silver", "whitesmoke", "aquamarine", "chocolate", "coral",
          "darkgreen", "indigo", "khaki", "aliceblue", "cadetblue", "darkorchid", "honeydew", "moccasin",
          "palevioletred", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
          "lightsalmon", "mediumaquamarine", "olivedrab", "sienna", "tomato", "violet"]
        self.fig_size = fig_size
        self.labels_from_cols = labels_from_cols
        self.label_shortening_start = label_shortening_start
        self.label_shortening_amount = label_shortening_amount
        self.custom_labels = []
        self.titles = []
        self.banned_label_params = banned_label_params


    def add_score_file(self, csv_file, title, sep = ";", custom_labels=None):
        """
        Add .csv file with hyperparamether tuning data. This data is used in plot_scores() method.
        :param csv_file: filepath of .csv file.
        :param title: str title to show in legend.
        :param sep: separator in .csv file.
        :param custom_labels: (row0_label, row1_label, ...) with size == csv_file's rows
        """
        new_score_df = pd.read_csv(csv_file, sep = sep)
        if title != None:
            self.titles.append(title)
        if custom_labels != None:
            if isinstance(custom_labels, str):
                custom_labels = [custom_labels]

            if len(custom_labels) != len(new_score_df):
                raise Exception("custom_labels count %s")

            self.custom_labels = list(custom_labels)

        self.score_data.append(new_score_df)

    def get_best_by_score(self, data_frame, amount, score = "BIC") -> pd.DataFrame:
        asc_bool = False
        if score == "test_BIC":
            asc_bool = True

        indices = data_frame[score].sort_values(ascending=asc_bool).index
        sorted = data_frame.loc[indices[:amount]]

        return sorted

    def create_labels(self, data_frame, ):
        def shorten_label(label) -> str:
            label = label.replace("(", ",")
            label = label.replace(")", ",")
            l_split = label.split(",")

            # Check paramethers
            for w_i, par in enumerate(l_split[1:-2]):
                e_split = par.split("=")
                if e_split[0] in self.banned_label_params:
                    e_split.pop(0)
                    e_split.pop(0)
                else:
                    if len(e_split[0]) > self.label_shortening_start:
                        # print("shorten: %s to %s" %(e_split[0], e_split[0][:label_shortening_amount]))
                        e_split[0] = e_split[0][:self.label_shortening_amount]+"."

                l_split[w_i+1] = "=".join(e_split, )

            #fix parenthesis
            l_split[0] += '('+l_split[1]
            l_split[-1] = l_split[-2]+')'+l_split[-1]

            #pop duplicates
            l_split.pop(1)
            l_split.pop(-2)


            # obj_name = l_split[0]
            return ", ".join(l_split)

        temp = np.array([])

        for lab in self.labels_from_cols:
            labels = np.array(data_frame[lab])
            for l_i, label in enumerate(labels):
                labels[l_i] = shorten_label(label)
            
            temp = np.concatenate((temp, labels), axis = None)
        temp = list(temp)

        ret_arr = [""]*(len(data_frame))
        for i in range(len(temp)//len(self.labels_from_cols)):
            # print("temp i:", temp[i])
            # print("ret i:", ret_arr[i])
            for j in range(len(self.labels_from_cols)):
                j*=len(data_frame)
                ret_arr[i] += temp[i+j]+"\n"
        
        return ret_arr


    def create_legend(self):
        lines = []
        for i, title in enumerate(self.titles):
            # Define custom proxy artists
            line = Line2D([0], [0], color=self.colors[i], linewidth=8)
            lines.append(line)

        plt.legend(lines, self.titles)

    def plot_scores(self,
                    params_to_plot = ("test_BIC"),
                    plot_n_best_from_files = 5,
                    ):
        """
        Plots bar plot containing comparison of scoring paramethers from different classifications.
        Load data with add_score_file() method.

        :param params_to_plot: Tuple[str] or "all" with scores to plot.
            Each score is plotted on different figure. Possible scores:
            ["test_BIC", "test_sensitivity", "test_specificity",
            "test_PPV", "test_NPV", "test_balanced_accuracy",
             "test_f1"]
        :param plot_n_best_from_files: - number of rows from each file to plot
        """




        print("\nScore DataFrame columns: ")
        print(self.score_data[0].columns)
        if isinstance(params_to_plot, str):
            if params_to_plot.lower() == "all":
                params_to_plot = ["test_BIC", "test_sensitivity", "test_specificity",
                                  "test_PPV", "test_NPV", "test_balanced_accuracy",
                                  "test_f1"]
            else: # Tuple from one element is str. Convert str to list -> for loop
                params_to_plot = [params_to_plot]

        for param in params_to_plot:
            # print(param)
            asc_bool = False
            if param == "test_BIC":
                asc_bool = True

            plt.figure(figsize=self.fig_size)
            combined = pd.DataFrame()
            labels = np.array([])
            colors = np.array([])
            color_i = 0
            for d in self.score_data:
                d = self.get_best_by_score(d, plot_n_best_from_files, score="test_BIC")
                created_labels = self.create_labels(d)
                labels = np.concatenate((labels, created_labels), axis = 0)
                combined = pd.concat([combined, d], ignore_index= True)
                colors = np.concatenate((colors, [self.colors[color_i]]*len(d)), axis = 0)
                color_i += 1

            bar_x = combined[param].sort_values(ascending=asc_bool).index
            ax = combined[param].sort_values(ascending=asc_bool).plot(kind="bar", color=colors[bar_x])
            bar_y = combined[param].loc[bar_x]
            bar_err = combined[param + "_std"].loc[bar_x]

            if param != "test_BIC":
                bar_err[(bar_y + bar_err) > 1.0] = np.clip(bar_y + bar_err, 0.0, 1.0) - bar_y
                bar_err[(bar_y - bar_err) < 0.0] = np.clip(bar_y - bar_err, 0.0, 1.0) + bar_y

            ax.errorbar(bar_x, bar_y, yerr=bar_err, fmt='o', color='black',
                         ecolor='black', elinewidth=2, capsize=10, capthick=2)
            ax.set_xticklabels(labels)
            ax.set_title(param)
            self.create_legend()
            plt.tight_layout()
            plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    output_folder = r"%s\Outputs"%os.getcwd()
    plt.rcParams.update({'font.size': 16})


    plotter = Plotter(
        fig_size=(16, 16), #Wychodzi poza ekran, ale dobrze widoczne po skopiowaniu
        labels_from_cols=("pca", "umap", ), #"log"
        label_shortening_start=7,
        label_shortening_amount=6,
        banned_label_params=("solver")

    )
    # plotter.add_score_file(output_folder+r"\19BIC_grid_umap_fit_all_with_y.csv", "1")
    plotter.add_score_file(output_folder+r"\normalized_umap_test_ys_10files_GRID_13_06_2023_21_38_04.csv",
                           "UMAP fit train with ys")
    plotter.add_score_file(output_folder + r"\normalized_umap_test_no_ys_10files_GRID_13_06_2023_22_34_13.csv",
                           "UMAP fit train without ys")

    plotter.plot_scores(
        params_to_plot="all",
        plot_n_best_from_files=10
    )
