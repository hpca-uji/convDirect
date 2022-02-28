#!/usr/bin/env python

import sys
import os
import glob

# Global variables
input_dir = ""
plots_titles_suffix = ""
dfs = {}
best_dfs = {}
blis_dfs = {}


def my_help():
    print("""
Usage: {} INPUT_DIR [SUFFIX]

where:
 INPUT_DIR is the input directory, usually 'convdirect_output_hostname'.
 SUFFIX is the suffix that will be used on the plots titles, if this parameter is
        not provided, 'performance on Hostname' will be used.
 
    """.format(sys.argv[0]))


def process_args():
    global input_dir, plots_titles_suffix
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        my_help()
        sys.exit(1)

    input_dir = sys.argv[1]
    if len(sys.argv) == 3:
        plots_titles_suffix = sys.argv[2]
    else:
        host_info = "Unknown"
        host_info_file = os.path.join(input_dir, "host.nfo")
        if os.path.exists(host_info_file):
            with open(host_info_file) as f:
                host_info = f.readline().strip()
        else:
            parts = input_dir.split("_")
            if len(parts) == 3 and parts[0] == "convdirect" and parts[1] == "output":
                host_info = parts[2]
            else:
                print("Warning: the input dir does not has a 'host.nfo' file and it is not called as\n "
                      "         'convdirect_output_hostname'. Therefore, the name of the host were the\n"
                      "         evaluation was performed can not be guessed.")
        plots_titles_suffix = "performance on {}".format(host_info)


IM2COL, CONVGEMM, BLIS_8X12, BLIS_4X20, BLIS_BLIS, SHALOM, TZEMENG = ("Im2col", "ConvGemm",
                                                                      "Blis_mk-8x12", "Blis_mk-4x20", "Blis_mk-blis",
                                                                      "Shalom", "Tze-Meng")


def performance_results():
    global dfs, best_dfs, blis_dfs
    output_dir = os.path.join(input_dir, "performance_results")

    method_from = {
        "convdirect_im2row_nhwc_default": IM2COL,
        "convdirect_conv_gemm_nhwc_default": CONVGEMM,
        "convdirect_block_blis_nhwc_8x12": BLIS_8X12,
        "convdirect_block_blis_nhwc_4x20": BLIS_4X20,
        "convdirect_block_blis_nhwc_blis": BLIS_BLIS,
        "convdirect_block_shalom_nhwc_7x12_npa_u4": SHALOM,
        "convdirect_tzemeng_nhwc_7x12_u4": TZEMENG,
    }
    cols_ordered = [IM2COL, CONVGEMM, SHALOM, BLIS_8X12, BLIS_4X20, BLIS_BLIS, TZEMENG]

    try:
        import pandas as pd
        import tabulate
        from pandas.errors import EmptyDataError
    except ModuleNotFoundError:
        print("Error: The required modules for processing the results have not been found!\n"
              "       Please, install the pandas and tabulate python modules with:\n"
              "          pip install pandas tabulate\n"
              "       Add the '--user' install option if not in a python environment.")
        sys.exit(1)

    print("\033[1m\033[32mIMPORTING EVALUATION DATA\033[0m")
    nets = []
    df_parts = []
    for filename in glob.glob(os.path.join(input_dir, "*.csv")):
        print("Importing '{}'".format(filename))
        net, algorithm = os.path.basename(filename)[:-4].split("_-_")
        if net not in nets:
            nets.append(net)
        try:
            partial_df = pd.read_csv(filename, delimiter=";")
        except EmptyDataError:
            print("\033[1m\033[33mWaring: Filename '{}' has no input. Ignoring it!\033[0m".format(filename))
        else:
            partial_df = partial_df.rename(columns={"l": "#Layer", "Algorithm": "Method"})
            partial_df["Net"] = net
            partial_df["Method"] = method_from[algorithm]
            df_parts.append(partial_df)

    df = pd.concat(df_parts, ignore_index=True)
    print()

    print("\033[1m\033[32mEXPORTING PERFORMANCE RESULTS\033[0m")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for net in nets:
        print("Exporting {} performance results to csv, txt and tex...".format(net))
        df_net = df.query("Net == '{}'".format(net)).pivot_table(values="GFLOPS", index="#Layer", columns="Method")
        cols_present = df_net.columns.tolist()
        df_net = df_net[[c for c in cols_ordered if c in cols_present]]
        df_net["Im2col vs ConvGemm"] = df_net[[IM2COL, CONVGEMM]].max(axis=1)
        blis_cols = [c for c in [BLIS_8X12, BLIS_4X20, BLIS_BLIS] if c in cols_present]
        blis_label = None
        if len(blis_cols) > 1:
            blis_label = " vs ".join(blis_cols)
            df_net[blis_label] = df_net[blis_cols].max(axis=1)
        elif len(blis_cols) == 1:
            blis_label = blis_cols[0]
        basename = os.path.join(output_dir, net)
        df_net.to_csv("{}.csv".format(basename), sep=";")
        df_net.to_markdown("{}.txt".format(basename))
        try:
            df_net.style.to_latex("{}.tex".format(basename))
        except (AttributeError, ImportError):
            df_net.to_latex("{}.tex".format(basename))
        dfs[net] = df_net
        # ---
        cols_present = df_net.columns.tolist()
        best_cols = [c for c in ["Im2col vs ConvGemm", TZEMENG, SHALOM] if c in cols_present]
        if blis_label:
            best_cols.append(blis_label)
        best_df_net = df_net[best_cols].reset_index()
        best_dfs[net] = best_df_net
        # print(best_df_net.to_markdown())
        # ---
        if len(blis_cols):
            blis_df_net = df_net[blis_cols].reset_index()
            blis_dfs[net] = blis_df_net
            # print(blis_df_net.to_markdown())

    print()
    print("The performance results are in '{}'\n".format(os.path.abspath(output_dir)))


def plots_matplotlib():
    global best_dfs, blis_dfs
    output_dir = os.path.join(input_dir, "plots")

    print("\033[1m\033[32mGENERATING PLOTS\033[0m")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        print("Error: The next python modules are required to do the plots: numpy and matplotlib.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    def bars(df, plot_title, filename):
        print("Generating '{}'...".format(filename))
        # Labels
        labels = df["#Layer"].to_list()
        x = np.arange(len(labels))  # the label locations
        # Methods
        methods = [c for c in df.columns.to_list() if c != "#Layer"]
        n_methods = len(methods)
        # Create figure
        figure_width = 11  # a4 paper is 11-3/4 inches width
        figure_height = figure_width * 30 / 120  # same relation aspect as in Héctor version
        plt.figure()
        plt.subplots(figsize=(figure_width, figure_height), dpi=300, layout='constrained')
        # Bar subplots
        total_bar_width = 0.8
        bar_width = total_bar_width / n_methods
        # See matplotlib color maps in https://matplotlib.org/stable/tutorials/colors/colormaps.html
        cm = plt.cm.get_cmap('tab20b')
        ic = len(methods) / 2 - .5  # i centered
        for i, method in enumerate(methods):
            values = df[method].to_list()
            plt.bar(x + (i - ic) * bar_width, values, bar_width, label=method, zorder=3, color=cm.colors[i])
        # Add title, legend, labels, etc.
        plt.title(plot_title)
        plt.legend(loc="upper left", fontsize=8, ncol=n_methods)
        plt.margins(x=0.005, y=0.2, tight=True)
        plt.grid(True, zorder=0)
        plt.ylabel('GFLOPS')
        plt.xlabel("#CNN layer")
        plt.xticks(x, labels, fontsize=8, rotation=60)
        # Save figure
        plt.savefig(filename)

    for net, df in best_dfs.items():
        plot_title = "{} {}".format(net, plots_titles_suffix)
        filename = "best_{}.pdf".format(plot_title.replace(" ", "_").replace("(", "").replace(")", "")).lower()
        bars(df, plot_title, os.path.join(output_dir, filename))

    for net, df in blis_dfs.items():
        plot_title = "{} {}".format(net, plots_titles_suffix)
        filename = "blis_{}.pdf".format(plot_title.replace(" ", "_").replace("(", "").replace(")", "")).lower()
        bars(df, plot_title, os.path.join(output_dir, filename))

    print()
    print("The plots are in '{}'\n".format(os.path.abspath(output_dir)))


def main():
    process_args()
    performance_results()
    plots_matplotlib()


if __name__ == '__main__':
    main()
