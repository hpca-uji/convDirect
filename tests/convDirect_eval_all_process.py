#!/usr/bin/env python3
import math
import sys
import os
import glob

# Global variables
input_dir = ""
plots_titles_suffix = ""
dfs = {}
best_dfs = {}
blis_dfs = {}
shalom_dfs = {}

# Follow the nomenclature used on "Reformulating the Direct Convolution for High-Performance on ARM Processors" article
for_article = True


def my_help():
    print("""
Usage: {} INPUT_DIR [SUFFIX]

where:
 INPUT_DIR is the input directory, usually 'convdirect_output_hostname'.
 SUFFIX is the suffix that will be used on the plots titles, if this parameter is
        not provided, 'performance on Hostname' will be used. To generate an empty
        suffix, "" can be used.
 
    """.format(sys.argv[0]))


def process_args():
    global input_dir, plots_titles_suffix
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        my_help()
        sys.exit(1)

    input_dir = sys.argv[1]
    if len(sys.argv) == 3:
        plots_titles_suffix = sys.argv[2].strip()
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


if not for_article:
    (IM2COL, CONVGEMM, TZEMENG,
     SHALOM_7x12, SHALOM_6x16,
     BLIS_8X12, BLIS_4X20, BLIS_BLIS,) = ("Im2col", "ConvGemm", "Tze-Meng",
                                          "Shalom (7x12)", "Shalom (6x16)",
                                          "BLIS (8x12)", "BLIS (4x20)", "BLIS (BLIS)")
else:
    (IM2COL, CONVGEMM, TZEMENG,
     SHALOM_7x12, SHALOM_6x16,
     BLIS_8X12, BLIS_4X20, BLIS_BLIS,) = ("Im2col", "ConvGemm", "BLOCKED",
                                          "NEW-A (7x12)", "NEW-A (6x16)",
                                          "NEW-B (8x12)", "NEW-B (4x20)", "NEW-B (BLIS)")


def performance_results():
    global dfs, best_dfs, blis_dfs, shalom_dfs
    output_dir = os.path.join(input_dir, "performance_results")

    method_from = {
        "convdirect_im2row_nhwc_default": IM2COL,
        "convdirect_conv_gemm_nhwc_default": CONVGEMM,
        "convdirect_tzemeng_nhwc_7x12_u4": TZEMENG,
        "convdirect_block_shalom_nhwc_7x12_npa_u4": SHALOM_7x12,
        "convdirect_block_shalom_nhwc_6x16_npa_u4": SHALOM_6x16,
        "convdirect_block_blis_nhwc_8x12": BLIS_8X12,
        "convdirect_block_blis_nhwc_4x20": BLIS_4X20,
        "convdirect_block_blis_nhwc_blis": BLIS_BLIS,
    }
    cols_ordered = [IM2COL, CONVGEMM, TZEMENG, SHALOM_7x12, SHALOM_6x16, BLIS_8X12, BLIS_4X20, BLIS_BLIS]

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
        net = net.replace("-", " ")
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
        # Best(Im2col, ConvGemm)
        im2col_convgemm_cols = [c for c in [IM2COL, CONVGEMM] if c in cols_present]
        im2col_convgemm_label = None
        if len(im2col_convgemm_cols) >= 1:
            im2col_convgemm_label = "LOWERING" if for_article else "Best({})".format(", ".join(im2col_convgemm_cols))
            df_net[im2col_convgemm_label] = df_net[im2col_convgemm_cols].max(axis=1)
        # Best of Shalom methods
        shalom_cols = [c for c in [SHALOM_7x12, SHALOM_6x16] if c in cols_present]
        shalom_label = None
        if len(shalom_cols) >= 1:
            shalom_label = "NEW-A" if for_article else "Best({})".format(", ".join(shalom_cols))
            df_net[shalom_label] = df_net[shalom_cols].max(axis=1)
        # Best of BLIS methods
        blis_cols = [c for c in [BLIS_8X12, BLIS_4X20, BLIS_BLIS] if c in cols_present]
        blis_label = None
        if len(blis_cols) >= 1:
            blis_label = "NEW-B" if for_article else "Best({})".format(", ".join(blis_cols))
            df_net[blis_label] = df_net[blis_cols].max(axis=1)
        # ---
        basename = os.path.join(output_dir, net.replace(" ", "_"))
        df_net.to_csv("{}.csv".format(basename), sep=";")
        df_net.to_markdown("{}.txt".format(basename))
        try:
            df_net.style.to_latex("{}.tex".format(basename))
        except (AttributeError, ImportError):
            df_net.to_latex("{}.tex".format(basename))
        dfs[net] = df_net
        # ---
        cols_present = df_net.columns.tolist()
        best_cols = [c for c in [im2col_convgemm_label, TZEMENG, shalom_label, blis_label] if c in cols_present]
        best_df_net = df_net[best_cols].reset_index()
        best_dfs[net] = best_df_net
        # print(best_df_net.to_markdown())
        # ---
        if len(blis_cols):
            blis_df_net = df_net[blis_cols].reset_index()
            blis_dfs[net] = blis_df_net
            # print(blis_df_net.to_markdown())
        if len(shalom_cols):
            shalom_df_net = df_net[shalom_cols].reset_index()
            shalom_dfs[net] = shalom_df_net
            # print(blis_df_net.to_markdown())

    print()
    print("The performance results are in '{}'\n".format(os.path.abspath(output_dir)))


def plots_matplotlib():
    global best_dfs, blis_dfs, shalom_dfs
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

    def bars(df_, plot_title_, filename_, colors_, pretty_max_, y_locs_, y_labels_):
        print("Generating '{}'...".format(filename_))
        # Labels
        labels = df_["#Layer"].to_list()
        x = np.arange(len(labels))  # the label locations
        # Methods
        methods = [c for c in df_.columns.to_list() if c != "#Layer"]
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
        # cm = plt.cm.get_cmap('tab20b')
        ic = len(methods) / 2 - .5  # i centered
        for i, method in enumerate(methods):
            values = df_[method].to_list()
            plt.bar(x + (i - ic) * bar_width, values, bar_width, label=method, zorder=3, color=colors_[i])
        # Add title, legend, labels, etc.
        plt.title(plot_title_)
        plt.legend(loc="upper left", fontsize=8, ncol=n_methods)
        plt.margins(x=0.005, y=0.2, tight=True)
        plt.grid(True, zorder=0)
        # Y
        plt.ylabel('GFLOPS')
        if y_locs_ is not None:
            plt.yticks(y_locs_, y_labels_)
        else:
            plt.ylim(0, pretty_max_)
        # X
        plt.xlabel("#CNN layer")
        plt.xticks(x, labels, fontsize=8, rotation=60)
        # Save figure
        plt.savefig(filename_)
        # Get locs and labels from yticks
        y_locs, y_labels = plt.yticks()
        # Close figure
        plt.close("all")
        # Return locs and labels
        return y_locs, y_labels

    def split(df_):
        columns = list(df.columns)
        columns.pop(columns.index("#Layer"))
        max_gflops = df[columns].max().max()
        order_of_magnitude = math.floor(math.log(max_gflops, 10))
        divisor = 10 ** order_of_magnitude / 2
        pretty_max = (max_gflops * 1.3) // divisor * divisor
        parts = len(df_.index) // 25 + 1
        if parts == 1:
            return [(df_, "", pretty_max), ]
        else:
            return [(x, str(i), pretty_max) for i, x in enumerate(np.array_split(df_, parts), 1)]

    def draw_bars(net_, df_, prefix_, plots_titles_suffix_, output_dir_, colors_):
        previous_y_locs = None
        previous_y_labels = None
        for df_part, suffix, pretty_max in split(df_):
            title_parts = [x for x in (net_, plots_titles_suffix_, suffix) if x != ""]
            plot_title = " ".join(title_parts)
            plot_title_escaped = plot_title.replace(" ", "_").replace("(", "").replace(")", "")
            filename = "{}_{}.pdf".format(prefix_, plot_title_escaped).lower()
            previous_y_locs, previous_y_labels = bars(df_part, plot_title, os.path.join(output_dir_, filename), colors_,
                                                      pretty_max, previous_y_locs, previous_y_labels)

    cm = plt.cm.get_cmap('tab10')
    colors = [cm.colors[x] for x in range(4)]
    colors.reverse()
    for net, df in best_dfs.items():
        draw_bars(net, df, "best", plots_titles_suffix, output_dir, colors)

    cm = plt.cm.get_cmap('tab20c')
    colors = [cm.colors[x] for x in range(3)]
    for net, df in blis_dfs.items():
        draw_bars(net, df, "blis", plots_titles_suffix, output_dir, colors)

    cm = plt.cm.get_cmap('tab20c')
    colors = [cm.colors[4 + x] for x in range(3)]
    for net, df in shalom_dfs.items():
        draw_bars(net, df, "shalom", plots_titles_suffix, output_dir, colors)

    print()
    print("The plots are in '{}'\n".format(os.path.abspath(output_dir)))


def main():
    process_args()
    performance_results()
    plots_matplotlib()


if __name__ == '__main__':
    main()
