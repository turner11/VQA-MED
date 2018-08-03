import os
import matplotlib.pyplot as plt
from parsers.VQA18 import Vqa18Base
from pre_processing.known_find_and_replace_items import all_tags, dbg_file_csv_train, dbg_file_xls_processed_train
from utils.os_utils import File

dump_plots_dir = os.path.abspath(os.path.join(".","dumped_plots\\"))
File.validate_dir_exists(dump_plots_dir )
def plot(data_path):
    parser = Vqa18Base.get_instance(data_path)
    df = parser.data
    plot_count_tags(df)


def plot_count_tags(df):
    # df = df.reindex_axis(df[all_tags].sum().sort_values().index, axis=1)
    df_cols = df[all_tags]
    data = df_cols.apply(sum)
    ax = data.plot(kind='barh', title="Tag Count", figsize=(15, 10), legend=False, fontsize=12)
    for i, v in enumerate(data):
        ax.text(v + 3, i + .25, str(v), color='black', fontweight='bold')
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("Tag", fontsize=12)

    path = os.path.join(dump_plots_dir,"tag_count.png")
    plt.savefig(path)
    plt.show()



def main():
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('-p', dest='path', help='')
    # args = parser.parse_args()

    fn = dbg_file_xls_processed_train# dbg_file_csv_train
    plot(fn)


if __name__ == '__main__':
    main()
