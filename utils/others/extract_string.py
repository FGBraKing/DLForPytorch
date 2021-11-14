import re
import os
# import time
from glob import glob
from matplotlib import pyplot as plt
import pandas as pd
# import numpy as np
# import seaborn as sns
# from utils.others.img_io import plot_2d
# from .img_io import plot_2d


def find_best_dice(logs_dir=None, pat=r"^number(?:.*\s)+(?:total.*\s).*?dice.*?(\d\.\d+).*"):
    # '^number(?:.*\s)+(?:total.*\s)dice.*?(\d\.\d+).*'

    result_files = glob(os.path.join(logs_dir, '*.txt'))
    print(f'there is nothing in {logs_dir}')

    pat = re.compile(pat)
    best_dice = 0
    best_weight = None
    for result_file in result_files:
        with open(result_file, mode='r') as fread:
            content = fread.read()

        dice_result = pat.match(content)
        if dice_result is not None:
            dice = float(dice_result.groups()[0])
            if dice > best_dice:
                best_dice = dice
                best_weight = os.path.basename(result_file).split('.')[0]
            elif dice == best_dice:
                if isinstance(best_weight, list):
                    best_weight.append(os.path.basename(result_file).split('.')[0])
                else:
                    best_weight = [best_weight]
                    best_weight.append(os.path.basename(result_file).split('.')[0])

    print(f'best_dice: {best_dice}\t'
          f'best_weight: {best_weight}')
    return {'best_dice': best_dice, 'best_weight': best_weight}


def extract_loss(loss_file, pat=r'^\(epoch.*\).*?(({}):\s+?(\d+\.\d+))', loss_name='dice'):
    pat = re.compile(pat.format(loss_name))
    loss_list = []
    with open(loss_file, 'r') as f_loss:
        for line in f_loss.readlines():
            match_result = pat.match(line)
            if match_result is not None:
                # print(match_result.groups()[0])
                loss_list.append(float(match_result.groups()[-1]))
    batchs = len(loss_list)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(batchs), loss_list)
    ax.set_title('loss curve')
    # ax.axis('off')
    ax.set(xlabel='batch', ylabel=loss_name)
    plt.show()


def extract_metrics(resolving_file, dividually=False):
    # visual_names=('DC', 'recall', 'precision', 'accuracy'),
    vaild_pat = re.compile(r'^\((.*)\)\s*?(.*)\s*$')
    dict_pat = re.compile(r'(\w+):\s*(\d+(?:\.\d+)?)')

    info_dict_list = []
    meta_keys = None
    metrics_keys = None
    with open(resolving_file, 'r') as f_metrics:
        for line in f_metrics.readlines():
            vaild_match = vaild_pat.match(line)
            if vaild_match is not None:
                if meta_keys is None and metrics_keys is None:
                    meta_str = vaild_match.groups()[0]
                    metrics_str = vaild_match.groups()[1]
                    meta_keys = list(dict(dict_pat.findall(meta_str)).keys())
                    metrics_keys = list(dict(dict_pat.findall(metrics_str)).keys())
                info_dict_list.append(dict(dict_pat.findall(line)))
    print('meta_keys: {}\nmetrics_keys: {}'.format(meta_keys, metrics_keys))
    info_df = pd.DataFrame(data=info_dict_list, dtype=float)
    info_df[meta_keys] = info_df[meta_keys].astype(int)

    # print(info_df.info())
    print(info_df.describe())
    plt.figure()
    info_df[metrics_keys].plot()
    plt.title('metrics all', fontsize=14)
    plt.show()
    if dividually:
        for key in metrics_keys:
            plot_2d(range(len(info_df[key])), info_df[key], label=key, fig_title='Metrics dividually')
    return info_df


def plot_2d(x, y, *args, fig_title=None, ax_title=None, x_label=None, y_label=None, **kwargs):
    fig, ax = plt.subplots()
    if fig_title:
        fig.suptitle(fig_title, fontsize=14)    # , fontweight='bold'
    if ax_title:
        ax.set_title(ax_title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    ax.plot(x, y, *args, **kwargs)
    ax.legend()
    fig.show()
    plt.close(fig)


if __name__ == '__main__':
    test_dir = r'/home/lf/raid_lf/PROJECT/DLForPytorch/' \
               r'traces/results/trus_unet3d_DDP_SynBN_crop12_bs4_ch32_dc_adam_1e-4/train/slide_test_pad_noaug'

    # result = find_best_dice(test_dir)
    # print(result)

    loss_file = r'/home/lf/raid_lf/PROJECT/DLForPytorch/traces/logs/' \
                r'trus_unet3d_DDP_SynBN_crop128_bs3x4_ch32_dc_adam_1e-4/loss_log.txt'
    # extract_loss(loss_file)

    metrics_file = r'/home/lf/raid_lf/PROJECT/DLForPytorch/traces/' \
                   r'logs/trus_unet3d_DDP_SynBN_crop128_bs3x4_ch32_dc_adam_1e-4/metrics_log.txt'

    data_df = extract_metrics(metrics_file, dividually=True)
