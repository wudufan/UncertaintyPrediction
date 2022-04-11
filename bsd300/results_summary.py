'''
Summarize the results
'''

# %%
import pandas as pd
import os
import numpy as np
import argparse
import glob
import sys
import scipy.stats


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        default='/home/local/PARTNERS/dw640/mnt/women_health_internal/dufan.wu/bsd300_denoising/depth_1_ch_32_epoch_500'
    )
    parser.add_argument('--loss', default='PSNR')
    parser.add_argument('--data', default='Pred')
    parser.add_argument('--val_col', default='Test')

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args([])
        args.debug = True
    else:
        args = parser.parse_args()
        args.debug = False

    for k in vars(args):
        print(k, '=', getattr(args, k))

    return args


# %%
def get_filenames(input_dir):
    filenames = glob.glob(os.path.join(input_dir, '*_seed_*.csv'))

    # build a manifest for each method
    df_methods = {}
    for filename in filenames:
        basename = os.path.basename(filename)
        ind = basename.find('_seed_')
        method_name = basename[:ind]
        seed = basename[ind + 1:-4].split('_')[1]

        if method_name not in df_methods:
            df_methods[method_name] = []

        df_methods[method_name].append({
            'Seed': seed,
            method_name: filename
        })

    # convert to df
    for k in df_methods:
        df_methods[k] = pd.DataFrame(df_methods[k])

    # merge everything
    df = None
    for k in df_methods:
        if df is None:
            df = df_methods[k].copy()
        else:
            df = df.merge(df_methods[k])

    return df


# %%
def get_values(
    df_filename: pd.DataFrame,
    loss='PSNR',
    data='Pred',
    val_col='Test'
) -> pd.DataFrame:
    df_res = []
    for i, row in df_filename.iterrows():
        row_res = {'Seed': row['Seed']}

        for col in df_filename:
            if col == 'Seed':
                continue
            df_model = pd.read_csv(row[col])
            df_model = df_model[(df_model['Loss'] == loss) & (df_model['Data'] == data)]
            row_res[col] = df_model[val_col].values[0]
        df_res.append(row_res)
    df_res = pd.DataFrame(df_res)

    return df_res


# %%
def analysis_results(df_res: pd.DataFrame):
    df_res = df_res.copy()

    # rename columns
    rename_map = {}
    for k in df_res:
        if k == 'Seed':
            continue

        if k.startswith('norm_2'):
            loss = 'L2'
        elif k.startswith('norm_1'):
            loss = 'L1'

        noise_level = int(float(k.split('_')[-1]))
        if noise_level == 0:
            method = 'Noise2Clean'
        else:
            method = 'Noise2Noise'

        rename_map[k] = loss + '_' + method
    df_res = df_res.rename(columns=rename_map)

    return df_res


# %%
def get_mean_and_std(df_res: pd.DataFrame):
    df_report = []

    methods = [
        'L2_Noise2Clean',
        'L2_Noise2Noise',
        'L1_Noise2Clean',
        'L1_Noise2Noise'
    ]

    for method in methods:
        vals = df_res[method].values
        df_report.append({
            'Method': method,
            'Count': len(vals),
            'Mean': np.mean(vals),
            'Std': np.std(vals),
        })
    df_report = pd.DataFrame(df_report)

    return df_report


def get_difference(df_res: pd.DataFrame):
    df_report = []

    method_pairs = [
        ['L2_Noise2Clean', 'L2_Noise2Noise'],
        ['L1_Noise2Clean', 'L1_Noise2Noise'],
    ]

    for method_pair in method_pairs:
        val1 = df_res[method_pair[0]].values
        val2 = df_res[method_pair[1]].values
        mae = np.mean(np.abs(val1 - val2))
        _, pval = scipy.stats.ttest_rel(val1, val2)
        df_report.append({
            'Method': method_pair[1],
            'MAE': mae,
            'pvalue': pval
        })
    df_report = pd.DataFrame(df_report)

    return df_report


# %%
def main(args):
    df_filename = get_filenames(args.input_dir)
    df_res = get_values(df_filename, args.loss, args.data, args.val_col)
    df_res = analysis_results(df_res)

    df_report = get_mean_and_std(df_res)
    df_report = df_report.merge(get_difference(df_res), 'outer')

    return df_report


# %%
if __name__ == '__main__':
    args = get_args()
    df = main(args)
