
import argparse
from builtins import breakpoint
from utils.file import read_pickle
from pathlib import Path

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from utils.pdb import register_pdb_hook
register_pdb_hook()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdirs', nargs='*')
    parser.add_argument('--x', type=str, default='n_iter')
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--log_y', action='store_true')
    parser.add_argument('--log_x', action='store_true')
    parser.add_argument('--output_file', type=str, default='bagnet_results.png')
    args = parser.parse_args()

    return args

def group(ys, xs):
    # we have to find the max and min and minimum resolution to resample the ys with a linear interpolation
    
    min_x = min([x[0] for x in xs])
    max_x = max([x[-1] for x in xs])

    pad_x = np.arange(min_x, max_x)

    pad_ys = []
    for y, x in zip(ys, xs):
        f = interpolate.interp1d(x, y, fill_value=(np.nan, y[-1]), bounds_error=False)
        y_resampled = f(pad_x)
        pad_ys.append(y_resampled)

    pad_ys = np.stack(pad_ys, 0)
    return pad_ys, pad_x


def get_plot_content(log_contents, xaxis='n_iter', top_k=1):
    ys, xs = [], []
    for log_content in log_contents:
        design_list = log_content['db']
        
        cum_design_list = []
        yarr, xarr = [], []
        solution_found = False
        for step_cnt, dsn_list in enumerate(design_list):
            cum_design_list += dsn_list
            db_sorted = sorted(cum_design_list, key=lambda x: x['cost'])
            avg_cost = np.mean([x['cost'] for x in db_sorted[:top_k]])
            if db_sorted[0]['cost'] == 0 and not solution_found:
                print(f'solution found at step {step_cnt}')
                solution_found = True

            yarr.append(avg_cost)

            if xaxis == 'n_iter':
                xarr.append(step_cnt)
            elif xaxis == 'n_sim':
                # n_sim here refers to the total number of simulations done 
                if 'n_query' in log_content:
                    xarr.append(log_content['n_query'][step_cnt])
                else:
                    xarr.append(len(cum_design_list))
            elif xaxis in ('n_nn_query', 'n_query'):
                xarr.append(log_content[xaxis][step_cnt])

        ys.append(yarr)
        xs.append(xarr)

    ys, x = group(ys, xs)

    mean_y = np.mean(ys, 0)
    margin = np.std(ys, 0) / np.sqrt(len(ys))
    plot_content = dict(
        x=x,
        y=mean_y,
        margin={
            'upper': mean_y + margin,
            'lower': mean_y - margin,
        },
    )

    return plot_content 

def main(pargs):

    plt.close()
    _, ax = plt.subplots(1,1)
    for logdir in pargs.logdirs:

        logdir = Path(logdir)

        if (logdir / 'db_time.pkl').exists():
            log_content = read_pickle(logdir / 'db_time.pkl')
            plot_content = get_plot_content([log_content], xaxis=pargs.x, top_k=pargs.top_k)
            label = f'{logdir.parent.stem}-{logdir.stem}'
        else:
            # assume this dir includes different seeds
            log_contents = []
            for seed_dir in logdir.iterdir():
                if seed_dir.is_dir():
                    log_content = read_pickle(seed_dir / 'db_time.pkl')
                    log_contents.append(log_content)
            plot_content = get_plot_content(log_contents, xaxis=pargs.x, top_k=pargs.top_k)
            label = logdir.stem
        
        ax.plot(plot_content['x'], plot_content['y'], label=label)
        print(f'[{label}] Last x value = {plot_content["x"][-1]}')

        if 'margin' in plot_content:
            ax.fill_between(x=plot_content['x'],
                            y1=plot_content['margin']['upper'],
                            y2=plot_content['margin']['lower'],
                            alpha=0.4)
        
    if pargs.log_y:
        ax.set_yscale('log')
    if pargs.log_x:
        ax.set_xscale('log')

    ax.legend()
    plt.savefig(pargs.output_file)

if __name__ == '__main__':
    main(_parse_args())

