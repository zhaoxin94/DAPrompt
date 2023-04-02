import os.path as osp
import argparse
from collections import defaultdict, OrderedDict
import re
import numpy as np

from dassl.utils.tools import listdir_nohidden, check_isfile


def write_now(row, colwidth=10):
    sep = "  "

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.1f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    return sep.join([format_val(x) for x in row]) + "\n"


# compute results across different seeds
def parse_function(*metrics, directory="", end_signal=None):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.1f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = np.std(values)
        print(f"* {key}: {avg:.1f}% +- {std:.1f}%")
        output_results[key] = avg
    print("===")

    return output_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default='', help="Method")

    args = parser.parse_args()

    ###############################################################################

    base_dir = args.path

    method = base_dir.split('/')[1]

    print('*****************************************************************')
    print(f'Extract results from {base_dir}')
    print(
        '*****************************************************************\n')

    # parse results
    end_signal = "Finish training"

    metrics = []
    metric_names = ["average"]
    for metric_name in metric_names:
        regex_str = re.compile(fr"\* {metric_name}: ([\.\deE+-]+)%")
        metric = {"name": metric_name, "regex": regex_str}
        metrics.append(metric)

    final_results = defaultdict(list)
    tasks = ['Avg']

    for directory in listdir_nohidden(base_dir, sort=True):
        directory = osp.join(base_dir, directory)
        if osp.isdir(directory):
            results = parse_function(*metrics,
                                     directory=directory,
                                     end_signal=end_signal)

            for key, value in results.items():
                final_results[key].append(value)

            dir_name = osp.basename(directory)
            split_names = dir_name.split('_')
            source = split_names[0].capitalize()[0]
            find_to = split_names.index('to')
            target = split_names[find_to + 1].capitalize()[0]
            task = source + '2' + target

            tasks.append(task)

    print("Average performance")
    for key, values in final_results.items():
        avg = np.mean(values)
        print(f"* {key}: {avg:.1f}%")
        final_results[key].insert(0, avg)

    results_path = osp.join(base_dir, 'collect_results.txt')
    with open(results_path, 'w') as f:
        f.write(write_now([method] + tasks))
        for key in metric_names:
            row = [key]
            row += list(final_results[key])
            f.write(write_now(row))