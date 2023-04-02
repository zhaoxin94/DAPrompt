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
def parse_function(category_info, directory="", end_signal=None):
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

                match = category_info.search(line)
                if match and good_to_go:
                    if "file" not in output:
                        output["file"] = fpath
                    name = match.group(1)
                    acc = float(match.group(2))
                    output[name] = acc
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
    parser.add_argument("--path", "-p", type=str, default='', help="Path")

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
    category_info = re.compile(
        fr"\* class: [\d]+ \(([a-zA-Z\_]+)\).*acc: ([\.\deE+-]+)%")

    tasks = []
    all_results = []

    for directory in listdir_nohidden(base_dir, sort=True):
        directory = osp.join(base_dir, directory)
        if osp.isdir(directory):
            results = parse_function(category_info,
                                     directory=directory,
                                     end_signal=end_signal)

            all_results.append(results)

            dir_name = osp.basename(directory)
            split_names = dir_name.split('_')
            source = split_names[0].capitalize()[0]
            find_to = split_names.index('to')
            target = split_names[find_to + 1].capitalize()[0]
            task = source + '2' + target

            tasks.append(task)

    class_names = []
    for name in all_results[0]:
        class_names.append(name)
    print(class_names)

    final_results = defaultdict(list)
    for i, task in enumerate(tasks):
        results = all_results[i]
        accs = []
        for name in results:
            accs.append(results[name])
        final_results[task] = accs
    
    print("Average performance")
    for key, values in final_results.items():
        avg = np.mean(values)
        print(f"* {key}: {avg:.1f}%")
        final_results[key].append(avg)

    results_path = osp.join(base_dir, 'collect_cls.txt')
    with open(results_path, 'w') as f:
        f.write(write_now([method] + class_names + ['AVG']))
        for task in tasks:
            row = [task] + final_results[task]
            f.write(write_now(row))
