#!/usr/bin/env python3

from bisect import bisect
import argparse, json

details = [ \
    'date', 'index_name', 'out_file', 'n_pixels', 'n_thresholds', \
    'threshold_range', 'threshold_power', 'n_uvs', 'uv_range', 'uv_power', \
    'max_depth', 'max_nodes', 'seed']

parser = argparse.ArgumentParser(description=
    'Compare multiple test results from test_rdt',
    epilog=
    'Example usage: compare-tests.py -i threshold_range=0.5 -i threshold_power=1 \\\n' \
    '               -x uv_range -o graph.png tree-1.json tree-2.json',
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-i', '--include', action='append', type=str,
                    help='Include trees only if this history attribute matches')
parser.add_argument('-d', '--details', default=False, action='store_true',
                    help='Output test parameters for top results')
parser.add_argument('-n', '--rank', default=10, type=int,
                    help='The number of test results to output')
parser.add_argument('-o', '--output', type=str,
                    help='Output path for graph visualisation')
parser.add_argument('-t', '--title', type=str, default='Label inference error',
                    help='Title of graph visualisation')
parser.add_argument('-x', '--xaxis', type=str,
                    help='Variable to display on the X axis from tree history')
parser.add_argument('-e', '--errorbars', default=False, action='store_true',
                    help='Display best/worst inference score as error bars')
parser.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Verbose output')

parser.add_argument('file', nargs='*',
                    help='A JSON test result output from test_rdt')

args = parser.parse_args()
files = args.file

if len(files) < 2:
    msg = 'At least 2 files must be specified'
    raise argparse.ArgumentTypeError(msg)

rank_mean = ([], [])
rank_median = ([], [])
rank_worst = ([], [])
rank_best = ([], [])

table_value_link = [(rank_mean, 'average'), (rank_median, 'median'), \
                    (rank_worst, 'worst'), (rank_best, 'best')]

def parse_value(string):
    if string.lower() == 'true':
        return True
    elif string.lower() == 'false':
        return False
    else:
        try:
            if '.' in string:
                return float(string)
            else:
                return int(string)
        except ValueError:
            return string

def isclose(a, b, rel_tol=1e-06, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

print('Filtering and ordering results...')

for filename in files:
    with open(filename, 'r') as file:
        result = json.load(file)

        # Filter out trees that don't match any of the include arguments
        if args.include:
            skip = False
            for tree_filename in result['params']['trees']:
                if skip:
                    break

                with open(tree_filename, 'r') as tree_file:
                    tree = json.load(tree_file)

                    for inc_arg in args.include:
                        if skip:
                            break
                        (inc_key, inc_value) = inc_arg.split('=')
                        inc_value = parse_value(inc_value)

                        for entry in tree['history']:
                            if skip:
                                break
                            if type(entry[inc_key]) is float:
                                if isclose(entry[inc_key], inc_value) is False:
                                    skip = True
                            elif entry[inc_key] != inc_value:
                                skip = True
                            if args.verbose and skip:
                                print('Filtering out tree', filename,
                                      'because', inc_key, '==',
                                      entry[inc_key], 'expected', inc_value)

            if skip:
                continue

        try:
            for table, value in table_value_link:
                score = result['accuracy'][value]
                index = bisect(table[0], score)
                table[0].insert(index, score)
                table[1].insert(index, filename)
        except AttributeError as e:
            print('%s:' % filename, e)
            pass

for table, value in table_value_link:
    print('\nTop %d test files for %s:' % (args.rank, value))
    top = min(args.rank + 1, len(table[0]) + 1)
    for i in range(1, top):
        print('  %2d. %f - %s' % (i, table[0][-i], table[1][-i]))
        if args.details:
            try:
                with open(table[1][-i], 'r') as file:
                    result = json.load(file)
                    for tree_filename in result['params']['trees']:
                        with open(tree_filename, 'r') as tree_file:
                            tree = json.load(tree_file)
                            for entry in tree['history']:
                                for detail in details:
                                    if detail in entry:
                                        print('    %s:' % detail, entry[detail])
            except Exception as e:
                print('%s:' % table[1][-i], e)

if args.output:
    print('\nGenerating graph...')
    import math
    import matplotlib.pyplot as plt
    top = min(args.rank + 1, len(rank_mean[0]) + 1)
    x = []
    y = []
    yerr = [[], []]
    for i in range(1, top):
        tree_path = rank_mean[1][-i]
        if args.xaxis is None:
            x.append(tree_path[-10:-5])
        y.append(rank_mean[0][-i])

        if args.errorbars or args.xaxis:
            with open(tree_path, 'r') as file:
                result = json.load(file)
                yerr[0].append(y[-1] - result['accuracy']['worst'])
                yerr[1].append(result['accuracy']['best'] - y[-1])
                if args.xaxis:
                    tree_filename = result['params']['trees'][0]
                    with open(tree_filename, 'r') as tree_file:
                        tree = json.load(tree_file)
                        x.append(tree['history'][0][args.xaxis])

    if args.errorbars is False:
        yerr = None

    if type(x[0]) is float or type(x[0]) is int:
        # Sort values by the x-axis so that the lines link up correctly
        (x, y) = [list(i) for i in zip(*sorted(zip(x, y), key=lambda p: p[0]))]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(bottom=0.20)
    ax.set_title(args.title)
    if args.xaxis:
        ax.set_xlabel(args.xaxis)
    ax.grid(linestyle=':', linewidth=0.5)
    ax.errorbar(x, y, yerr=yerr, fmt='-x', ecolor='r')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    fig.savefig(args.output, dpi=300)
