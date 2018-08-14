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
        'Example usage: $(prog)s test-1.json test-2.json')

parser.add_argument('-o', '--output', type=str,
                    help='Output path for graph visualisation')
parser.add_argument('-d', '--details', default=False, action='store_true',
                    help='Output test parameters for top results')
parser.add_argument('-n', '--rank', default=10, type=int,
                    help='The number of test results to output')

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

for filename in files:
    with open(filename, 'r') as file:
        result = json.load(file)
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
        x.append(rank_mean[1][-i][-10:-5])
        y.append(rank_mean[0][-i])
        with open(rank_mean[1][-i], 'r') as file:
            result = json.load(file)
            yerr[0].append(result['accuracy']['best'] - y[-1])
            yerr[1].append(y[-1] - result['accuracy']['worst'])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(bottom=0.20)
    ax.set_title('Label inference error')
    ax.errorbar(x, y, yerr=yerr, fmt='o')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    fig.savefig(args.output, dpi=300)
