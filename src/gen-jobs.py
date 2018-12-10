#!/usr/bin/env python

import argparse, json, copy

parser = argparse.ArgumentParser(description=
        'Generate job files for train_rdt',
        epilog=
        'Example usage: %(prog)s --param-set n_thresholds,100 --param-range uv_range,0.4,0.9,10')

def linspace(min, max, n_values):
    step = (max - min) / (int(n_values) - 1)
    i = 0
    while i < n_values - 1:
        yield min
        min += step
        i += 1
    yield max

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

def parse_param(string):
    params = string.split(',')
    if len(params) != 2:
        msg = '%r has an incorrect number of parameters, expected 2' % string
        raise argparse.ArgumentTypeError(msg)
    return (params[0], [parse_value(params[1])])

def parse_list(string):
    params = string.split(',')
    if len(params) < 3:
        msg = '%r does not have at least two values' % string
        raise argparse.ArgumentTypeError(msg)
    return (params[0], map(lambda x:parse_value(x), params[1:]))

def parse_range(string):
    params = string.split(',')
    if len(params) != 4:
        msg = '%r has an incorrect number of parameters, expected 4' % string
        raise argparse.ArgumentTypeError(msg)

    min = parse_value(params[1])
    max = parse_value(params[2])

    n_values = int(params[3])
    if n_values < 2:
        msg = '%r specifies less than 2 values' % string
        raise argparse.ArgumentTypeError(msg)

    min_type = type(min)
    max_type = type(max)
    if (min_type != int and min_type != float) or \
       (max_type != int and max_type != float):
        msg = '%r has incorrect types for min or max values, expected numbers' % string
        raise argparse.ArgumentTypeError(msg)

    return (params[0], list(linspace(min, max, n_values)))

parser.add_argument('-o', '--output-prefix', type=str, default='out',
                    help='Prefix for out_file property')
parser.add_argument('-s', '--param-set', action='append', type=parse_param,
                    help='Set a named parameter')
parser.add_argument('-l', '--param-list', action='append', type=parse_list,
                    help='Vary a named parameter with these specific values')
parser.add_argument('-r', '--param-range', action='append', type=parse_range,
                    help='Vary a number parameter over a range. e.g. -r example,0.1,0.9,10')

args = parser.parse_args()
args_list = [args.param_set, args.param_list, args.param_range]
props = [item for sublist in [x for x in args_list if x is not None] for item in sublist]

output_id = 0
def add_jobs_recursive(jobs=[], job=None, n_prop=0):
    global output_id, props

    if n_prop >= len(props):
        return jobs

    if job is None:
        job = {}
        job['out_file'] = '%s-%05d.json' % (args.output_prefix, output_id)
        output_id += 1;
        jobs.append(job)

    prop = props[n_prop]
    job[prop[0]] = prop[1][0]
    add_jobs_recursive(jobs, job, n_prop + 1)

    for i in range(1, len(prop[1])):
        job_copy = copy.deepcopy(job)
        job_copy['out_file'] = '%s-%05d.json' % (args.output_prefix, output_id)
        output_id += 1;
        jobs.append(job_copy)
        job_copy[prop[0]] = prop[1][i]
        add_jobs_recursive(jobs, job_copy, n_prop + 1)

    return jobs

jobs = add_jobs_recursive()
print(json.dumps(jobs, indent=2))
