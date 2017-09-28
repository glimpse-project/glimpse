#!/usr/bin/env python3

import os
import sys
import json
import numpy as np

if len(sys.argv) != 3:
    print('Usage: json-to-jnt.py <joint-map> <directory>')
    sys.exit(0)

def find_files(base_dir, extensions, name=None):
    for root, dirs, files in os.walk(base_dir):
        for basename in files:
            if name and not basename.startswith(name):
                continue
            for ext in extensions:
                if basename.endswith('.' + ext):
                    filename = os.path.join(root, basename)
                    yield filename
                    break

def json_to_jnt(filename, joint_map):
    with open(filename, 'r') as f:
        joints = json.load(f);

    jnt_data = []
    bones = joints['bones']
    for joint in joint_map:
        name = joint['name'].split('.')

        if len(name) != 2:
            print("Bad bone name \"" + joint['name'] + "\", expected map " +
                  "names to all be in the form: bone_name.head or bone_name.tail");
            sys.exit(1)

        end = name[1]
        name = name[0]

        found = False
        for bone in bones:
            if bone['name'] != name:
                continue

            found = True

            if end in bone:
                point = np.array(bone[end], dtype=np.float32)
                jnt_data.append(point)
            else:
                print("Specified bone end \"%s.%s\" not found in " % (name, end, filename))
                sys.exit(1)
            break
        if found == False:
            print("Failed to find \"%s.%s\" bone in json file %s" % (name, end, filename))
            sys.exit(1)

    assert(len(jnt_data) == len(joint_map)), \
        '%d != %d' % (len(jnt_data), len(joint_map))
    return np.array(jnt_data, dtype=np.float32)

with open(sys.argv[1], 'r') as f:
    joint_map = []
    for joint_text in f.read().split('\n'):
        parts = joint_text.split(',')
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        labels = [int(x.strip()) for x in parts[1:]]
        joint_map.append({ 'name': parts[0].strip(), 'labels': labels });

    for json_file in find_files(sys.argv[2], ['json']):
        print('Processing %s' % json_file)
        jnt_data = json_to_jnt(json_file, joint_map)
        jnt_file = os.path.splitext(json_file)[0] + '.jnt'
        with open(jnt_file, 'wb') as o:
            o.write(jnt_data.tobytes())
