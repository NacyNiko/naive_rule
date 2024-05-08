# Copyright (c) Facebook, Inc. and its affiliates.

import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import json

import numpy as np

from collections import defaultdict

# DATA_PATH = pkg_resources.resource_filename('tkbc', 'data/')
DATA_PATH = 'data'

def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(timestamp)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r', encoding='utf-8')
        for line in to_read.readlines():
            instance =  line.strip().split('\t')
            lhs, rel, rhs, timestamp = instance[0], instance[1], instance[2], instance[3]
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            timestamps.add(timestamp)
            if len(instance) == 5:
                end_time = instance[4]
                timestamps.add(end_time)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    #json.dump(relations_to_id,open('ICEWS05-15_rid.json','w'),indent=2)
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}
    #json.dump(timestamps_to_id,open('ICEWS05-15_tid.json', 'w'),indent=2)
    # json.dump(entities_to_id,open('ICEWS14_eid.json', 'w'),indent=2)

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+', encoding='utf-8')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r', encoding='utf-8')
        examples = []
        for line in to_read.readlines():
            instance =  line.strip().split('\t')
            lhs, rel, rhs, ts = instance[0], instance[1], instance[2], instance[3]
            if len(instance) == 5:
                end_time = instance[4]
            try:
                if len(instance) == 4:
                    examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[ts]])
                else:
                    examples.append(
                        [entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[ts],timestamps_to_id[end_time]])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        if len(examples[0]) == 4:
            for lhs, rel, rhs, ts in examples:
                to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
                to_skip['rhs'][(lhs, rel, ts)].add(rhs)
        else:
            for lhs, rel, rhs, ts, te in examples:
                to_skip['lhs'][(rhs, rel + n_relations, ts, te)].add(lhs)  # reciprocals
                to_skip['rhs'][(lhs, rel, ts, te)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for ins in examples:
        lhs, rel, rhs, _, _ = ins
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
    datasets = ['YAGO']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        prepare_dataset(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'src_data', d
            ),
            d
        )
        # try:
        #     prepare_dataset(
        #         os.path.join(
        #             os.path.dirname(os.path.realpath(__file__)), 'src_data', d
        #         ),
        #         d
        #     )
        # except OSError as e:
        #     if e.errno == errno.EEXIST:
        #         print(e)
        #         print("File exists. skipping...")
        #     else:
        #         raise

