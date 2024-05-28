import pickle
import time

from datasets import TemporalDataset
import argparse
import torch
from typing import Tuple, List, Dict
import numpy as np
import random
import pandas as pd
import os

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)

parser.add_argument(
    '--dataset', type=str, default='YAGO',
    help="Dataset name"
)

parser.add_argument(
    '--ICEWS15_rels', type=str, default='static'
)

parser.add_argument(
    '--use_valid', action='store_true'
)
args = parser.parse_args()

if args.dataset == 'ICEWS14' or args.dataset == 'ICEWS14RR':
    symmetry_rel = {53:53,103:103,115:115, 207:207, 98:98, 90:90, 162:162, 134:134, 57:57}
    reverse_rel = {142:155, 155:142, 223:170, 170:223}
elif args.dataset == 'ICEWS14TIT':
    symmetry_rel = {18:118,44:44,41:41, 77:77, 0:0, 43:43, 10:10}
    reverse_rel = {14:7, 7:14, 37:46, 46:37}
elif args.dataset == 'ICEWS18':
    symmetry_rel = {61:61, 47:47, 76:76, 59:59, 25:25, 41:41, 125:125, 17:17, 26:26}
    reverse_rel = {31: 32, 32: 31, 107: 108, 108: 107, 13:14, 14:13}
elif args.dataset == 'WIKI':
    symmetry_rel = {8:8}
    reverse_rel = {}
elif args.dataset == 'YAGO':
    symmetry_rel = {4:4}
    reverse_rel = {}
elif args.dataset == 'ICEWS05-15' or args.dataset == 'ICEWS15RR':
    if args.ICEWS15_rels == 'original':
        symmetry_rel = {224:224, 56:56, 111:111, 95:95, 174:174, 129:129, 105:105, 60:60, 128:128, 133:133, 127:127,
                        229:229, 144:144, 250:250, 11:11, 137:137, 168:168, 59:59, 208:208, 172:172, 242:242}
        reverse_rel = {125: 68, 125: 174, 142: 189, 186: 189, 167: 152, 137: 152, 59: 168, 105: 224, 105: 125, 105: 68,
                       105: 57, 181: 244, 224: 125, 224: 68, 68: 125, 174: 125, 189: 142, 189: 186, 152: 167, 152: 137,
                       168: 59, 224: 105, 125: 105, 68: 105, 57: 105, 244: 181, 125: 224, 68: 224}
    elif args.ICEWS15_rels == 'ICEWS14':
        symmetry_rel = {56:56, 111:111, 129:129, 224:224, 105:105, 95:95, 174:174, 144:144, 60:60}
        reverse_rel = {152:1367, 167:152, 244:181, 181:244}
    elif args.ICEWS15_rels == 'static':
        symmetry_rel = {56:56, 95:95, 111:111, 224:224, 174:174, 60:60, 129:129, 128:128, 105:105, 127:127}
        reverse_rel = {152:1367, 167:152, 189:186, 186:189}
elif args.dataset == 'gdelt':
    symmetry_rel = {9: 9}
    reverse_rel = {}

dataset = TemporalDataset(args.dataset)
sizes = dataset.get_shape()
train = dataset.get_train()
valid = dataset.data['valid']
train = np.vstack((train, valid))

def rank_by_freq(e_list):
    e_list = pd.DataFrame(e_list)
    e_list_count = e_list.value_counts()
    return e_list_count.index.tolist()


def merge_repeated_entities(e_list):
    e_list = list(e_list)
    set_e = list(set(e_list))
    set_e.sort(key=e_list.index)
    return set_e


def find_symmetry_temporal(instance):
    head, rel, tail, time = instance
    selected = ((train[:,1] == rel) & (train[:,2] == head) )& (train[:,3]==time)
    selected = train[selected]
    if len(selected) > 0:
        symmetry_temporal_entity = selected[:,0]
        sorted_symmetry_temporal_entity = rank_by_freq(symmetry_temporal_entity)
    else:
        sorted_symmetry_temporal_entity = []
    return sorted_symmetry_temporal_entity


def find_reverse_temporal(instance):
    head, rel, tail, time = instance
    selected = ((train[:,1] == reverse_rel[rel]) & (train[:,2] == head) )& (train[:,3]==time)
    selected = train[selected]
    if len(selected) > 0:
        reverse_temporal_entity = selected[:,0]
        sorted_reverse_temporal_entity = rank_by_freq(reverse_temporal_entity)
    else:
        sorted_reverse_temporal_entity = []
    return sorted_reverse_temporal_entity


def find_symmetry(instance):
    head, rel, tail, time = instance
    selected = ((train[:,1] == rel) & (train[:,2] == head) )
    selected = train[selected]
    if len(selected) > 0:
        symmetry_entity = selected[:,0]
        sorted_symmetry_entity = rank_by_freq(symmetry_entity)
    else:
        sorted_symmetry_entity = []
    return sorted_symmetry_entity


def find_reverse(instance):
    head, rel, tail, time = instance
    selected = ((train[:,1] == reverse_rel[rel]) & (train[:,2] == head) )
    selected = train[selected]
    if len(selected) > 0:
        reverse_entity = selected[:,0]
        sorted_reverse_entity = rank_by_freq(reverse_entity)
    else:
        sorted_reverse_entity = []
    return sorted_reverse_entity


def find_freq_entity_rel(instance):
    if len(instance) == 4:
        head, rel, tail, time = instance
    else:
        head, rel, tail, time, time_e = instance
    selected = (train[:,1] == rel) & (train[:, 0] == head)
    selected = train[selected]
    if len(selected) > 0:
        freq_entity_rel = selected[:, 2]
        sorted_freq_entity_rel = rank_by_freq(freq_entity_rel)
    else:
        sorted_freq_entity_rel = []
    return sorted_freq_entity_rel

def find_recent_entity_rel(instance,index=0):
    if len(instance) == 4:
        head, rel, tail, time = instance
    else:
        head, rel, tail, time, time_e = instance
    selected = train[:, index] == instance[index]
    selected = train[selected]
    if len(selected) > 0:
        time_gap = [abs(int(i) - int(time)) for i in selected[:,3]]
        index = np.argsort(time_gap)
        recent_selected = merge_repeated_entities(selected[index][:,2])
        sorted_freq_entity = rank_by_freq(recent_selected)
    else:
        sorted_freq_entity = []
    return sorted_freq_entity

def find_freq_entity(instance,index=0):
    if len(instance) == 4:
        head, rel, tail, time = instance
    else:
        head, rel, tail, time, time_e = instance
    selected = train[:, index] == instance[index]
    selected = train[selected]
    if len(selected) > 0:
        freq_entity = selected[:,2]
        sorted_freq_entity = rank_by_freq(freq_entity)
    else:
        sorted_freq_entity = []
    return sorted_freq_entity

def find_recent_entity(instance):
    if len(instance) == 4:
        head, rel, tail, time = instance
    else:
        head, rel, tail, time, time_e = instance
    selected = (train[:,1] == rel) & (train[:, 0] == head)
    selected = train[selected]
    if len(selected) > 0:
        time_gap = [abs(int(i) - int(time)) for i in selected[:,3]]
        index = np.argsort(time_gap)
        recent_selected = selected[index][:,2]
        recent_selected = merge_repeated_entities(recent_selected)
        sorted_freq_entity = rank_by_freq(recent_selected)
    else:
        sorted_freq_entity = []
    return sorted_freq_entity

def make_dictionary(testset, if_relation=False):
    if not os.path.exists(f'dict_{args.dataset}_{if_relation}_rhs.pkl'):
        dict = {}
        if not if_relation:
            for i, query in enumerate(testset):
                #print(if_relation, i)
                if query[0] not in dict.keys():
                    dict[query[0]] = find_freq_entity(query)
        elif if_relation == 'recent':
            for i, query in enumerate(testset):
                #print(if_relation, i)
                if query[0] not in dict.keys():
                    dict[(query[0],query[1])] = find_recent_entity(query)
        elif if_relation == 'rel':
            for i, query in enumerate(testset):
                #print(if_relation, i)
                if query[1] not in dict.keys():
                    dict[(query[1])] = find_freq_entity(query, index=1)
        elif if_relation == 'recent_rel':
            for i, query in enumerate(testset):
                # print(if_relation, i)
                if query[1] not in dict.keys():
                    dict[(query[1])] = find_recent_entity_rel(query, index=1)
        else:
            for i, query in enumerate(testset):
                #print(if_relation, i)
                if (query[0], query[1]) not in dict.keys():
                    dict[(query[0], query[1])] = find_freq_entity_rel(query)
        with open(f'dict_{args.dataset}_{if_relation}_rhs.pkl', 'wb') as f:
            pickle.dump(dict, f)


def rank_entity_rules(instance, filter_out):
    if len(instance) == 4:
        head, rel, tail, time = instance
    else:
        head, rel, tail, time, time_e = instance
    rank_entity, entity_temporal, entity_static, entity_rel = [], [], [], []
    if args.dataset in ['ICEWS14RR', 'ICEWS15RR']:
        if rel in symmetry_rel.keys():
            entity_temporal = find_symmetry_temporal(instance)
            entity_static = find_symmetry(instance)
        elif rel in reverse_rel.keys():
            entity_temporal = find_reverse_temporal(instance)
            entity_static = find_reverse(instance)

    dict_True = pickle.load(open('dict_' + str(args.dataset) + '_True_rhs.pkl', 'rb'))
    dict_False = pickle.load(open('dict_' + str(args.dataset) + '_False_rhs.pkl', 'rb'))

    freq_entity_rel = dict_True[(instance[0], instance[1])]
    freq_entity = dict_False[instance[0]]

    if args.dataset in ['YAGO','WIKI']:
        # dict_recent = pickle.load(open('dict_' + str(args.dataset) + '_recent_rhs.pkl', 'rb'))
        # entity_temporal = dict_recent[(instance[0],instance[1])][:1]
        # entity_static = freq_entity_rel
        # freq_entity_rel = freq_entity
        dict_rel = pickle.load(open('dict_' + str(args.dataset) + '_rel_rhs.pkl', 'rb'))
        entity_rel = dict_rel[instance[1]]
        # dict_rel = pickle.load(open('dict_' + str(args.dataset) + '_recent_rel_rhs.pkl', 'rb'))
        # freq_entity = dict_rel[instance[1]]

    rank_entity = entity_temporal + entity_static + freq_entity_rel + freq_entity + entity_rel
    rank_entity = merge_repeated_entities(rank_entity)
    # #filter_entities = [i for i in rank_entity if ((i not in filter_out) or (i == tail))]
    # entities = list(range(sizes[0]))
    # entities = [i for i in entities if i not in rank_entity]
    #random.shuffle(entities)
    #rank_entity = rank_entity+entities
    return rank_entity, (entity_temporal, entity_static, freq_entity_rel, freq_entity)


# def get_ranking_num(query, filters):
#     rank_entity = rank_entity_rules(query)
#     filter_out = filters[(query[0], query[1], query[3])]
#     filter_entities = [i for i in rank_entity if ((i not in filter_out) or (i == query[2]))]
#     rank = filter_entities.index(query[2]) + 1
#     return rank

def get_ranking(
):
    """
    Returns filtered ranking for each queries.
    :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
    :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
    :param batch_size: maximum number of queries processed at once
    :param chunk_size: maximum number of candidates processed at once
    :return:
    """
    queries = dataset.get_examples('test')
    ranks = np.ones(len(queries))
    rule_hits=np.zeros((len(queries),5))
    filters = dataset.to_skip['rhs']
    data_log = None
    make_dictionary(queries, False)
    make_dictionary(queries, True)
    if args.dataset in ['YAGO','WIKI']:
        make_dictionary(queries,'recent')
        make_dictionary(queries,'rel')
        make_dictionary(queries, 'recent_rel')
    for i, query in enumerate(queries):
        if len(query) == 4:
            filter_out = filters[(query[0], int(query[1]), query[3])]
        else:
            filter_out = filters[(query[0], int(query[1]), query[3], query[4])]
        rank_entity, rules = rank_entity_rules(query, filter_out)
        # filter_out = filters[(query[0], query[1], query[3])]
        # temp_dict = {'key': [query[0], query[1], query[3]], 'filter out': filter_out, 'label': [query[2]],
        #              'entity_temporal': rules[0], 'entity_static': rules[1], 'freq_entity_rel': rules[2], 'freq_entity': rules[3]}
        rule_hits[i,0] = query[2] in rules[0]
        rule_hits[i, 1] = query[2] in rules[1]
        rule_hits[i, 2] = query[2] in rules[2]
        rule_hits[i, 3] = query[2] in rules[3]
        #data_list = [temp_dict[key] for key in temp_dict]
        #ndarray = np.array(data_list, dtype=object)
        # reshaped_array = ndarray.reshape(-1, 7)
        # if data_log is None:
        #     data_log = reshaped_array
        # else:
        #     data_log = np.concatenate([data_log, reshaped_array], axis=0)

        #filter_entities = [i for i in rank_entity if ((i not in filter_out) or (i == query[2]))]
        if query[2] in rank_entity:
            rank = rank_entity.index(query[2])
            count = 0
            for j in filter_out:
                if j in rank_entity[:rank]:
                    count = count + 1
            #filter_out_index_less = [1 for i in filter_out if i in rank_entity]
            #filter_out_index_less = [i<rank for i in filter_out_index]
            #rank = rank - sum(filter_out_index_less) + 1
            rank = rank - count + 1
        else:
            rank = random.randint(len(rank_entity)+1,sizes[0])

        print(str(i) + ' ' + str(rank))
        rule_hits[i, 4] = rank
        ranks[i] = rank
    np.savetxt('rule_hit_'+str(args.dataset) + str(args.ICEWS15_rels) + f'_test_{time.time()}.csv', rule_hits.astype('int'), delimiter=',')
    # np.save(f'my_array_{args.dataset}.npy', data_log)
    return ranks

def mean_rank_max(q, ranks):
    test = q
    # ranks = ranks.to('cpu').numpy()
    test_data_no_time = pd.DataFrame([(s[0], s[1], s[2]) for s in list(test)]).astype(int)
    e_list_count = test_data_no_time.value_counts()
    freq_edge_test = list(zip(e_list_count.index, e_list_count.values))
    max_ranks = np.ones(len(freq_edge_test))
    for i in range(len(freq_edge_test)):
        edge, count = freq_edge_test[i]
        selected = (test[:, 1] == edge[1]) & (test[:, 0] == edge[0]) & (test[:, 2] == edge[2])
        rank = np.array(ranks[selected], dtype=int)
        max_ranks[i] = np.mean(1. / rank)
    return np.mean(max_ranks)


def hits_k_max(q, ranks):
    test = q
    # ranks = ranks.to('cpu').numpy()
    test_data_no_time = pd.DataFrame([(s[0], s[1], s[2]) for s in list(test)])
    e_list_count = test_data_no_time.value_counts()
    freq_edge_test = list(zip(e_list_count.index, e_list_count.values))
    hits_at = {}
    for m in [1, 3, 10]:
        max_ranks = np.ones(len(freq_edge_test))
        for i in range(len(freq_edge_test)):
            edge, count = freq_edge_test[i]
            selected = (test[:, 1] == edge[1]) & (test[:, 0] == edge[0]) & (test[:, 2] == edge[2])
            rank = np.array(ranks[selected], dtype=int)
            max_ranks[i] = np.mean(rank <= m)
        hits_at[m] = np.mean(max_ranks)
    return hits_at

if __name__ == '__main__':
    ranks = get_ranking().astype(int)
    macro = mean_rank_max(dataset.get_examples('test'), ranks)
    hits_macro = hits_k_max(dataset.get_examples('test'), ranks)
    print(f'Macro_rhs:{macro}')
    print(f'Macro_hits:{hits_macro}')
    print(ranks.mean())
    mean_reciprocal_rank = np.mean(1. / ranks)
    print(mean_reciprocal_rank)
    hits_at_1 = np.mean(ranks<=1)
    hits_at_3 = np.mean(ranks<=3)
    hits_at_10 = np.mean(ranks<=10)
    print(hits_at_1)
    print(hits_at_3)
    print(hits_at_10)

# rhs_results = pd.DataFrame(columns=['Rank mean', 'MRR', 'hits 1', 'hits 3', 'hits 10'])
# for _ in range(5):
#     ranks = get_ranking()
#     ranks = ranks.astype(int)
#     print(ranks.mean())
#     mean_reciprocal_rank = np.mean(1. / ranks)
#     print(mean_reciprocal_rank)
#     hits_at_1 = np.mean(ranks<=1)
#     hits_at_3 = np.mean(ranks<=3)
#     hits_at_10 = np.mean(ranks<=10)
#     print(hits_at_1)
#     print(hits_at_3)
#     print(hits_at_10)
#     results = [ranks.mean(), mean_reciprocal_rank, hits_at_1, hits_at_3, hits_at_10]
#     tem = pd.DataFrame(np.array(results).reshape(1, 5), columns=['Rank mean', 'MRR', 'hits 1', 'hits 3', 'hits 10'])
#     rhs_results = pd.concat([rhs_results, tem], axis=0)
# rhs_results.to_csv(f'rhs_results_{args.dataset}.csv')