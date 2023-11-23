# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import math
import torch
from torch import nn
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int, queries: torch.Tensor):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward_over_time(self, x: torch.Tensor):
        pass

    def batch_mul(self, x, y):
        if len(x.shape) != len(y.shape):
            result = torch.einsum('ij,ikj->ik', [x, y])
        else:
            result = torch.einsum('ij,kj->ik', [x, y])
        return result

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size,queries)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    # rhs = rhs.t()
                    # scores = q @ rhs
                    scores = self.batch_mul(q,rhs)
                    targets = self.score(these_queries)
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        # filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_auc(
            self, queries: torch.Tensor, batch_size: int = 1000
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, begin, end)
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        all_scores, all_truth = [], []
        all_ts_ids = None
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                scores = self.forward_over_time(these_queries)
                all_scores.append(scores.cpu().numpy())
                if all_ts_ids is None:
                    if torch.cuda.is_available():
                        all_ts_ids = torch.arange(0, scores.shape[1]).cuda()[None, :]
                    else:
                        all_ts_ids = torch.arange(0, scores.shape[1])[None, :]
                assert not torch.any(torch.isinf(scores) + torch.isnan(scores)), "inf or nan scores"
                truth = (all_ts_ids <= these_queries[:, 4][:, None]) * (all_ts_ids >= these_queries[:, 3][:, None])
                all_truth.append(truth.cpu().numpy())
                b_begin += batch_size

        return np.concatenate(all_truth), np.concatenate(all_scores)

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            q = self.get_queries(queries)
            targets = self.score(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size,queries)
                scores = self.batch_mul(q,rhs)
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores.shape[1], f"fuck {scores.shape[1]} {max_to_filter}"
                        scores[i, filter_in_chunk] = -1e6
                    else:
                        scores[i, filter_out] = -1e6
                ranks += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks


class ComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    @staticmethod
    def has_time():
        return False

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), None

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)




class HTNTAttE(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(HTNTAttE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb

        # attention
        self.context_vec = nn.Embedding(self.sizes[3], 2*rank)
        self.context_vec.weight.data *= init_size
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(2*self.rank)]).cuda()

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        context_vec = self.context_vec(x[:, 3])#.view((-1, 1, 2*self.rank))
        #print(context_vec.size())
        #exit()

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time_phase, time_phase_h = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))

        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        phase_time_h = time_phase_h

        time = torch.cos(phase_time), torch.sin(phase_time)
        time_h = torch.cosh(phase_time_h), torch.sinh(phase_time_h)
        

        q_rot_t_temp = rel[0] * time[0] - rel[1] * time[1],   rel[0] * time[1] + rel[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))
        #print(q_rot_t[0].size())
        #print(q_rot_t_0.size())

        q_ref_t_temp = rel[0] * time_h[0] + rel[1] * time_h[1],   rel[0] * time_h[1] + rel[1] * time_h[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))

        #q_ref_t = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        #q_ref_t[0] = q_ref_t[0].view((-1, 1, self.rank))
        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        #print(cands[0].size())
        #print(context_vec[0].size())

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])
        #print(q_rot_t[0].size())
        #print(att_weights[0].size())

        rel_t = torch.sum(att_weights[0] * cands[0], dim=1) + rel_no_time[0], torch.sum(att_weights[1] * cands[1], dim=1) + rel_no_time[1]
        #print(lhs_t[0].size())         
        #print(rel[0].size())
        #print(rhs[0].size())
        #exit()

        return torch.sum(
            (lhs[0] * rel_t[0] - lhs[1] * rel_t[1]) * rhs[0] +
            (lhs[0] * rel_t[1] + lhs[1] * rel_t[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        #exit()
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        context_vec = self.context_vec(x[:, 3])  

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time_phase, time_phase_h = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        phase_time_h = time_phase
        time = torch.cos(phase_time), torch.sin(phase_time)
        time_h = torch.cosh(phase_time_h), torch.sinh(phase_time_h)

        q_rot_t_temp = rel[0] * time[0] - rel[1] * time[1],   rel[0] * time[1] + rel[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))
        #print(q_rot_t[0].size())
        #print(q_rot_t_0.size())

        q_ref_t_temp = rel[0] * time_h[0] + rel[1] * time_h[1],   rel[0] * time_h[1] + rel[1] * time_h[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))
        #q_ref_t = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        #q_ref_t[0] = q_ref_t[0].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        #print(cands[0].size())
        #print(context_vec[0].size())

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])
        #print(q_rot_t[0].size())
        #print(att_weights[0].size())

        rel_t = torch.sum(att_weights[0] * cands[0], dim=1) + rel_no_time[0], torch.sum(att_weights[1] * cands[1], dim=1)+ rel_no_time[1]


        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        return (
                       (lhs[0] * rel_t[0] - lhs[1] * rel_t[1]) @ right[0].t() +
                       (lhs[0] * rel_t[1] + lhs[1] * rel_t[0]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel_t[0] ** 2 + rel_t[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        context_vec = self.context_vec(queries[:, 3])


        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        time_phase, time_phase_h = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        phase_time_h = time_phase
        time = torch.cos(phase_time), torch.sin(phase_time)
        time_h = torch.cosh(phase_time_h), torch.sinh(phase_time_h)

        q_rot_t_temp = rel[0] * time[0] - rel[1] * time[1],   rel[0] * time[1] + rel[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))
        #print(q_rot_t[0].size())
        #print(q_rot_t_0.size())

        q_ref_t_temp = rel[0] * time_h[0] + rel[1] * time_h[1],   rel[0] * time_h[1] + rel[1] * time_h[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))
        #q_ref_t = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        #q_ref_t[0] = q_ref_t[0].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        #print(cands[0].size())
        #print(context_vec[0].size())

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])

        rel_t = torch.sum(att_weights[0] * cands[0], dim=1) + rel_no_time[0], torch.sum(att_weights[1] * cands[1], dim=1) + rel_no_time[1]


        return torch.cat([
            lhs[0] * rel_t[0] - lhs[1] * rel_t[1],
            lhs[0] * rel_t[1] + lhs[1] * rel_t[0]
        ], 1)



class TNTAttE(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TNTAttE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb

        # attention
        self.context_vec = nn.Embedding(self.sizes[3], 2*rank)
        self.context_vec.weight.data *= init_size
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(2*self.rank)]).cuda()

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        context_vec = self.context_vec(x[:, 3])#.view((-1, 1, 2*self.rank))
        #print(context_vec.size())
        #exit()

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time_phase, _ = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))

        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()

        time = torch.cos(phase_time), torch.sin(phase_time)


        q_rot_t_temp = rel[0] * time[0] - rel[1] * time[1],   rel[0] * time[1] + rel[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))
        #print(q_rot_t[0].size())
        #print(q_rot_t_0.size())

        q_ref_t_temp = rel[0] * time[0] + rel[1] * time[1],   rel[0] * time[1] - rel[1] * time[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))
        #q_ref_t = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        #q_ref_t[0] = q_ref_t[0].view((-1, 1, self.rank))
        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        #print(cands[0].size())
        #print(context_vec[0].size())

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])
        #print(q_rot_t[0].size())
        #print(att_weights[0].size())

        rel_t = torch.sum(att_weights[0] * cands[0], dim=1) + rel_no_time[0], torch.sum(att_weights[1] * cands[1], dim=1) + rel_no_time[1]
        #print(lhs_t[0].size())         
        #print(rel[0].size())
        #print(rhs[0].size())
        #exit()

        return torch.sum(
            (lhs[0] * rel_t[0] - lhs[1] * rel_t[1]) * rhs[0] +
            (lhs[0] * rel_t[1] + lhs[1] * rel_t[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        #exit()
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        context_vec = self.context_vec(x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time_phase, _ = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        time = torch.cos(phase_time), torch.sin(phase_time)

        q_rot_t_temp = rel[0] * time[0] - rel[1] * time[1],   rel[0] * time[1] + rel[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))
        #print(q_rot_t[0].size())
        #print(q_rot_t_0.size())

        q_ref_t_temp = rel[0] * time[0] + rel[1] * time[1],   rel[0] * time[1] - rel[1] * time[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))
        #q_ref_t = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        #q_ref_t[0] = q_ref_t[0].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        #print(cands[0].size())
        #print(context_vec[0].size())

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])
        #print(q_rot_t[0].size())
        #print(att_weights[0].size())

        rel_t = torch.sum(att_weights[0] * cands[0], dim=1) + rel_no_time[0], torch.sum(att_weights[1] * cands[1], dim=1)+ rel_no_time[1]


        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        return (
                       (lhs[0] * rel_t[0] - lhs[1] * rel_t[1]) @ right[0].t() +
                       (lhs[0] * rel_t[1] + lhs[1] * rel_t[0]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel_t[0] ** 2 + rel_t[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        context_vec = self.context_vec(queries[:, 3])


        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        time_phase, _ = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        time = torch.cos(phase_time), torch.sin(phase_time)

        q_rot_t_temp = rel[0] * time[0] - rel[1] * time[1],   rel[0] * time[1] + rel[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))
        #print(q_rot_t[0].size())
        #print(q_rot_t_0.size())

        q_ref_t_temp = rel[0] * time[0] + rel[1] * time[1],   rel[0] * time[1] - rel[1] * time[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))
        #q_ref_t = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        #q_ref_t[0] = q_ref_t[0].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        #print(cands[0].size())
        #print(context_vec[0].size())

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])

        rel_t = torch.sum(att_weights[0] * cands[0], dim=1) + rel_no_time[0], torch.sum(att_weights[1] * cands[1], dim=1) + rel_no_time[1]


        return torch.cat([
            lhs[0] * rel_t[0] - lhs[1] * rel_t[1],
            lhs[0] * rel_t[1] + lhs[1] * rel_t[0]
        ], 1)




class TAttE(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TAttE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

        # attention
        self.context_vec = nn.Embedding(self.sizes[3], 2*rank)
        self.context_vec.weight.data *= init_size
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(2*self.rank)]).cuda()

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        context_vec = self.context_vec(x[:, 3])#.view((-1, 1, 2*self.rank))
        #print(context_vec.size())
        #exit()

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time_phase, _ = time[:, :self.rank], time[:, self.rank:]
       
        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))

        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
     
        time = torch.cos(phase_time), torch.sin(phase_time)      

         
        q_rot_t_temp = lhs[0] * time[0] - lhs[1] * time[1],   lhs[0] * time[1] + lhs[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))
        #print(q_rot_t[0].size())
        #print(q_rot_t_0.size())

        q_ref_t_temp = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))
        #q_ref_t = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        #q_ref_t[0] = q_ref_t[0].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        #print(cands[0].size())
        #print(context_vec[0].size())
     
        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])
        #print(q_rot_t[0].size())
        #print(att_weights[0].size())

        lhs_t = torch.sum(att_weights[0] * cands[0], dim=1), torch.sum(att_weights[1] * cands[1], dim=1)
        #print(lhs_t[0].size())         
        #print(rel[0].size())
        #print(rhs[0].size())
        #exit()

        return torch.sum(
            (lhs_t[0] * rel[0] - lhs_t[1] * rel[1]) * rhs[0] +
            (lhs_t[0] * rel[1] + lhs_t[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        #exit()
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        context_vec = self.context_vec(x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time_phase, _ = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        time = torch.cos(phase_time), torch.sin(phase_time)

        q_rot_t_temp = lhs[0] * time[0] - lhs[1] * time[1],   lhs[0] * time[1] + lhs[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))
        #print(q_rot_t[0].size())
        #print(q_rot_t_0.size())

        q_ref_t_temp = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))
        #q_ref_t = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        #q_ref_t[0] = q_ref_t[0].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        #print(cands[0].size())
        #print(context_vec[0].size())

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])
        #print(q_rot_t[0].size())
        #print(att_weights[0].size())

        lhs_t = torch.sum(att_weights[0] * cands[0], dim=1), torch.sum(att_weights[1] * cands[1], dim=1)


        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        #rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        #full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs_t[0] * rel[0] - lhs_t[1] * rel[1]) @ right[0].t() +
                       (lhs_t[0] * rel[1] + lhs_t[1] * rel[0]) @ right[1].t()
               ), (
                   torch.sqrt(lhs_t[0] ** 2 + lhs_t[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        context_vec = self.context_vec(queries[:, 3])


        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time_phase, _ = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        time = torch.cos(phase_time), torch.sin(phase_time)

        q_rot_t_temp = lhs[0] * time[0] - lhs[1] * time[1],   lhs[0] * time[1] + lhs[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))
        #print(q_rot_t[0].size())
        #print(q_rot_t_0.size())

        q_ref_t_temp = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))
        #q_ref_t = lhs[0] * time[0] + lhs[1] * time[1],   lhs[0] * time[1] - lhs[1] * time[0]
        #q_ref_t[0] = q_ref_t[0].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        #print(cands[0].size())
        #print(context_vec[0].size())

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])
        #print(q_rot_t[0].size())
        #print(att_weights[0].size())

        lhs_t = torch.sum(att_weights[0] * cands[0], dim=1), torch.sum(att_weights[1] * cands[1], dim=1)


        return torch.cat([
            lhs_t[0] * rel[0] - lhs_t[1] * rel[1],
            lhs_t[0] * rel[1] + lhs_t[1] * rel[0]
        ], 1)

class TComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)


class HTNTComplEx2(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, use_attention = True
    ):
        super(HTNTComplEx2, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        #self.bn0 = torch.nn.BatchNorm1d(self.rank*2)

        self.no_time_emb = no_time_emb
        self.use_attention = use_attention

    @staticmethod
    def has_time():
        return True

    def calculate(self,x):
        lhs = self.embeddings[0](x[:, 0])
        #lhs = self.bn0(lhs)
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        #rhs = self.bn0(rhs)
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]

        return lhs, rhs, rt, rnt

    def score(self, x):
        # lhs = self.embeddings[0](x[:, 0])
        # #lhs = self.bn0(lhs)
        # rel = self.embeddings[1](x[:, 1])
        # rel_no_time = self.embeddings[3](x[:, 1])
        # rhs = self.embeddings[0](x[:, 2])
        # #rhs = self.bn0(rhs)
        # time = self.embeddings[2](x[:, 3])
        #
        # lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        # rel = rel[:, :self.rank], rel[:, self.rank:]
        # rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        # time = time[:, :self.rank], time[:, self.rank:]
        # rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        #
        # rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        # full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        lhs, rhs, rt, rnt = self.calculate(x)

        full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.sum(
            (lhs[0] * full_rel[0] + lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * (-rhs[1]),
            1, keepdim=True
        )

    def forward(self, x):
        # lhs = self.embeddings[0](x[:, 0])
        # #lhs = self.bn0(lhs)
        # rel = self.embeddings[1](x[:, 1])
        # rel_no_time = self.embeddings[3](x[:, 1])
        # rhs = self.embeddings[0](x[:, 2])
        # #rhs = self.bn0(rhs)
        # time = self.embeddings[2](x[:, 3])
        #
        # lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        # rel = rel[:, :self.rank], rel[:, self.rank:]
        # rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        # time = time[:, :self.rank], time[:, self.rank:]
        #
        # rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        lhs, rhs, rt, rnt = self.calculate(x)
        #rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] + rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]


        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] + lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ (-right[1].t())
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]
        
        return torch.cat([
            lhs[0] * full_rel[0] + lhs[1] * full_rel[1],
            -(lhs[1] * full_rel[0] + lhs[0] * full_rel[1])
        ], 1)


class HTNTComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, use_attention=0, att_weight=0.5, num_form='split',use_reverse=False,
    ):
        super(HTNTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        # self.bn0 = torch.nn.BatchNorm1d(self.rank*2)

        self.no_time_emb = no_time_emb
        self.use_attention = use_attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = init_size * torch.randn(self.sizes[1], self.rank)
        self.rel_vec = nn.Embedding(self.sizes[1], self.rank)
        self.rel_vec.weight.data = init_size * torch.randn(self.sizes[1], self.rank)
        self.rel_att_weight=att_weight
        self.geo_att_weight=att_weight
        self.act = nn.Softmax(dim=1)
        self.num_form = num_form
        self.use_reverse = use_reverse

    @staticmethod
    def has_time():
        return True

    def get_attention_weights(self, queries: torch.Tensor,
            batch_size: int = 1000, chunk_size: int = -1):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        rel_weights_real = torch.zeros((self.sizes[1], self.sizes[3],2))
        rel_weights_img = torch.zeros((self.sizes[1], self.sizes[3], 2))
        rel_counts = (torch.ones((self.sizes[1], self.sizes[3]))-1)
        geo_weights_real = torch.zeros((self.sizes[1], self.sizes[3],3))
        #geo_weights_img = torch.zeros((self.sizes[1], self.sizes[3], 3)).to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                #rhs = self.get_rhs(c_begin, chunk_size,queries)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    lhs, _,_,_, full_rel = self.cal_rel(these_queries)
                    _ = self.cal_num_form(lhs, full_rel, these_queries, self.num_form)

                    for i, query in enumerate(these_queries):
                        rel_weights_real[query[1],query[3]] = self.rel_att_weight[0][i].squeeze(-1).cpu() + rel_weights_real[query[1],query[3]]
                        rel_weights_img[query[1], query[3]] = self.rel_att_weight[1][i].squeeze(-1).cpu() + rel_weights_img[
                            query[1], query[3]]
                        rel_counts[query[1],query[3]] = rel_counts[query[1],query[3]] + 1
                        geo_weights_real[query[1],query[3]] = self.geo_att_weight[i].squeeze(-1).cpu() + geo_weights_real[query[1],query[3]]
                        #geo_weights_img[query[1], query[3]] = self.geo_att_weight[1][i].squeeze(-1) + geo_weights_img[
                            #query[1], query[3]]
                    b_begin += batch_size

                c_begin += chunk_size

            print(torch.sum(rel_counts))
            rel_counts[torch.where(rel_counts == 0)] = 1
            rel_counts_rel = rel_counts.unsqueeze(-1).repeat((1,1,2))
            rel_counts_geo = rel_counts.unsqueeze(-1).repeat((1, 1, 3))
            rel_weights_real, rel_weights_img, geo_weights_real = rel_weights_real / rel_counts_rel, rel_weights_img/rel_counts_rel, geo_weights_real/rel_counts_geo
        return rel_weights_real.cpu(), rel_weights_img.cpu(), geo_weights_real.cpu(), rel_counts.cpu()

    def cal_num_form(self,x,y,input,num='split'):
        if num=='split':
            return x[0] * y[0] + x[1] * y[1], x[0] * y[1]+x[1] * y[0]
        elif num == 'dual':
            return x[0]*y[0], x[0]*y[1]+x[1]*y[0]
        elif num == 'complex':
            return x[0] * y[0] - x[1] * y[1], x[0] * y[1]+x[1] * y[0]
        elif num == 'equal':
            split, dual, complex = [self.cal_num_form(x, y, input, i) for i in ['split', 'dual', 'complex']]
            return (split[0] + dual [0] + complex[0])/3, (split[1] + dual [1] + complex[1])/3
        else:
            split, dual, complex = [self.cal_num_form(x, y, input, i) for i in ['split', 'dual', 'complex']]
            if num =='rel':
                query = self.rel_vec(input[:, 1]).view((-1, 1, self.rank))
            elif num:
                query = y
            att_l = self.cal_att_num_form(split[0], dual[0], complex[0], query[0])
            att_r = complex[1]
            return att_l, att_r

    def cal_att_num_form(self, x, y, z, query):
        x = x.view((-1, 1, self.rank))
        y = y.view((-1, 1, self.rank))
        z = z.view((-1, 1, self.rank))
        query = query.view((-1, 1, self.rank))

        cands = torch.cat([x, y, z], dim=1)
            #context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
            #context_vec = self.context_vec(x[:, 0]).view((-1, 1, self.rank))
        self.geo_att_weight = torch.sum(query * cands /np.sqrt(self.rank), dim=-1, keepdim=True)
        self.geo_att_weight = self.act(self.geo_att_weight)
        #print(self.att_weight[0][:5])
        att_lhs = torch.sum(self.geo_att_weight * cands, dim=1)
        return att_lhs

    def cal_rel(self, x):
        lhs = self.embeddings[0](x[:, 0])
        # lhs = self.bn0(lhs)
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        # rhs = self.bn0(rhs)
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rwt = rt[0] + rt[3], rt[1] + rt[2]

        if self.use_attention==1:
            rwt = rwt[0].view((-1, 1, self.rank)), rwt[1].view((-1, 1, self.rank))
            rnt = rt[0].view((-1, 1, self.rank)), rnt[1].view((-1, 1, self.rank))
            cands = torch.cat([rwt[0], rnt[0]], dim=1), torch.cat([rwt[1], rnt[1]], dim=1)
            context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
            #context_vec = self.context_vec(x[:, 0]).view((-1, 1, self.rank))
            self.rel_att_weight = torch.sum(context_vec * cands[0] /np.sqrt(self.rank), dim=-1, keepdim=True), torch.sum(context_vec * cands[1] / np.sqrt(self.rank), dim=-1, keepdim=True)
            self.rel_att_weight = self.act(self.rel_att_weight[0]), self.act(self.rel_att_weight[1])
            #print(self.att_weight[0][:5])
            full_rel = torch.sum(self.rel_att_weight[0] * cands[0], dim=1), torch.sum(self.rel_att_weight[1] * cands[1], dim=1)
        # elif self.use_attention == 2:
        #     rwt = torch.exp(rwt[0]), torch.exp(rwt[1])
        #     rnt = torch.exp(rnt[0]), torch.exp(rnt[1])
        #     context = rwt[0]+rnt[0], rwt[1]+rnt[1]
        #     self.rel_att_weight = rwt[0]/context[0], rwt[1]/context[1]
        #     #print(self.att_weight[0][:5])
        #     full_rel = self.att_weight[0]*rwt[0] + (1-self.att_weight[0])*rnt[0], \
        #                self.att_weight[1]*rwt[1] + (1-self.att_weight[1])*rnt[1],
        else:
            full_rel = rwt[0] + rnt[0], rwt[1] + rnt[1]
        return lhs, rhs,rt,rnt, full_rel

    def score(self, x):

        lhs, rhs, rt, rnt, full_rel = self.cal_rel(x)

        #full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]
        real_part, img_part = self.cal_num_form(lhs,full_rel, x, self.num_form)
        if self.use_reverse:
            lhs_rel_1 = real_part+ img_part
            lhs_rel_2 = -real_part + img_part
            #lhs_rel_1, lhs_rel_2 = lhs_rel_1 + lhs_rel_2, -lhs_rel_1 + lhs_rel_2
        else:
            lhs_rel_1 = real_part
            lhs_rel_2 = img_part
        return torch.sum( lhs_rel_1 * rhs[0] + lhs_rel_2 * (-rhs[1]),  1, keepdim=True)

    def forward(self, x):
        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        lhs, rhs,rt,rnt, full_rel = self.cal_rel(x)
        # rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] + rt[3], rt[1] + rt[2]
        # full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]
        real_part, img_part = self.cal_num_form(lhs, full_rel,x, self.num_form)
        if self.use_reverse:
            lhs_rel_1 = real_part+ img_part
            lhs_rel_2 = -real_part + img_part
            #lhs_rel_1, lhs_rel_2 = lhs_rel_1 + lhs_rel_2, -lhs_rel_1 + lhs_rel_2
        else:
            lhs_rel_1 = real_part
            lhs_rel_2 = img_part
        regularizer = (
            math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
            torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
            math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((lhs_rel_1  @ right[0].t() + lhs_rel_2 @ (-right[1].t())
                ), regularizer,
                self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int,queries: torch.Tensor):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ]

    def get_queries(self, queries: torch.Tensor):

        lhs, _, rt, rnt,full_rel = self.cal_rel(queries)
        #full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]
        real_part, img_part = self.cal_num_form(lhs, full_rel, queries, self.num_form)
        if self.use_reverse:
            lhs_rel_1 = real_part+ img_part
            lhs_rel_2 = -real_part + img_part
            #lhs_rel_1, lhs_rel_2 = lhs_rel_1 + lhs_rel_2, -lhs_rel_1 + lhs_rel_2
        else:
            lhs_rel_1 = real_part
            lhs_rel_2 = img_part
        return torch.cat([lhs_rel_1,-lhs_rel_2], 1)

class HTNTComplEx4(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, use_attention=0, att_weight=0.5, num_form='split'
    ):
        super(HTNTComplEx4, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        # self.bn0 = torch.nn.BatchNorm1d(self.rank*2)

        self.no_time_emb = no_time_emb
        self.use_attention = use_attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = init_size * torch.randn(self.sizes[1], self.rank)
        self.att_weight=att_weight
        self.act = nn.Softmax(dim=1)
        self.num_form = num_form

    @staticmethod
    def has_time():
        return True

    def cal_num_form(self,x,y,num='split'):
        if num=='split':
            return x[0] * y[0] + x[1] * y[1], x[0] * y[1]+x[1] * y[0]
        elif num == 'dual':
            return x[0]*y[0], x[0]*y[1]+x[1]*y[0]
        else:
            return x[0] * y[0] - x[1] * y[1], x[0] * y[1]+x[1] * y[0]

    def cal_rel(self, x):
        lhs = self.embeddings[0](x[:, 0])
        # lhs = self.bn0(lhs)
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        # rhs = self.bn0(rhs)
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rwt = rt[0] + rt[3], rt[1] + rt[2]

        if self.use_attention==1:
            rwt = rwt[0].view((-1, 1, self.rank)), rwt[1].view((-1, 1, self.rank))
            rnt = rt[0].view((-1, 1, self.rank)), rnt[1].view((-1, 1, self.rank))
            cands = torch.cat([rwt[0], rnt[0]], dim=1), torch.cat([rwt[1], rnt[1]], dim=1)
            context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
            #context_vec = self.context_vec(x[:, 0]).view((-1, 1, self.rank))
            self.att_weight = torch.sum(context_vec * cands[0] /np.sqrt(self.rank), dim=-1, keepdim=True), torch.sum(context_vec * cands[1] / np.sqrt(self.rank), dim=-1, keepdim=True)
            self.att_weight = self.act(self.att_weight[0]), self.act(self.att_weight[1])
            #print(self.att_weight[0][:5])
            full_rel = torch.sum(self.att_weight[0] * cands[0], dim=1), torch.sum(self.att_weight[1] * cands[1], dim=1)
        elif self.use_attention == 2:
            rwt = torch.exp(rwt[0]), torch.exp(rwt[1])
            rnt = torch.exp(rnt[0]), torch.exp(rnt[1])
            context = rwt[0]+rnt[0], rwt[1]+rnt[1]
            self.att_weight = rwt[0]/context[0], rwt[1]/context[1]
            #print(self.att_weight[0][:5])
            full_rel = self.att_weight[0]*rwt[0] + (1-self.att_weight[0])*rnt[0], \
                       self.att_weight[1]*rwt[1] + (1-self.att_weight[1])*rnt[1],
        else:
            full_rel = rwt[0] + rnt[0], rwt[1] + rnt[1]
        return lhs, rhs,rt,rnt, full_rel

    def score(self, x):

        lhs, rhs, rt, rnt, full_rel = self.cal_rel(x)

        #full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]
        real_part, img_part = self.cal_num_form(lhs,full_rel,self.num_form)
        if self.num_form == 'split':
            return torch.sum( real_part * rhs[0] + img_part * (-rhs[1]),  1, keepdim=True)
        # elif self.num_form == 'dual':
        #     return torch.sum(real_part * rhs[0]), 1, keepdim=True)
        else:
            return torch.sum( real_part * rhs[0] + img_part * (rhs[1]),  1, keepdim=True)

    def forward(self, x):
        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        lhs, rhs,rt,rnt, full_rel = self.cal_rel(x)
        # rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] + rt[3], rt[1] + rt[2]
        # full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]
        real_part, img_part = self.cal_num_form(lhs, full_rel, self.num_form)
        regularizer = (
            math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
            torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
            math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

        if self.num_form == 'split':
            sum = real_part @ right[0].t() + img_part @ (-right[1].t())
        # elif self.num_form == 'dual':
        #     return torch.sum(real_part * rhs[0]), 1, keepdim=True)
        else:
            sum = real_part @ right[0].t() + img_part @ (right[1].t())

        return (sum
                , regularizer,
                self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int,queries: torch.Tensor):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ]

    def get_queries(self, queries: torch.Tensor):

        lhs, _, rt, rnt,full_rel = self.cal_rel(queries)
        #full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]
        real_part, img_part = self.cal_num_form(lhs, full_rel, self.num_form)
        if self.num_form == 'split':
            return torch.cat([real_part,-img_part], 1)
        else:
            return torch.cat([real_part,img_part], 1)

class HTNTComplEx3(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, use_attention=0, att_weight=0.5, num_form='split'
    ):
        super(HTNTComplEx3, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        # self.bn0 = torch.nn.BatchNorm1d(self.rank*2)

        self.no_time_emb = no_time_emb
        self.use_attention = use_attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = init_size * torch.randn(self.sizes[1], self.rank)
        self.att_weight=att_weight
        self.act = nn.Softmax(dim=1)
        self.num_form = num_form

    @staticmethod
    def has_time():
        return True

    def cal_num_form(self,x,y,num='split'):
        if num=='split':
            return x[0] * y[0] + x[1] * y[1], x[0] * y[1]+x[1] * y[0]
        elif num == 'dual':
            return x[0]*y[0], x[0]*y[1]+x[1]*y[0]
        else:
            return x[0] * y[0] - x[1] * y[1], x[0] * y[1]+x[1] * y[0]

    def cal_rel(self, x):
        lhs = self.embeddings[0](x[:, 0])
        # lhs = self.bn0(lhs)
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        # rhs = self.bn0(rhs)
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]

        if self.num_form == 'spilt':
            rwt = rt[0] + rt[3], rt[1] + rt[2]
        elif self.num_form == 'complex':
            rwt = rt[0] - rt[3], rt[1] + rt[2]
        else:
            rwt = rt[0] , rt[1] + rt[2]

        if self.use_attention==1:
            rwt = rwt[0].view((-1, 1, self.rank)), rwt[1].view((-1, 1, self.rank))
            rnt = rt[0].view((-1, 1, self.rank)), rnt[1].view((-1, 1, self.rank))
            cands = torch.cat([rwt[0], rnt[0]], dim=1), torch.cat([rwt[1], rnt[1]], dim=1)
            context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
            #context_vec = self.context_vec(x[:, 0]).view((-1, 1, self.rank))
            self.att_weight = torch.sum(context_vec * cands[0] /np.sqrt(self.rank), dim=-1, keepdim=True), torch.sum(context_vec * cands[1] / np.sqrt(self.rank), dim=-1, keepdim=True)
            self.att_weight = self.act(self.att_weight[0]), self.act(self.att_weight[1])
            #print(self.att_weight[0][:5])
            full_rel = torch.sum(self.att_weight[0] * cands[0], dim=1), torch.sum(self.att_weight[1] * cands[1], dim=1)
        elif self.use_attention == 2:
            rwt = torch.exp(rwt[0]), torch.exp(rwt[1])
            rnt = torch.exp(rnt[0]), torch.exp(rnt[1])
            context = rwt[0]+rnt[0], rwt[1]+rnt[1]
            self.att_weight = rwt[0]/context[0], rwt[1]/context[1]
            #print(self.att_weight[0][:5])
            full_rel = self.att_weight[0]*rwt[0] + (1-self.att_weight[0])*rnt[0], \
                       self.att_weight[1]*rwt[1] + (1-self.att_weight[1])*rnt[1],
        else:
            full_rel = rwt[0] + rnt[0], rwt[1] + rnt[1]
        return lhs, rhs,rt,rnt, full_rel

    def score(self, x):

        lhs, rhs, rt, rnt, full_rel = self.cal_rel(x)

        #full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]
        real_part, img_part = self.cal_num_form(lhs,full_rel,self.num_form)
        return torch.sum( real_part * rhs[0] + img_part * (-rhs[1]),  1, keepdim=True)

    def forward(self, x):
        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        lhs, rhs,rt,rnt, full_rel = self.cal_rel(x)
        # rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] + rt[3], rt[1] + rt[2]
        # full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]
        real_part, img_part = self.cal_num_form(lhs, full_rel, self.num_form)
        regularizer = (
            math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
            torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
            math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((real_part @ right[0].t() + img_part @ (-right[1].t())
                ), regularizer,
                self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int,queries: torch.Tensor):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ]

    def get_queries(self, queries: torch.Tensor):

        lhs, _, rt, rnt,full_rel = self.cal_rel(queries)
        #full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]
        real_part, img_part = self.cal_num_form(lhs, full_rel, self.num_form)
        return torch.cat([real_part,-img_part], 1)

class TFieldE(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, hrh_filter = False,kg_setting=True,rhsODE=True,
            hidrank=30, thidrank=2
    ):
        super(TFieldE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.hidrank = hidrank
        self.thidrank = thidrank
        self.kg_setting = kg_setting
        self.rhsODE = rhsODE
        self.hrh_filter = hrh_filter
        self.no_time_emb = no_time_emb

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, r, sparse=True)
            for (s, r) in [(sizes[0],rank), (sizes[1],self.hidrank*self.rank), (sizes[3],self.thidrank*self.rank)]  # last embedding modules contains no_time embeddings
        ])
        self.rel_hidden_embedding = nn.Sequential(
                        nn.Linear(self.rank, self.hidrank),
                        #nn.Tanh(),
                        #nn.Linear(self.hidrank,self.hidrank),
                        nn.ReLU()
            )
        self.time_hidden_embedding = nn.Sequential(
                        nn.Linear(self.rank, self.thidrank),
                        #nn.Tanh(),
                        #nn.Linear(self.hidrank,self.hidrank),
                        nn.ReLU()
            )
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        # self.bn0 = torch.nn.BatchNorm1d(self.rank*2)

    @staticmethod
    def has_time():
        return True


    def TimeODE(self, entity_emb, queries):
        hidden = self.time_hidden_embedding(entity_emb)
        time_trans = self.embeddings[2](queries[:, 3])
        time_trans = time_trans.view(time_trans.size()[0], self.rank, self.thidrank)
        if hidden.shape[0] == time_trans.shape[0]:
            timeh = torch.einsum('ijk,ik->ij', [time_trans, hidden])
        else:
            timeh = torch.einsum('ijk,mk->imj', [time_trans, hidden])
        timeh1 = torch.tanh(timeh)
        entity_time_trans = timeh1 + entity_emb

        return entity_time_trans

    def RelODE(self, entity_emb, queries):
        hidden = self.rel_hidden_embedding(entity_emb)
        rel_trans = self.embeddings[1](queries[:, 1])
        relation = rel_trans.view(rel_trans.size()[0], self.rank, self.hidrank)
        relationh = torch.einsum('ijk,ik->ij', [relation, hidden])
        relationh1 = torch.tanh(relationh)
        entity_r_trans = relationh1 + entity_emb
        return entity_r_trans

    def euc_sqdistance(x, y, eval_mode=False):
        """Compute euclidean squared distance between tensors.


        Args:
            x: torch.Tensor of shape (N1 x d)
            y: torch.Tensor of shape (N2 x d)
            eval_mode: boolean

        Returns:
            torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
            else torch.Tensor of shape N1 x N2 with all-pairs distances

        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        if eval_mode:
            if len(x.shape) == len(y.shape):
                y2 = y2.t()
                xy = x @ y.t()
            else:
                y2 = y2.squeeze(2)
                # entity_size = y.shape[1]
                x_repeat = x.unsqueeze(1)
                xy = torch.matmul(x_repeat, y.transpose(1, 2)).squeeze(1)
                # xy = torch.diagonal(xy,dim1=1,dim2=2)
        else:
            assert x.shape[0] == y.shape[0]
            xy = torch.sum(x * y, dim=-1, keepdim=True)
        score = x2 + y2 - 2 * xy
        return score

    def score(self, x):
        lhs = self.get_queries(x)
        rhs = self.embeddings[0](x[:, 2])
        if self.rhsODE:
            rhs = self.TimeODE(rhs, x)
        #full_rel = (rt[0] + rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]
        #real_part, img_part = self.cal_num_form(lhs,full_rel,self.num_form)
        return torch.sum( (lhs*rhs),  1, keepdim=True)

    def forward(self, x):
        right = self.embeddings[0].weight
        lhs = self.get_queries(x)
        rhs = self.get_rhs(0,self.sizes[0],x)
        rel = self.embeddings[1](x[:,1])
        rhs_reg = math.pow(2, 1 / 3) * torch.sqrt(rhs ** 2)
        # if self.rhsODE:
        #     rhs_reg = rhs_reg/self.sizes[0]
        # rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        regularizer = (
            math.pow(2, 1 / 3) * torch.sqrt(lhs ** 2 ),
            math.pow(2, 1 / 3) * torch.sqrt(rel ** 2),
            rhs_reg
        )
        return (self.batch_mul(lhs,rhs), regularizer,
                self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int, queries: torch.Tensor):
        rhs_e = self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ]
        if self.rhsODE:
            rhs_e = self.TimeODE(rhs_e, queries)
        return rhs_e

    def get_queries(self, queries: torch.Tensor):
        head_e_full = self.embeddings[0](queries[:, 0])

        head_time_trans = self.TimeODE(head_e_full,queries)
        head_r_trans = self.RelODE(head_time_trans,queries)
        lhs_e = head_r_trans
        return lhs_e


class TNTComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TNTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def TimeODE(self, entity_emb, queries):
        hidden = self.time_hidden_embedding(entity_emb)
        time_trans = self.embeddings[2](queries[:, 3])
        time_trans = time_trans.view(time_trans.size()[0], self.rank, self.thidrank)
        if hidden.shape[0] == time_trans.shape[0]:
            timeh = torch.einsum('ijk,ik->ij', [time_trans, hidden])
        else:
            timeh = torch.einsum('ijk,mk->imj', [time_trans, hidden])
        timeh1 = torch.tanh(timeh)
        entity_time_trans = timeh1 + entity_emb

        return entity_time_trans

    def RelODE(self, entity_emb, queries):
        hidden = self.rel_hidden_embedding(entity_emb)
        rel_trans = self.embeddings[1](queries[:, 1])
        relation = rel_trans.view(rel_trans.size()[0], self.rank, self.hidrank)
        relationh = torch.einsum('ijk,ik->ij', [relation, hidden])
        relationh1 = torch.tanh(relationh)
        entity_r_trans = relationh1 + entity_emb
        return entity_r_trans

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        #head_time_trans = self.TimeODE(lhs,queries)
        lhs = self.RelODE(lhs, queries)
        #lhs = self.RelODE(head_time_trans,queries)

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)

class AttTNTComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(AttTNTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding mod
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb

        # attention
        self.context_vec = nn.Embedding(self.sizes[3], 2*rank)
        self.context_vec.weight.data *= init_size
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(2*self.rank)]).cuda()


    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        context_vec = self.context_vec(x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        time_phase, time_phase_h = time[:, :self.rank], time[:, self.rank:]
        #rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        phase_time_h = time_phase_h
        time = torch.cos(phase_time), torch.sin(phase_time)
        time_h = torch.cosh(phase_time_h), torch.sinh(phase_time_h)


        q_rot_t_temp = rel[0] * time[0] - rel[1] * time[1],   rel[0] * time[1] + rel[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))

        q_ref_t_temp = rel[0] * time_h[0] + rel[1] * time_h[1],   rel[0] * time_h[1] + rel[1] * time_h[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)
        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])


        rrt = torch.sum(att_weights[0] * cands[0], dim=1), torch.sum(att_weights[1] * cands[1], dim=1)
        full_rel = rrt[0]  + rel_no_time[0], rrt[1] + rel_no_time[1]        

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        context_vec = self.context_vec(x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]           
        time_phase, time_phase_h = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        phase_time_h = time_phase
        time = torch.cos(phase_time), torch.sin(phase_time)
        time_h = torch.cosh(phase_time_h), torch.sinh(phase_time_h)

        q_rot_t_temp = rel[0] * time[0] - rel[1] * time[1],   rel[0] * time[1] + rel[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))

        q_ref_t_temp = rel[0] * time_h[0] + rel[1] * time_h[1],   rel[0] * time_h[1] + rel[1] * time_h[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])

        rrt = torch.sum(att_weights[0] * cands[0], dim=1), torch.sum(att_weights[1] * cands[1], dim=1)
        full_rel = rrt[0]  + rel_no_time[0], rrt[1] + rel_no_time[1]
        

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rel_no_time[0] ** 2 + rel_no_time[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        context_vec = self.context_vec(queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        time_phase, time_phase_h = time[:, :self.rank], time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        pi = 3.14159265358979323846
        phase_time = time_phase/(torch.tensor([10.0/self.rank])/pi).cuda()
        phase_time_h = time_phase
        time = torch.cos(phase_time), torch.sin(phase_time)
        time_h = torch.cosh(phase_time_h), torch.sinh(phase_time_h)

        q_rot_t_temp = rel[0] * time[0] - rel[1] * time[1],   rel[0] * time[1] + rel[1] * time[0]
        q_rot_t = q_rot_t_temp[0].view((-1, 1, self.rank)), q_rot_t_temp[1].view((-1, 1, self.rank))

        q_ref_t_temp = rel[0] * time_h[0] + rel[1] * time_h[1],   rel[0] * time_h[1] + rel[1] * time_h[0]
        q_ref_t = q_ref_t_temp[0].view((-1, 1, self.rank)), q_ref_t_temp[1].view((-1, 1, self.rank))

        cands = torch.cat((q_rot_t[0], q_ref_t[0]), dim=1), torch.cat((q_rot_t[1], q_ref_t[1]), dim=1)

        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])

        full_rel = torch.sum(att_weights[0] * cands[0], dim=1) + rel_no_time[0], torch.sum(att_weights[1] * cands[1], dim=1) + rel_no_time[1]
        

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)

class AttTNTHComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(AttTNTHComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1], sizes[3]]  # last embedding mod
        ])
        self.embeddings[0].weight.data *= init_size#entity
        self.embeddings[1].weight.data *= init_size#relation
        self.embeddings[2].weight.data *= init_size#time
        self.embeddings[3].weight.data *= init_size#relation no time
        self.embeddings[4].weight.data *= init_size

        self.no_time_emb = no_time_emb

        # attention
        self.context_vec = nn.Embedding(self.sizes[3], 2*rank)
        self.context_vec.weight.data *= init_size
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(2*self.rank)]).cuda()


    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        timec = self.embeddings[2](x[:, 3])
        timeh = self.embeddings[4](x[:, 3])
        context_vec = self.context_vec(x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        time_c = timec[:, :self.rank], timec[:, self.rank:]
        time_h = timeh[:, :self.rank], timeh[:, self.rank:]
        #rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))


        rrt_c_t = rel[0] * time_c[0] - rel[1] * time_c[1],  rel[0] * time_c[1] + rel[1] * time_c[0]
        rrt_c = rrt_c_t[0].view((-1, 1, self.rank)), rrt_c_t[1].view((-1, 1, self.rank))
        full_rel_c = rrt_c[0]  + rel_no_time[0], rrt_c[1] + rel_no_time[1]
        q_c = lhs[0] * full_rel_c[0] - lhs[1] * full_rel_c[1], lhs[1] * full_rel_c[0] + lhs[0] * full_rel_c[1]


        rrt_h_t = rel[0] * time_h[0] + rel[1] * time_h[1], rel[0] * time_h[1] + rel[1] * time_h[0]
        rrt_h = rrt_h_t[0].view((-1, 1, self.rank)), rrt_h_t[1].view((-1, 1, self.rank))

        full_rel_h = rrt_h
        q_h = lhs[0] * full_rel_h[0] + lhs[1] * full_rel_h[1], lhs[1] * full_rel_h[0] + lhs[0] * full_rel_h[1]

        cands = torch.cat((q_c[0], q_h[0]), dim=1), torch.cat((q_c[1], -q_h[1]), dim=1)
        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])

        q = torch.sum(att_weights[0] * cands[0], dim=1), torch.sum(att_weights[1] * cands[1], dim=1)        

        return torch.sum(
            (q[0]) * rhs[0] +
            (q[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        timec = self.embeddings[2](x[:, 3])
        timeh = self.embeddings[4](x[:, 3])
        context_vec = self.context_vec(x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time_c = timec[:, :self.rank], timec[:, self.rank:]
        time_h = timeh[:, :self.rank], timeh[:, self.rank:]

        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))
        rrt_c_t = rel[0] * time_c[0] - rel[1] * time_c[1],  rel[0] * time_c[1] + rel[1] * time_c[0]
        rrt_c = rrt_c_t[0].view((-1, 1, self.rank)), rrt_c_t[1].view((-1, 1, self.rank))
        full_rel_c = rrt_c[0]  + rel_no_time[0], rrt_c[1] + rel_no_time[1]
        q_c = lhs[0] * full_rel_c[0] - lhs[1] * full_rel_c[1], lhs[1] * full_rel_c[0] + lhs[0] * full_rel_c[1]


        rrt_h_t = rel[0] * time_h[0] + rel[1] * time_h[1], rel[0] * time_h[1] + rel[1] * time_h[0]
        rrt_h = rrt_h_t[0].view((-1, 1, self.rank)), rrt_h_t[1].view((-1, 1, self.rank))

        full_rel_h = rrt_h
        q_h = lhs[0] * full_rel_h[0] + lhs[1] * full_rel_h[1], lhs[1] * full_rel_h[0] + lhs[0] * full_rel_h[1]

        cands = torch.cat((q_c[0], q_h[0]), dim=1), torch.cat((q_c[1], -q_h[1]), dim=1)
        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])

        q = torch.sum(att_weights[0] * cands[0], dim=1), torch.sum(att_weights[1] * cands[1], dim=1)  

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt_c[0] ** 2 + rrt_c[1] ** 2),
           torch.sqrt(rrt_h[0] ** 2 + rrt_h[1] ** 2),
           torch.sqrt(rel_no_time[0] ** 2 + rel_no_time[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (q[0]) @ right[0].t() +
               (q[1]) @ right[1].t()
            ), regularizer,                                                                                            
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time_c = time[:, :self.rank], time[:, self.rank:]
        time_h = time[:, :self.rank], time[:, self.rank:]


        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        timec = self.embeddings[2](queries[:, 3])
        timeh = self.embeddings[4](queries[:, 3])
        context_vec = self.context_vec(queries[:, 3])

        time_c = timec[:, :self.rank], timec[:, self.rank:]
        time_h = timeh[:, :self.rank], timeh[:, self.rank:]
       

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rel_no_time = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]


        context_vec = context_vec[:, :self.rank].view((-1, 1, self.rank)), context_vec[:, self.rank:].view((-1, 1, self.rank))

        rrt_c_t = rel[0] * time_c[0] - rel[1] * time_c[1],  rel[0] * time_c[1] + rel[1] * time_c[0]
        rrt_c = rrt_c_t[0].view((-1, 1, self.rank)), rrt_c_t[1].view((-1, 1, self.rank))
        full_rel_c = rrt_c[0]  + rel_no_time[0], rrt_c[1] + rel_no_time[1]
        q_c = lhs[0] * full_rel_c[0] - lhs[1] * full_rel_c[1], lhs[1] * full_rel_c[0] + lhs[0] * full_rel_c[1]


        rrt_h_t = rel[0] * time_h[0] + rel[1] * time_h[1], rel[0] * time_h[1] + rel[1] * time_h[0]
        rrt_h = rrt_h_t[0].view((-1, 1, self.rank)), rrt_h_t[1].view((-1, 1, self.rank))

        full_rel_h = rrt_h
        q_h = lhs[0] * full_rel_h[0] + lhs[1] * full_rel_h[1], lhs[1] * full_rel_h[0] + lhs[0] * full_rel_h[1]

        cands = torch.cat((q_c[0], q_h[0]), dim=1), torch.cat((q_c[1], -q_h[1]), dim=1)
        att_weights = torch.sum(context_vec[0] * cands[0] * self.scale, dim=-1, keepdim=True), torch.sum(context_vec[1] * cands[1] * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights[0]), self.act(att_weights[1])

        q = torch.sum(att_weights[0] * cands[0], dim=1), torch.sum(att_weights[1] * cands[1], dim=1)

        return torch.cat([
            q[0],
            q[1]
        ], 1)

class TComplEx2(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, use_attention=0, att_weight=0.5, num_form='split'
    ):
        super(TComplEx2, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.model_name = "TComplEx2"

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.use_attention = use_attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = init_size * torch.randn(self.sizes[1], self.rank)
        self.att_weight=att_weight
        self.act = nn.Softmax(dim=1)
        self.num_form = num_form

    @staticmethod
    def has_time():
        return True

    def cal_num_form(self,x,y,input,num='split'):
        if num=='split':
            return x[0] * y[0] + x[1] * y[1], -(x[0] * y[1]+x[1] * y[0])
        elif num == 'dual':
            return x[0]*y[0], -(x[0]*y[1]+x[1]*y[0])
        elif num == 'complex':
            return x[0] * y[0] - x[1] * y[1], x[0] * y[1]+x[1] * y[0]
        else:
            split, dual, complex = [self.cal_num_form(x, y, input, i) for i in ['split', 'dual', 'complex']]
            if num =='rel':
                query = self.rel_vec(input[:, 1]).view((-1, 1, self.rank))
            else:
                query = y
            att_l, att_r = [self.cal_att_num_form(split[i], dual[i], complex[i], query[i]) for i in [0, 1]]
            return att_l, att_r

    def cal_att_num_form(self, x, y, z, query):

        x, y, z, query = [i.view((-1, 1, self.rank)) for i in [x,y,z, query]]
        cands = torch.cat([x,y,z], dim=1)
        #context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
        #context_vec = self.context_vec(x[:, 0]).view((-1, 1, self.rank))
        self.att_weight = torch.sum(query * cands /np.sqrt(self.rank), dim=-1, keepdim=True)
        self.att_weight = self.act(self.att_weight)
        #print(self.att_weight[0][:5])
        att_lhs = torch.sum(self.att_weight * cands, dim=1)
        return att_lhs

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        if self.num_form == 'dual':
            full_rel = rt[0], rt[1] + rt[2]
        elif self.num_form == 'complex':
            full_rel = rt[0] - rt[3], rt[1] + rt[2]
        else:
            full_rel = rt[0] + rt[3], rt[1] + rt[2]

        real_part, img_part = self.cal_num_form(lhs,full_rel,x,self.num_form)

        return torch.sum(real_part*rhs[0] + img_part*rhs[1], dim =1, keepdim=True)

        # if self.num_form == 'complex':
        #     return torch.sum(
        #         (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
        #          lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
        #         (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
        #          lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
        #         1, keepdim=True
        #     )
        # elif self.num_form == 'split':
        #     return torch.sum(
        #         (lhs[0] * rel[0] * time[0] + lhs[1] * rel[1] * time[0] +
        #          lhs[1] * rel[0] * time[1] + lhs[0] * rel[1] * time[1]) * rhs[0] +
        #         (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
        #          lhs[0] * rel[0] * time[1] + lhs[1] * rel[1] * time[1]) * -rhs[1],
        #         1, keepdim=True
        #     )
        # else:
        #     return torch.sum(
        #         (lhs[0] * rel[0] * time[0]) * rhs[0] +
        #         (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
        #          lhs[0] * rel[0] * time[1] ) * -rhs[1],
        #         1, keepdim=True
        #     )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]

        if self.num_form == 'spilt':
            full_rel = rt[0] + rt[3], rt[1] + rt[2]
        elif self.num_form == 'complex':
            full_rel = rt[0] - rt[3], rt[1] + rt[2]
        else:
            full_rel = rt[0] , rt[1] + rt[2]

        real_part, img_part = self.cal_num_form(lhs,full_rel,x,self.num_form)

        # if self.num_form == 'dual' or self.num_form == 'split':
        #     img_part = -img_part

        # if self.num_form == 'complex':
        #     real_part = lhs[0] * full_rel[0] - lhs[1] * full_rel[1]
        #     img_part = lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        # elif self.num_form == 'dual':
        #     real_part = lhs[0] * full_rel[0]
        #     img_part = -(lhs[1] * full_rel[0] + lhs[0] * full_rel[1])
        # else:
        #     real_part = lhs[0] * full_rel[0] + lhs[1] * full_rel[1]
        #     img_part = -(lhs[1] * full_rel[0] + lhs[0] * full_rel[1])

        return (
                       (real_part) @ right[0].t() +
                       (img_part) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int,queries: torch.Tensor):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ]

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]

        if self.num_form == 'spilt':
            full_rel = rt[0] + rt[3], rt[1] + rt[2]
        elif self.num_form == 'complex':
            full_rel = rt[0] - rt[3], rt[1] + rt[2]
        else:
            full_rel = rt[0] , rt[1] + rt[2]

        real_part, img_part = self.cal_num_form(lhs,full_rel,queries,self.num_form)

        return torch.cat(
            [(real_part),
             (img_part)],
            1
        )

        # if self.num_form == 'complex':
        #     return torch.cat(
        #         [(lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
        #          lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]),
        #         (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
        #          lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1])],
        #         1
        #     )
        # elif self.num_form == 'split':
        #     return torch.cat(
        #         [(lhs[0] * rel[0] * time[0] + lhs[1] * rel[1] * time[0] +
        #          lhs[1] * rel[0] * time[1] + lhs[0] * rel[1] * time[1]),
        #         -(lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
        #          lhs[0] * rel[0] * time[1] + lhs[1] * rel[1] * time[1])],
        #         1
        #     )
        # else:
        #     return torch.cat(
        #         [(lhs[0] * rel[0] * time[0]) ,
        #         -(lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
        #          lhs[0] * rel[0] * time[1] )],
        #         1
        #     )
