import torch
import torch.nn as nn
import logging
import time
import numpy as np
import random
import math
from GAT import GAT
from torch.utils.data import WeightedRandomSampler

class NAT(torch.nn.Module):
  def __init__(self, n_feat, e_feat, memory_dim, total_nodes, get_checkpoint_path=None, get_ngh_store_path=None, get_self_rep_path=None, get_prev_raw_path=None, time_dim=2, pos_dim=0, n_head=4, num_neighbors=['1', '32'],
      dropout=0.1, linear_out=False, verbosity=1, seed=1, n_hops=2, replace_prob=0.9, self_dim=100, ngh_dim=8, device=None):
    super(NAT, self).__init__()
    self.logger = logging.getLogger(__name__)
    self.dropout = dropout
    self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
    self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)
    self.feat_dim = self.n_feat_th.shape[1]  # node feature dimension
    self.e_feat_dim = self.e_feat_th.shape[1]  # edge feature dimension
    self.time_dim = time_dim  # default to be time feature dimension
    self.self_dim = self_dim
    self.ngh_dim = ngh_dim
    # embedding layers and encoders
    self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
    self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
    self.time_encoder = self.init_time_encoder() # fourier
    self.device = device

    self.pos_dim = pos_dim
    self.trainable_embedding = nn.Embedding(num_embeddings=64, embedding_dim=self.pos_dim) # position embedding
    
    # final projection layer
    self.linear_out = linear_out
    self.affinity_score = MergeLayer(self.feat_dim + self_dim, self.feat_dim + self_dim, self.feat_dim + self_dim, 1, non_linear=not self.linear_out) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
    self.out_layer = OutLayer(self.feat_dim + self_dim + self_dim, self.feat_dim + self_dim + self_dim, 1)
    self.get_checkpoint_path = get_checkpoint_path
    self.get_ngh_store_path = get_ngh_store_path
    self.get_self_rep_path = get_self_rep_path
    self.get_prev_raw_path = get_prev_raw_path
    self.src_idx_l_prev = self.tgt_idx_l_prev = self.cut_time_l_prev = self.e_idx_l_prev = None
    self.num_neighbors = num_neighbors
    self.n_hops = n_hops
    self.ngh_id_idx = 0
    self.e_raw_idx = 1
    self.ts_raw_idx = 2
    self.num_raw = 3

    self.ngh_rep_idx = [self.num_raw, self.num_raw + self.ngh_dim]

    self.memory_dim = memory_dim
    self.verbosity = verbosity
    
    self.attn_dim = self.feat_dim + self.ngh_dim + self.pos_dim
    self.gat = GAT(1, [n_head], [self.attn_dim, self.feat_dim], add_skip_connection=False, bias=True,
                 dropout=dropout, log_attention_weights=False)
    self.total_nodes = total_nodes
    self.replace_prob = replace_prob
    self.self_rep_linear = nn.Linear(self.self_dim + self.time_dim + self.e_feat_dim, self.self_dim, bias=False)
    self.ngh_rep_linear = nn.Linear(self.self_dim + self.time_dim + self.e_feat_dim, self.ngh_dim, bias=False)
    self.self_aggregator = self.init_self_aggregator() # RNN
    self.ngh_aggregator = self.init_ngh_aggregator() # RNN
  def set_seed(self, seed):
    self.seed = seed

  def clear_store(self):
    self.neighborhood_store = None

  def reset_store(self):
    ngh_stores = []
    for i in self.num_neighbors:
      max_e_idx = self.total_nodes * i
      raw_store = torch.zeros(max_e_idx, self.num_raw)
      hidden_store = torch.empty(max_e_idx, self.ngh_dim)
      ngh_store = torch.cat((raw_store, nn.init.xavier_uniform_(hidden_store)), -1).to(self.device)
      ngh_stores.append(ngh_store)
    self.neighborhood_store = ngh_stores
    self.self_rep = torch.zeros(self.total_nodes, self.self_dim).to(self.device)
    self.prev_raw = torch.zeros(self.total_nodes, 3).to(self.device)
  
  def get_neighborhood_store(self):
    return self.neighborhood_store

  def set_neighborhood_store(self, neighborhood_store):
    self.neighborhood_store = neighborhood_store

  def set_num_neighbors_stored(self, num_neighbors_stored):
    self.num_neighbors_stored = num_neighbors_stored

  def clear_self_rep(self):
    self.self_rep = None
    self.prev_raw = None

  def reset_self_rep(self):
    self.self_rep = torch.zeros_like(self.self_rep)
    self.prev_raw = torch.zeros_like(self.prev_raw)

  def set_self_rep(self, self_rep, prev_raw):
    self.self_rep = self_rep
    self.prev_raw = prev_raw

  def set_device(self, device):
    self.device = device

  def log_time(self, desc, start, end):
    if self.verbosity > 1:
      self.logger.info('{} for the minibatch, time eclipsed: {} seconds'.format(desc, str(end-start)))
  
  def position_bits(self, bs, hop):
    # return torch.zeros(bs * self.num_neighbors[hop], device=self.device) << hop
    return torch.ones(bs * self.num_neighbors[hop], device=self.device) << hop

  def contrast(self, src_l_cut, tgt_l_cut, bad_l_cut, cut_time_l, e_idx_l=None, test=False):
    start = time.time()
    start_t = time.time()
    batch_size = len(src_l_cut)
    
    # Move data to the GPU
    src_th = torch.from_numpy(src_l_cut).to(dtype=torch.long, device=self.device)
    tgt_th = torch.from_numpy(tgt_l_cut).to(dtype=torch.long, device=self.device)
    bad_th = torch.from_numpy(bad_l_cut).to(dtype=torch.long, device=self.device)
    
    idx_th = torch.cat((src_th, tgt_th, bad_th), 0)
    cut_time_th = torch.from_numpy(cut_time_l).to(dtype=torch.float, device=self.device)
    e_idx_th = torch.from_numpy(e_idx_l).to(dtype=torch.long, device=self.device)
    end = time.time()
    batch_idx = torch.arange(batch_size * 3, device=self.device)
    start = time.time()

    self.neighborhood_store[0][idx_th, 0] = idx_th.float()
    # n_id is the node idx of neighbors of query node
    # dense_idx is the position of each neighbors in the batch*nngh tensor
    # sprase_idx is a tensor of batch idx repeated with ngh_n timesfor each node
   
    
    h0_pos_bit = self.position_bits(3 * batch_size, hop=0)
    updated_mem_h0 = self.batch_fetch_ncaches(idx_th, cut_time_th.repeat(3), hop=0)
    updated_mem_h0_with_pos = torch.cat((updated_mem_h0, h0_pos_bit.unsqueeze(1)), -1)
    feature_dim = self.memory_dim + 1
    updated_mem = updated_mem_h0_with_pos.view(3 * batch_size, self.num_neighbors[0], -1)
    updated_mem_h1 = None
    if self.n_hops > 0:
      h1_pos_bit = self.position_bits(3 * batch_size, hop=1)
      updated_mem_h1 = self.batch_fetch_ncaches(idx_th, cut_time_th.repeat(3), hop=1)
      updated_mem_h1_with_pos = torch.cat((updated_mem_h1, h1_pos_bit.unsqueeze(1)), -1)
      updated_mem = torch.cat((
        updated_mem,
        updated_mem_h1_with_pos.view(3 * batch_size, self.num_neighbors[1], -1)), 1)
    if self.n_hops > 1:
      # second-hop N-cache access
      h2_pos_bit = self.position_bits(3 * batch_size, hop=2)
      updated_mem_h2 = torch.cat((self.batch_fetch_ncaches(idx_th, cut_time_th.repeat(3), hop=2), h2_pos_bit.unsqueeze(1)), -1)
      updated_mem = torch.cat((updated_mem, updated_mem_h2.view(3 * batch_size, self.num_neighbors[2], -1)), 1)

    updated_mem = updated_mem.view(-1, feature_dim)
    ngh_id = updated_mem[:, self.ngh_id_idx].long()
    ngh_exists = torch.nonzero(ngh_id, as_tuple=True)[0]
    ngh_count = torch.count_nonzero(ngh_id.view(3, batch_size, -1), dim=-1)

    ngh_id = ngh_id.index_select(0, ngh_exists)
    updated_mem = updated_mem.index_select(0, ngh_exists)
    src_ngh_n_th, tgt_ngh_n_th, bad_ngh_n_th = ngh_count[0], ngh_count[1], ngh_count[2]
    ngh_n_th = torch.cat((src_ngh_n_th, tgt_ngh_n_th, bad_ngh_n_th), 0)
    ori_idx = torch.repeat_interleave(idx_th, ngh_n_th)
    sparse_idx = torch.repeat_interleave(batch_idx, ngh_n_th).long()
    src_nghs = torch.sum(src_ngh_n_th)
    tgt_nghs = torch.sum(tgt_ngh_n_th)
    bad_nghs = torch.sum(bad_ngh_n_th)

    node_features = self.node_raw_embed(ngh_id)

    pos_raw = updated_mem[:, -1]
    src_pos_raw = pos_raw[0:src_nghs]
    # for the target nodes, shift all the bits by 3 to differentiate from the source nodes
    tgt_pos_raw = pos_raw[src_nghs:src_nghs + tgt_nghs] << 3
    bad_pos_raw = pos_raw[src_nghs + tgt_nghs:] << 3
    pos_raw = torch.cat((src_pos_raw, tgt_pos_raw, bad_pos_raw), -1)
    hidden_states = torch.cat((node_features, updated_mem[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]], pos_raw.unsqueeze(1)), -1)
    
    src_prev_f = hidden_states[0:src_nghs]
    tgt_prev_f = hidden_states[src_nghs:src_nghs + tgt_nghs]
    bad_prev_f = hidden_states[src_nghs + tgt_nghs:]

    src_ngh_id = ngh_id[0:src_nghs]
    tgt_ngh_id = ngh_id[src_nghs:src_nghs + tgt_nghs]
    bad_ngh_id = ngh_id[src_nghs + tgt_nghs:]
    src_sparse_idx = sparse_idx[0:src_nghs]
    src_n_sparse_idx = src_sparse_idx + batch_size
    tgt_bad_sparse_idx = sparse_idx[src_nghs:] - batch_size
    tgt_sparse_idx = sparse_idx[src_nghs:src_nghs + tgt_nghs] - batch_size
    bad_sparse_idx = sparse_idx[src_nghs + tgt_nghs:] - batch_size
    
    # joint features construction
    joint_p, ngh_and_batch_id_p = self.get_joint_feature(src_sparse_idx, tgt_sparse_idx, src_ngh_id, tgt_ngh_id, src_prev_f, tgt_prev_f)
    joint_n, ngh_and_batch_id_n = self.get_joint_feature(src_n_sparse_idx, bad_sparse_idx, src_ngh_id, bad_ngh_id, src_prev_f, bad_prev_f)
    joint_p = self.get_position_encoding(joint_p)
    joint_n = self.get_position_encoding(joint_n)
 

    features = torch.cat((joint_p, joint_n), 0)

    src_self_rep = self.updated_self_rep(src_th)
    tgt_self_rep = self.updated_self_rep(tgt_th)
    bad_self_rep = self.updated_self_rep(bad_th)

    p_score, n_score, attn_score = self.forward(ngh_and_batch_id_p, ngh_and_batch_id_n, features, batch_size, src_self_rep, tgt_self_rep, bad_self_rep)
    end = time.time()
    self.log_time('attention', start, end)
    
    self.self_rep[src_th] = src_self_rep.detach()
    self.self_rep[tgt_th] = tgt_self_rep.detach()

    self.prev_raw[src_th] = torch.stack([tgt_th, e_idx_th, cut_time_th], dim = 1)
    self.prev_raw[tgt_th] = torch.stack([src_th, e_idx_th, cut_time_th], dim = 1)


    # N-cache update
    self.update_memory(src_th, tgt_th, e_idx_th, cut_time_th, updated_mem_h0, updated_mem_h1, batch_size)
    return p_score.sigmoid(), n_score.sigmoid()
  
  def get_position_encoding(self, joint):
    if self.pos_dim == 0:
      return joint[:, :-1]
    pos_raw = joint[:, -1]
    pos_encoding = self.trainable_embedding(pos_raw.long())
    return torch.cat((joint[:, :-1], pos_encoding), -1)
    

  def updated_self_rep(self, node_id):
    self_store = self.prev_raw[node_id]
    oppo_id = self_store[:, self.ngh_id_idx].long()
    e_raw = self_store[:,self.e_raw_idx].long()
    ts_raw = self_store[:,self.ts_raw_idx]
    e_feat = self.edge_raw_embed(e_raw)
    ts_feat = self.time_encoder(ts_raw)
    prev_self_rep = self.self_rep[node_id]
    prev_oppo_rep = self.self_rep[oppo_id]
    updated_self_rep = self.self_aggregator(self.self_rep_linear(torch.cat((prev_oppo_rep, e_feat, ts_feat), -1)), prev_self_rep)
    return updated_self_rep

  def update_memory(self, src_th, tgt_th, e_idx_th, cut_time_th, updated_mem_h0, updated_mem_h1, batch_size):
    ori_idx = torch.cat((src_th, tgt_th), 0)
    cut_time_th = cut_time_th.repeat(2)
    opp_th = torch.cat((tgt_th, src_th), 0)
    e_idx_th = e_idx_th.repeat(2)
    # Update neighbors
    batch_id = torch.arange(batch_size * 2, device=self.device)
    if self.n_hops > 0:
      updated_mem_h1 = updated_mem_h1.detach()[:2 * batch_size * self.num_neighbors[1]]
      # Update second hop neighbors
      if self.n_hops > 1:
        ngh_h1_id = updated_mem_h1[:, self.ngh_id_idx].long()
        ngh_exists = torch.nonzero(ngh_h1_id, as_tuple=True)[0]
        updated_mem_h2 = updated_mem_h1.index_select(0, ngh_exists)
        ngh_count = torch.count_nonzero(ngh_h1_id.view(2 * batch_size, self.num_neighbors[1]), dim=-1)
        opp_expand_th = torch.repeat_interleave(opp_th, ngh_count, dim=0)
        self.update_ncaches(opp_expand_th, updated_mem_h2, 2)
      updated_mem_h1 = updated_mem_h1[(batch_id * self.num_neighbors[1] + self.ncache_hash(opp_th, 1))]
      ngh_id_is_match = (updated_mem_h1[:, self.ngh_id_idx] == opp_th).unsqueeze(1).repeat(1, self.memory_dim)
      updated_mem_h1 = updated_mem_h1 * ngh_id_is_match

      candidate_ncaches = torch.cat((opp_th.unsqueeze(1), e_idx_th.unsqueeze(1), cut_time_th.unsqueeze(1), updated_mem_h1[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]]), -1)
      self.update_ncaches(ori_idx, candidate_ncaches, 1)
    # Update self
    updated_mem_h0 = updated_mem_h0.detach()[:batch_size * self.num_neighbors[0] * 2]
    candidate_ncaches = torch.cat((ori_idx.unsqueeze(1), e_idx_th.unsqueeze(1), cut_time_th.unsqueeze(1), updated_mem_h0[:, self.ngh_rep_idx[0]:self.ngh_rep_idx[1]]), -1)
    self.update_ncaches(ori_idx, candidate_ncaches, 0)

  def ncache_hash(self, ngh_id, hop):
    ngh_id = ngh_id.long()
    return ((ngh_id * (self.seed % 100) + ngh_id * ngh_id * ((self.seed % 100) + 1)) % self.num_neighbors[hop]).int()

  def update_ncaches(self, self_id, candidate_ncaches, hop):
    if self.num_neighbors[hop] == 0:
      return
    ngh_id = candidate_ncaches[:, self.ngh_id_idx]
    idx = self_id * self.num_neighbors[hop] + self.ncache_hash(ngh_id, hop)
    is_occupied = torch.logical_and(self.neighborhood_store[hop][idx,self.ngh_id_idx] != 0, self.neighborhood_store[hop][idx,self.ngh_id_idx] != ngh_id)
    should_replace =  (is_occupied * torch.rand(is_occupied.shape[0], device=self.device)) < self.replace_prob
    idx *= should_replace
    idx *= ngh_id != 0
    self.neighborhood_store[hop][idx] = candidate_ncaches

  def store_memory(self, n_id, e_pos_th, ts_th, e_th, agg_p):
    prev_data = torch.cat((n_id.unsqueeze(1), e_th.unsqueeze(1), ts_th.unsqueeze(1), agg_p), -1)
    self.neighborhood_store[0][e_pos_th.long()] = prev_data

  def get_joint_neighborhood(self, src_sparse_idx, tgt_sparse_idx, src_n_id, tgt_n_id, src_hidden, tgt_hidden):
    sparse_idx = torch.cat((src_sparse_idx, tgt_sparse_idx), 0)
    n_id = torch.cat((src_n_id, tgt_n_id), 0)
    all_hidden = torch.cat((src_hidden, tgt_hidden), 0)
    feat_dim = src_hidden.shape[-1]
    key = torch.cat((sparse_idx.unsqueeze(1), n_id.unsqueeze(1)), -1) # tuple of (idx in the current batch, n_id)
    unique, inverse_idx = key.unique(return_inverse=True, dim=0)
    # SCATTER ADD FOR TS WITH INV IDX
    relative_ts = torch.zeros(unique.shape[0], feat_dim, device=self.device)
    relative_ts.scatter_add_(0, inverse_idx.unsqueeze(1).repeat(1,feat_dim), all_hidden)
    relative_ts = relative_ts.index_select(0, inverse_idx)
    assert(relative_ts.shape[0] == sparse_idx.shape[0] == all_hidden.shape[0])

    return relative_ts

  def get_joint_feature(self, src_sparse_idx, tgt_sparse_idx, src_n_id, tgt_n_id, src_hidden, tgt_hidden):
    sparse_idx = torch.cat((src_sparse_idx, tgt_sparse_idx), 0)
    n_id = torch.cat((src_n_id, tgt_n_id), 0)
    all_hidden = torch.cat((src_hidden, tgt_hidden), 0)
    feat_dim = src_hidden.shape[-1]
    key = torch.cat((n_id.unsqueeze(1), sparse_idx.unsqueeze(1)), -1) # tuple of (idx in the current batch, n_id)
    unique, inverse_idx = key.unique(return_inverse=True, dim=0)
    # SCATTER ADD FOR TS WITH INV IDX
    relative_ts = torch.zeros(unique.shape[0], feat_dim, device=self.device)
    relative_ts.scatter_add_(0, inverse_idx.unsqueeze(1).repeat(1,feat_dim), all_hidden)
    return relative_ts, unique

  def batch_fetch_ncaches(self, ori_idx, curr_time, hop):
    ngh = self.neighborhood_store[hop].view(self.total_nodes, self.num_neighbors[hop], self.memory_dim)[ori_idx].view(ori_idx.shape[0] * (self.num_neighbors[hop]), self.memory_dim)
    curr_time = curr_time.repeat_interleave(self.num_neighbors[hop])
    ngh_id = ngh[:,self.ngh_id_idx].long()
    ngh_e_raw = ngh[:,self.e_raw_idx].long()
    ngh_ts_raw = ngh[:,self.ts_raw_idx]
    prev_ngh_rep = ngh[:,self.ngh_rep_idx[0]:self.ngh_rep_idx[1]]
    e_feat = self.edge_raw_embed(ngh_e_raw)
    ts_feat = self.time_encoder(ngh_ts_raw)
    ngh_self_rep = self.self_rep[ngh_id]
    updated_self_rep = self.ngh_aggregator(self.ngh_rep_linear(torch.cat((ngh_self_rep, e_feat, ts_feat), -1)), prev_ngh_rep)
    updated_self_rep *= (ngh_ts_raw != 0).unsqueeze(1).repeat(1, self.ngh_dim)
    ori_idx = torch.repeat_interleave(ori_idx, self.num_neighbors[hop])
    # msk = ngh_ts_raw <= curr_time
    updated_mem = torch.cat((ngh[:, :self.num_raw], updated_self_rep), -1)
    # updated_mem *= msk.unsqueeze(1).repeat(1, self.memory_dim)
    return updated_mem


  def forward(self, ngh_and_batch_id_p, ngh_and_batch_id_n, feat, bs, src_self_rep=None, tgt_self_rep=None, bad_self_rep=None):
    edge_idx = torch.cat((ngh_and_batch_id_p, ngh_and_batch_id_n), dim=0).T
    embed, _, attn_score = self.gat((feat, edge_idx.long(), 2*bs))
    p_embed = embed[:bs]
    n_embed = embed[bs:2*bs]
    if src_self_rep is not None:
      assert(tgt_self_rep is not None)
      assert(bad_self_rep is not None)
      p_embed = torch.cat((p_embed, src_self_rep, tgt_self_rep), -1)
      n_embed = torch.cat((n_embed, src_self_rep, bad_self_rep), -1)
    p_score = self.out_layer(p_embed).squeeze_(dim=-1)
    n_score = self.out_layer(n_embed).squeeze_(dim=-1)
    return p_score, n_score, attn_score

  def init_time_encoder(self):
    return TimeEncode(expand_dim=self.time_dim)

  def init_self_aggregator(self):
    return FeatureEncoderGRU(self.self_dim, self.self_dim, self.dropout)

  def init_ngh_aggregator(self):
    return FeatureEncoderGRU(self.ngh_dim, self.ngh_dim, self.dropout)

class FeatureEncoderGRU(torch.nn.Module):
  def __init__(self, input_dim, ngh_dim, dropout_p=0.1):
    super(FeatureEncoderGRU, self).__init__()
    self.gru = nn.GRUCell(input_dim, ngh_dim)
    self.dropout = nn.Dropout(dropout_p)
    self.ngh_dim = ngh_dim

  def forward(self, input_features, hidden_state, use_dropout=False):
    encoded_features = self.gru(input_features, hidden_state)
    if use_dropout:
      encoded_features = self.dropout(encoded_features)
    
    # return input_features
    return encoded_features

class TimeEncode(torch.nn.Module):
  def __init__(self, expand_dim, factor=5):
    super(TimeEncode, self).__init__()

    self.time_dim = expand_dim
    self.factor = factor
    self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
    self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())


  def forward(self, ts):
    # ts: [N, 1]
    batch_size = ts.size(0)

    ts = ts.view(batch_size, 1)  # [N, 1]
    map_ts = ts * self.basis_freq.view(1, -1) # [N, time_dim]
    map_ts += self.phase.view(1, -1) # [N, time_dim]
    harmonic = torch.cos(map_ts)

    # return torch.zeros_like(ts)
    return harmonic #self.dense(harmonic)

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
    super().__init__()
    #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

    # special linear layer for motif explainability
    self.non_linear = non_linear
    if not non_linear:
      assert(dim1 == dim2)
      self.fc = nn.Linear(dim1, 1)
      torch.nn.init.xavier_normal_(self.fc1.weight)

  def forward(self, x1, x2):
    z_walk = None
    if self.non_linear:
      x = torch.cat([x1, x2], dim=-1)
      #x = self.layer_norm(x)
      h = self.act(self.fc1(x))
      z = self.fc2(h)
    else: # for explainability
      # x1, x2 shape: [B, M, F]
      x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
      z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
      z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
    return z, z_walk

class OutLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim2)
    self.fc2 = torch.nn.Linear(dim2, dim3)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    h = self.act(self.fc1(x))
    z = self.fc2(h)
    return z
