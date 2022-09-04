import pandas as pd
from log import *
from parser import *
from eval import *
from utils import *
from train import *
from module import NAT
import resource
import torch.nn as nn
import statistics

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
DATA = args.data
NUM_HOP = args.n_hop
LEARNING_RATE = args.lr
POS_DIM = args.pos_dim
TOLERANCE = args.tolerance
VERBOSITY = args.verbosity
SEED = args.seed
TIME_DIM = args.time_dim
REPLACE_PROB = args.replace_prob
SELF_DIM = args.self_dim
NGH_DIM = args.ngh_dim
assert(NUM_HOP < 3) # only up to second hop is supported
set_random_seed(SEED)
logger, get_checkpoint_path, get_ngh_store_path, get_self_rep_path, get_prev_raw_path, best_model_path, best_model_ngh_store_path = set_up_logger(args, sys_argv)


# Load data and sanity check
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
src_l = g_df.u.values.astype(int)
tgt_l = g_df.i.values.astype(int)
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))


e_idx_l = g_df.idx.values.astype(int)
e_idx_l = np.zeros_like(e_idx_l)
label_l = g_df.label.values
ts_l = g_df.ts.values


max_idx = max(src_l.max(), tgt_l.max())
assert(np.unique(np.stack([src_l, tgt_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed
assert(n_feat.shape[0] == max_idx + 1)  # the nodes need to map one-to-one to the node feat matrix

# split and pack the data by generating valid train/val/test mask according to the "mode"
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
transductive_auc = []
transductive_ap = []
inductive_auc = []
inductive_ap = []
test_times = []
early_stoppers = []
total_time = []
for run in range(args.run):
  if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

  else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    # pick some nodes to mask (i.e. reserved for testing) for inductive setting
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(tgt_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_tgt_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_tgt_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
    all_train_val_flag = ts_l <= test_time
    inductive_train_val_flag = (ts_l <= test_time) * (none_mask_node_flag <= 0.5)
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))

  # split data according to the mask
  train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], tgt_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]
  val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], tgt_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]
  test_src_l, test_tgt_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], tgt_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
  if args.mode == 'i':
    all_train_val_src_l, all_train_val_tgt_l, all_train_val_ts_l, all_train_val_e_idx_l, all_train_val_label_l = src_l[all_train_val_flag], tgt_l[all_train_val_flag], ts_l[all_train_val_flag], e_idx_l[all_train_val_flag], label_l[all_train_val_flag]
    inductive_train_val_src_l, inductive_train_val_tgt_l, inductive_train_val_ts_l, inductive_train_val_e_idx_l, inductive_train_val_label_l = src_l[inductive_train_val_flag], tgt_l[inductive_train_val_flag], ts_l[inductive_train_val_flag], e_idx_l[inductive_train_val_flag], label_l[inductive_train_val_flag]
  train_data = train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_label_l
  val_data = val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_label_l
  train_val_data = (train_data, val_data)

  # # create random samplers to generate train/val/test instances
  train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_tgt_l, ))
  val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_tgt_l, val_tgt_l))
  if args.mode == 'i':
    all_train_val_rand_sampler = RandEdgeSampler((all_train_val_src_l, ), (all_train_val_tgt_l, ))
  test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_tgt_l, val_tgt_l, test_tgt_l))
  rand_samplers = train_rand_sampler, val_rand_sampler

  # multiprocessing memory setting
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

  # model initialization
  device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  feat_dim = n_feat.shape[1]
  e_feat_dim = e_feat.shape[1]
  time_dim = TIME_DIM
  model_dim = feat_dim + e_feat_dim + time_dim
  hidden_dim = e_feat_dim + time_dim
  num_raw = 3
  memory_dim = NGH_DIM + num_raw
  num_neighbors = [1]
  for i in range(NUM_HOP):
    num_neighbors.extend([int(NUM_NEIGHBORS[i])])
  # num_neighbors.extend([int(n) for n in NUM_NEIGHBORS]) # the 0-hop neighborhood has only 1 node

  total_start = time.time()
  nat = NAT(n_feat, e_feat, memory_dim, max_idx + 1, time_dim=TIME_DIM, pos_dim=POS_DIM, n_head=ATTN_NUM_HEADS, num_neighbors=num_neighbors, dropout=DROP_OUT,
    linear_out=args.linear_out, get_checkpoint_path=get_checkpoint_path, get_ngh_store_path=get_ngh_store_path, get_self_rep_path=get_self_rep_path, get_prev_raw_path=get_prev_raw_path, verbosity=VERBOSITY,
  n_hops=NUM_HOP, replace_prob=REPLACE_PROB, self_dim=SELF_DIM, ngh_dim=NGH_DIM, device=device)
  nat.to(device)
  nat.reset_store()

  optimizer = torch.optim.Adam(nat.parameters(), lr=LEARNING_RATE)
  criterion = torch.nn.BCELoss()
  early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

  # start train and val phases
  train_val(train_val_data, nat, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, rand_samplers, logger, model_dim, n_hop=NUM_HOP)

  # final testing
  print("_*"*50)
  if args.mode == 'i':
    nat.reset_store()
    nat.reset_self_rep()
    train_acc, train_ap, train_f1, train_auc = eval_one_epoch('test for {} nodes'.format(args.mode), nat, all_train_val_rand_sampler, all_train_val_src_l, all_train_val_tgt_l, all_train_val_ts_l, all_train_val_label_l, all_train_val_e_idx_l, bs=32)
  test_start = time.time()
  test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), nat, test_rand_sampler, test_src_l, test_tgt_l, test_ts_l, test_label_l, test_e_idx_l)
  test_end = time.time()
  logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}, time: {}'.format(args.mode, test_acc, test_auc, test_ap, test_end - test_start))
  test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
  if args.mode == 'i':
    inductive_auc.append(test_auc)
    inductive_ap.append(test_ap)
  else:
    transductive_auc.append(test_auc)
    transductive_ap.append(test_ap)
  test_times.append(test_end - test_start)
  early_stoppers.append(early_stopper.best_epoch + 1)
  # save model
  logger.info('Saving NAT model ...')
  torch.save(nat.state_dict(), best_model_path)
  logger.info('NAT model saved')

  # save one line result
  save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
  # save walk_encodings_scores
  total_end = time.time()
  print("NAT experiment statistics:")
  if args.mode == "t":
    nat_results(logger, transductive_auc, "transductive_auc")
    nat_results(logger, transductive_ap, "transductive_ap")
  else:
    nat_results(logger, inductive_auc, "inductive_auc")
    nat_results(logger, inductive_ap, "inductive_ap")
  
  nat_results(logger, test_times, "test_times")
  nat_results(logger, early_stoppers, "early_stoppers")
  total_time.append(total_end - total_start)
  nat_results(logger, total_time, "total_time")