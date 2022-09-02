import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def eval_one_epoch(hint, tgan, sampler, src, tgt, ts, label, e_id, bs=32):
  val_acc, val_ap, val_f1, val_auc = [], [], [], []
  with torch.no_grad():
    tgan = tgan.eval()
    TEST_BATCH_SIZE = bs
    num_test_instance = len(src)
    # 
    b_max = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    b_min = 0
    for k in range(b_min, b_max):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
      if s_idx == e_idx:
        continue
      batch_idx = np.arange(s_idx, e_idx)
      np.random.shuffle(batch_idx)
      src_l_cut = src[batch_idx]
      tgt_l_cut = tgt[batch_idx]
      ts_l_cut = ts[batch_idx]
      e_l_cut = e_id[batch_idx] if (e_idx is not None) else None

      size = len(src_l_cut)
      _, bad_l_cut = sampler.sample(size)
      pos_prob, neg_prob = tgan.contrast(src_l_cut, tgt_l_cut, bad_l_cut, ts_l_cut, e_l_cut, test=True)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      pred_label = pred_score > 0.5
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_acc.append((pred_label == true_label).mean())
      val_ap.append(average_precision_score(true_label, pred_score))
      # val_f1.append(f1_score(true_label, pred_label))
      val_auc.append(roc_auc_score(true_label, pred_score))
  return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc)