import re
import string
from collections import Counter
import numpy as np
from tinygrad.tensor import Tensor

def levenshtein(a, b):
  n, m = len(a), len(b)
  if n > m:
    a, b, n, m = b, a, m, n

  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)

  return current[n]

def word_error_rate(x, y):
  scores = words = 0
  for h, r in zip(x, y):
    h_list = h.split()
    r_list = r.split()
    words += len(r_list)
    scores += levenshtein(h_list, r_list)
  return float(scores) / words, float(scores), words

def one_hot(arr, num_classes=3):
  res = np.eye(num_classes)[np.array(arr).reshape(-1)]
  arr = res.reshape(list(arr.shape) + [num_classes])
  arr = arr.transpose((0, 4, 1, 2, 3)).astype(np.float32)
  return arr

def get_dice_score(prediction, target, channel_axis=1, smooth_nr=1e-6, smooth_dr=1e-6):
  reduce_axis = tuple(range(2, len(prediction.shape)))
  prediction = prediction.argmax(axis=channel_axis)
  target = np.squeeze(target, axis=channel_axis)
  prediction, target= one_hot(prediction)[:, 1:], one_hot(target)[:, 1:]
  intersection = np.sum(prediction * target, axis=reduce_axis)
  target_sum = np.sum(target, axis=reduce_axis)
  prediction_sum = np.sum(prediction, axis=reduce_axis)
  result = (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)
  return result[0]

def dice_ce_loss(y_pred, y_true, channel_axis=1, smooth_nr=1e-6, smooth_dr=1e-6):
  assert isinstance(y_pred, Tensor), "prediction must be a tensor for loss function"
  y_true = Tensor(one_hot(np.squeeze(y_true, axis=channel_axis)), requires_grad=False)
  cross_entropy = -y_true.mul(y_pred.clip(1e-10, 1).log()).mean()
  # cant use get_dice_score because it is not tensorized, and it one hot encodes the prediction
  intersection = Tensor.sum(y_pred*y_true, axis=(2, 3, 4))
  union = Tensor.sum(y_pred, axis=(2, 3, 4)) + Tensor.sum(y_true, axis=(2, 3, 4))
  dice_loss = 1 - ((2. * intersection + 1e-6) / (union + 1e-6))
  loss = (dice_loss + cross_entropy) / 2
  return loss.mean(axis=1)

def normalize_string(s):
  s = "".join(c for c in s.lower() if c not in string.punctuation)
  s = re.sub(r'\b(a|an|the)\b', ' ', s)
  return " ".join(s.split())

def f1_score(x, y):
  xt = normalize_string(x).split()
  yt = normalize_string(y).split()
  ct = Counter(xt) & Counter(yt)
  if (ns := sum(ct.values())) == 0:
    return 0.0
  p = ns / len(xt)
  r = ns / len(yt)
  return 2 * p * r / (p + r)
