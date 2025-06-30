# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generates bags."""
import os

import numpy as np
import pandas as pd
from scipy import linalg
from scipy import optimize
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.cluster import KMeans  # pylint: disable=g-importing-member
import tensorflow as tf


def save_isotropic_synthetic_data(data_dir, n=50000, d=32, sigma=0.5, ds_num=1):
  """Generates and saves synthetic isotropic data.

  Args:
    data_dir: Directory to save the data.
    n: Number of data points.
    d: Dimension of the data.
    sigma: Gaussian noise.
    ds_num: Dataset number.
  """
  path = f'{data_dir}/synthetic/isotropic/sigma_{sigma}/'
  os.makedirs(path, exist_ok=True)
  data = []
  for _ in range(d):
    mu = 0.0
    data.append(np.random.normal(mu, sigma, n))
  data = np.transpose(np.array(data))
  data_norm = np.linalg.norm(data, axis=1)
  for dim in range(d):
    data[:, dim] = np.divide(data[:, dim], data_norm)
  if ds_num == 1:
    theta_true = np.random.rand(
        d
    )  # weights of true halfspace passing through the origin
    theta_true = theta_true / np.sum(theta_true)
  else:
    with open(path + '1/0/theta_true.npy', 'rb') as f:
      theta_true = np.load(f)
  y_tilde = np.tensordot(data, theta_true, axes=1)
  # Adding gaussian noise to labels
  y = y_tilde + np.random.normal(0, sigma, len(y_tilde))

  data_split = np.split(np.arange(n), 5)
  for nn in range(5):
    test_data = data[data_split[nn]]
    test_y_tilde = y_tilde[data_split[nn]]
    test_y = y[data_split[nn]]
    not_nn = []
    for nnn in range(5):
      if nnn != nn:
        not_nn += list(data_split[nnn])
    train_data = data[not_nn]
    train_y_tilde = y_tilde[not_nn]
    train_y = y[not_nn]
    os.makedirs(path + f'{ds_num}/{nn}', exist_ok=True)
    with open(path + f'{ds_num}/{nn}/train_data.npy', 'wb') as f:
      np.save(f, train_data)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/train_y_tilde.npy', 'wb') as f:
      np.save(f, train_y_tilde)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/train_y.npy', 'wb') as f:
      np.save(f, train_y)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_data.npy', 'wb') as f:
      np.save(f, test_data)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_y_tilde.npy', 'wb') as f:
      np.save(f, test_y_tilde)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_y.npy', 'wb') as f:
      np.save(f, test_y)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/theta_true.npy', 'wb') as f:
      np.save(f, theta_true)  # pylint: disable=deprecated-function


def save_synthetic_nonisotropic_data(
    data_dir, n=50000, d=32, sigma=0.5, ds_num=1
):
  """Generates and saves synthetic non-isotropic data.

  Args:
    data_dir: Directory to save the data.
    n: Number of data points.
    d: Dimension of the data.
    sigma: Gaussian noise.
    ds_num: Dataset number.
  """
  path = f'{data_dir}/synthetic/non_isotropic/non_independent/sigma_{sigma}/'
  os.makedirs(path, exist_ok=True)
  # data = []
  L = np.random.standard_normal((d, d))  # pylint: disable=invalid-name
  cov_matrix = L.dot(L.T)
  data = np.random.multivariate_normal([0.0] * d, cov_matrix, n)
  data_norm = np.linalg.norm(data, axis=1)
  for dim in range(d):
    data[:, dim] = np.divide(data[:, dim], data_norm)
  if ds_num == 1:
    theta_true = np.random.rand(
        d
    )  # weights of true halfspace passing through the origin
    theta_true = theta_true / np.sum(theta_true)
  else:
    with open(path + '1/0/theta_true.npy', 'rb') as f:
      theta_true = np.load(f)
  y_tilde = np.tensordot(data, theta_true, axes=1)
  # Adding gaussian noise to labels
  y = y_tilde + np.random.normal(0, sigma, len(y_tilde))

  data_split = np.split(np.arange(n), 5)
  for nn in range(5):
    test_data = data[data_split[nn]]
    test_y_tilde = y_tilde[data_split[nn]]
    test_y = y[data_split[nn]]
    not_nn = []
    for nnn in range(5):
      if nnn != nn:
        not_nn += list(data_split[nnn])
    train_data = data[not_nn]
    train_y_tilde = y_tilde[not_nn]
    train_y = y[not_nn]
    os.makedirs(path + f'{ds_num}/{nn}', exist_ok=True)
    with open(path + f'{ds_num}/{nn}/train_data.npy', 'wb') as f:
      np.save(f, train_data)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/train_y_tilde.npy', 'wb') as f:
      np.save(f, train_y_tilde)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/train_y.npy', 'wb') as f:
      np.save(f, train_y)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_data.npy', 'wb') as f:
      np.save(f, test_data)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_y_tilde.npy', 'wb') as f:
      np.save(f, test_y_tilde)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_y.npy', 'wb') as f:
      np.save(f, test_y)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/theta_true.npy', 'wb') as f:
      np.save(f, theta_true)  # pylint: disable=deprecated-function


def save_synthetic_independent_nonisotropic_data(
    data_dir, n=50000, d=32, sigma=0.5, ds_num=1
):
  """Generates and saves synthetic non-isotropic data.

  Args:
    data_dir: Directory to save the data.
    n: Number of data points.
    d: Dimension of the data.
    sigma: Gaussian noise.
    ds_num: Dataset number.
  """
  path = f'{data_dir}/synthetic/non_isotropic/independent/sigma_{sigma}/'
  os.makedirs(path, exist_ok=True)
  # data = []
  var = np.random.uniform(1, 10, d)
  var = np.sort(var)
  cov_matrix = np.diag(var)
  data = np.random.multivariate_normal([0.0] * d, cov_matrix, n)
  data_norm = np.linalg.norm(data, axis=1)
  for dim in range(d):
    data[:, dim] = np.divide(data[:, dim], data_norm)
  if ds_num == 1:
    theta_true = np.random.rand(
        d
    )  # weights of true halfspace passing through the origin
    theta_true = theta_true / np.sum(theta_true)
  else:
    with open(path + '1/0/theta_true.npy', 'rb') as f:
      theta_true = np.load(f)
  y_tilde = np.tensordot(data, theta_true, axes=1)
  # Adding gaussian noise to labels
  y = y_tilde + np.random.normal(0, sigma, len(y_tilde))

  data_split = np.split(np.arange(n), 5)
  for nn in range(5):
    test_data = data[data_split[nn]]
    test_y_tilde = y_tilde[data_split[nn]]
    test_y = y[data_split[nn]]
    not_nn = []
    for nnn in range(5):
      if nnn != nn:
        not_nn += list(data_split[nnn])
    train_data = data[not_nn]
    train_y_tilde = y_tilde[not_nn]
    train_y = y[not_nn]
    os.makedirs(path + f'{ds_num}/{nn}', exist_ok=True)
    with open(path + f'{ds_num}/{nn}/train_data.npy', 'wb') as f:
      np.save(f, train_data)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/train_y_tilde.npy', 'wb') as f:
      np.save(f, train_y_tilde)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/train_y.npy', 'wb') as f:
      np.save(f, train_y)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_data.npy', 'wb') as f:
      np.save(f, test_data)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_y_tilde.npy', 'wb') as f:
      np.save(f, test_y_tilde)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_y.npy', 'wb') as f:
      np.save(f, test_y)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/theta_true.npy', 'wb') as f:
      np.save(f, theta_true)  # pylint: disable=deprecated-function


def save_wine_data(data_dir, wine_type):
  """Saves wine quality data.

  Args:
    data_dir: Directory to save the data.
    wine_type: Type of wine (white/red).
  """
  csv_path = f'{data_dir}/wine_quality/winequality-{wine_type}.csv'
  with open(csv_path, 'r') as f:
    df1 = pd.read_csv(f, sep=';')
  if wine_type == 'white':
    nk = 4850
  else:
    nk = 1550
  df1 = df1.sample(frac=1)[:nk]
  wine_data_dict = {'train': {}, 'test': {}}
  y = df1['quality'].to_numpy()
  data = df1.drop(columns=['quality']).to_numpy()
  scaler = preprocessing.StandardScaler().fit(data)
  data = scaler.transform(data)
  print(len(data))
  wine_data_dict['train'] = {'y': y, 'data': data}
  wine_data_dict['test'] = {
      'y': y[int(0.8 * len(data)) :],
      'data': data[int(0.8 * len(data)) :],
  }
  path = f'{data_dir}/wine_quality/{wine_type}/'
  n = wine_data_dict['train']['data'].shape[0]
  data_split = np.split(np.arange(n), 5)
  ds_num = 1
  for nn in range(5):
    val_data = wine_data_dict['train']['data'][data_split[nn]]
    val_y = wine_data_dict['train']['y'][data_split[nn]]
    val_y_tilde = wine_data_dict['train']['y'][data_split[nn]]
    not_nn = []
    for nnn in range(5):
      if nnn != nn:
        not_nn += list(data_split[nnn])
    train_data = wine_data_dict['train']['data'][not_nn]
    train_y_tilde = wine_data_dict['train']['y'][not_nn]
    train_y = wine_data_dict['train']['y'][not_nn]
    test_y = wine_data_dict['test']['y']
    test_data = wine_data_dict['test']['data']
    os.makedirs(path + f'{ds_num}/{nn}', exist_ok=True)
    with open(path + f'{ds_num}/{nn}/train_data.npy', 'wb') as f:
      np.save(f, train_data)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/train_y_tilde.npy', 'wb') as f:
      np.save(f, train_y_tilde)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/train_y.npy', 'wb') as f:
      np.save(f, train_y)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_data.npy', 'wb') as f:
      np.save(f, val_data)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_y_tilde.npy', 'wb') as f:
      np.save(f, val_y_tilde)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/test_y.npy', 'wb') as f:
      np.save(f, val_y)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/final_eval_data.npy', 'wb') as f:
      np.save(f, test_data)  # pylint: disable=deprecated-function
    with open(path + f'{ds_num}/{nn}/final_eval_y.npy', 'wb') as f:
      np.save(f, test_y)  # pylint: disable=deprecated-function


def load_data(data_dir, sigma, ds_num, fold):
  """Loads synthetic data.

  Args:
    data_dir: Directory to save the data.
    sigma: Gaussian noise.
    ds_num: Dataset number.
    fold: Fold number (CV)

  Returns:
    Train data, train labels, test data, test labels, true theta.
  """
  if sigma == 2:
    sigma = int(sigma)
  path = f'{data_dir}/sigma_{sigma}/'
  with open(path + f'{ds_num}/{fold}/theta_true.npy', 'rb') as f:
    theta_true = np.load(f)
  with open(path + f'{ds_num}/{fold}/train_data.npy', 'rb') as f:
    train_data = np.load(f)
  with open(path + f'{ds_num}/{fold}/train_y_tilde.npy', 'rb') as f:
    train_y_tilde = np.load(f)
  with open(path + f'{ds_num}/{fold}/train_y.npy', 'rb') as f:
    train_y = np.load(f)
  with open(path + f'{ds_num}/{fold}/test_data.npy', 'rb') as f:
    test_data = np.load(f)
  with open(path + f'{ds_num}/{fold}/test_y_tilde.npy', 'rb') as f:
    test_y_tilde = np.load(f)
  with open(path + f'{ds_num}/{fold}/test_y.npy', 'rb') as f:
    test_y = np.load(f)
  return (
      train_data,
      train_y_tilde,
      train_y,
      test_data,
      test_y_tilde,
      test_y,
      theta_true,
  )


def load_real_data(data_dir, ds_num, fold):
  """Loads real dataset.

  Args:
    data_dir: Directory to save the data.
    ds_num: Dataset number.
    fold: Fold number (CV)

  Returns:
    Train data, train labels, test data, test labels, eval data, eval labels.
  """
  path = data_dir + '/'
  with open(path + f'{ds_num}/{fold}/train_data.npy', 'rb') as f:
    train_data = np.load(f)
  with open(path + f'{ds_num}/{fold}/train_y_tilde.npy', 'rb') as f:
    train_y_tilde = np.load(f)
  with open(path + f'{ds_num}/{fold}/train_y.npy', 'rb') as f:
    train_y = np.load(f)
  with open(path + f'{ds_num}/{fold}/test_data.npy', 'rb') as f:
    test_data = np.load(f)
  with open(path + f'{ds_num}/{fold}/test_y_tilde.npy', 'rb') as f:
    test_y_tilde = np.load(f)
  with open(path + f'{ds_num}/{fold}/test_y.npy', 'rb') as f:
    test_y = np.load(f)
  with open(path + f'{ds_num}/{fold}/final_eval_data.npy', 'rb') as f:
    eval_data = np.load(f)
  with open(path + f'{ds_num}/{fold}/final_eval_y.npy', 'rb') as f:
    eval_y = np.load(f)

  return (
      train_data,
      train_y_tilde,
      train_y,
      test_data,
      test_y_tilde,
      test_y,
      eval_data,
      eval_y,
  )


def generate_random_superbags(
    X,  # pylint: disable=invalid-name
    y_tilde,
    y,
    k=25,
    batch_size=128,
    loss_type='bag_llp',
    non_private=True,
    epsilon=1,
    delta=1e-5,
):
  """Generates random superbags.

  Args:
    X: Features.
    y_tilde: True labels.
    y: Noisy labels.
    k: Bag size.
    batch_size: Batch size.
    loss_type: Loss type.
    non_private: Non-private - boolean.
    epsilon: Epsilon - privacy parameter.
    delta: Delta - privacy parameter.

  Returns:
    Dataset loader.
  """
  # k - bag size
  m = len(X) // (2 * k)  # num bags
  rr = len(X) % m
  ridx = np.arange(len(X) - rr)
  np.random.shuffle(ridx)
  print('m', m)
  print('X', X.shape)
  print('ridx', ridx.shape, ridx[:5])
  bag_assignment_idx = np.split(ridx, m)
  if loss_type in ['inst_mir', 'event_mir'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_term2_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    sigma_cluster_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    cluster_noise = np.random.normal(0, sigma_cluster_privacy, len(y_tilde))
    term2_noise = np.random.normal(0, sigma_term2_privacy, len(y_tilde))
    y_tilde = y_tilde + cluster_noise
    y = y + term2_noise + cluster_noise
  bag_features = []
  bag_label = []
  instance_labels = []
  y_tilde_labels = []
  event_labels = []
  for bag in bag_assignment_idx:
    bf, il = [], []  # pylint: disable=unused-variable
    ridx = np.arange(len(bag))
    np.random.shuffle(ridx)
    bf = X[bag][ridx][:k]
    il = y[bag][ridx][:k]
    yt = y_tilde[bag][ridx][:k]
    bag_features.append(bf)
    if loss_type in ['inst_mir', 'agg_mir', 'event_mir']:
      bag_label.append(il[np.random.randint(len(il))])
    elif loss_type in ['bag_llp', 'inst_llp', 'event_llp']:
      bag_label.append(sum(il) / len(il))
    instance_labels.append(il)
    y_tilde_labels.append(yt)
    event_labels.append([bag_label[-1]] * len(il))
    bf = X[bag][ridx][k:]
    il = y[bag][ridx][k:]
    yt = y_tilde[bag][ridx][k:]
    bag_features.append(bf)
    if loss_type in ['inst_mir', 'agg_mir', 'event_mir']:
      bag_label.append(il[np.random.randint(len(il))])
    elif loss_type in ['bag_llp', 'inst_llp', 'event_llp']:
      bag_label.append(sum(il) / len(il))
    instance_labels.append(il)
    y_tilde_labels.append(yt)
    event_labels.append([bag_label[-1]] * len(il))
  bag_features = np.array(bag_features)
  bag_label = np.array(bag_label)
  instance_labels = np.array(instance_labels)
  y_tilde_labels = np.array(y_tilde_labels)
  event_labels = np.array(event_labels)
  if loss_type in ['bag_llp'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (k * (epsilon**2))
    )
    bag_noise = np.random.normal(0, sigma_privacy, len(bag_label))
    bag_label = bag_label + bag_noise
  ds = tf.data.Dataset.from_tensor_slices((
      bag_features,
      bag_label,
      y_tilde_labels,
      instance_labels,
      event_labels,
  ))
  ds = ds.shuffle(len(ds))
  loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
  return loader


def generate_random_bags(
    X,  # pylint: disable=invalid-name
    y_tilde,
    y,
    k=25,
    batch_size=128,
    loss_type='inst_mir',
    non_private=True,
    epsilon=1,
    delta=1e-5,
):
  """Generates random bags.

  Args:
    X: Features.
    y_tilde: True labels.
    y: Noisy labels.
    k: Bag size.
    batch_size: Batch size.
    loss_type: Loss type.
    non_private: Non-private - boolean.
    epsilon: Epsilon - privacy parameter.
    delta: Delta - privacy parameter.

  Returns:
    Dataset loader.
  """
  # k - bag size
  m = len(X) // k  # num bags
  rr = len(X) % m
  ridx = np.arange(len(X) - rr)
  np.random.shuffle(ridx)
  print('m', m)
  print('X', X.shape)
  print('ridx', ridx.shape, ridx[:5])
  bag_assignment_idx = np.split(ridx, m)
  # if len(X)%m == 0:
  #   bag_assignment_idx = np.split(ridx, m)
  # else:
  #   bag_assignment_idx = np.array_split(ridx, m+1)[:-1]
  if loss_type in ['inst_mir', 'event_mir'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_term2_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    sigma_cluster_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    cluster_noise = np.random.normal(0, sigma_cluster_privacy, len(y_tilde))
    term2_noise = np.random.normal(0, sigma_term2_privacy, len(y_tilde))
    y_tilde = y_tilde + cluster_noise
    y = y + term2_noise + cluster_noise
  bag_features = []
  bag_label = []
  instance_labels = []
  y_tilde_labels = []
  event_labels = []
  for bag in bag_assignment_idx:
    bf, il = [], []  # pylint: disable=unused-variable
    bf = X[bag]
    il = y[bag]
    yt = y_tilde[bag]

    bag_features.append(bf)
    if loss_type in ['inst_mir', 'agg_mir', 'event_mir']:
      bag_label.append(il[np.random.randint(len(il))])
    elif loss_type in ['bag_llp', 'inst_llp', 'event_llp']:
      bag_label.append(sum(il) / len(il))
    instance_labels.append(il)
    y_tilde_labels.append(yt)
    event_labels.append([bag_label[-1]] * len(il))
  bag_features = np.array(bag_features)
  bag_label = np.array(bag_label)
  instance_labels = np.array(instance_labels)
  y_tilde_labels = np.array(y_tilde_labels)
  event_labels = np.array(event_labels)
  if loss_type in ['bag_llp'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (k * (epsilon**2))
    )
    bag_noise = np.random.normal(0, sigma_privacy, len(bag_label))
    bag_label = bag_label + bag_noise
  ds = tf.data.Dataset.from_tensor_slices((
      bag_features,
      bag_label,
      y_tilde_labels,
      instance_labels,
      event_labels,
  ))
  ds = ds.shuffle(len(ds))
  loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
  return loader


def generate_label_kmeans_superbags(
    X,  # pylint: disable=invalid-name
    y_tilde,
    y,
    k=25,
    batch_size=128,
    loss_type='inst_mir',
    non_private=True,
    epsilon=1,
    delta=1e-5,
):
  """Generates label kmeans superbags.

  Args:
    X: Features.
    y_tilde: True labels.
    y: Noisy labels.
    k: Bag size.
    batch_size: Batch size.
    loss_type: Loss type.
    non_private: Non-private - boolean.
    epsilon: Epsilon - privacy parameter.
    delta: Delta - privacy parameter.

  Returns:
    Dataset loader.
  """
  # k - bag size
  m = len(X) // (2 * k)  # num bags
  print('m', m, 'len(X)', len(X), 'k', k)
  if loss_type in ['inst_mir', 'event_mir'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_term2_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    sigma_cluster_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    cluster_noise = np.random.normal(0, sigma_cluster_privacy, len(y_tilde))
    term2_noise = np.random.normal(0, sigma_term2_privacy, len(y_tilde))
    y_tilde = y_tilde + cluster_noise
    y = y + term2_noise + cluster_noise
  sorted_idx = np.argsort(y)
  X_sorted = X[sorted_idx]  # pylint: disable=invalid-name
  y_sorted = y[sorted_idx]
  y_tilde_sorted = y_tilde[sorted_idx]
  bag_features = []
  bag_label = []
  instance_labels = []
  y_tilde_labels = []
  event_labels = []
  for i in range(0, m):
    il_super = y_sorted[i * 2 * k : (i + 1) * 2 * k]
    yt_super = y_tilde_sorted[i * 2 * k : (i + 1) * 2 * k]
    x_super = X_sorted[i * 2 * k : (i + 1) * 2 * k]
    ridx = np.arange(len(il_super))
    np.random.shuffle(ridx)
    il = il_super[ridx][:k]
    yt = yt_super[ridx][:k]
    x_ = x_super[ridx][:k]
    bag_features.append(x_)
    if loss_type in ['inst_mir', 'agg_mir', 'event_mir']:
      bag_label.append(il[np.random.randint(len(il))])
    elif loss_type in ['bag_llp', 'inst_llp', 'event_llp']:
      bag_label.append(sum(il) / len(il))
    instance_labels.append(il)
    y_tilde_labels.append(yt)
    event_labels.append([bag_label[-1]] * len(il))
    il = il_super[ridx][k:]
    yt = yt_super[ridx][k:]
    x_ = x_super[ridx][k:]
    bag_features.append(x_)
    if loss_type in ['inst_mir', 'agg_mir', 'event_mir']:
      bag_label.append(il[np.random.randint(len(il))])
    elif loss_type in ['bag_llp', 'inst_llp', 'event_llp']:
      bag_label.append(sum(il) / len(il))
    instance_labels.append(il)
    y_tilde_labels.append(yt)
    event_labels.append([bag_label[-1]] * len(il))
  bag_features = np.array(bag_features)
  bag_label = np.array(bag_label)
  if loss_type in ['bag_llp'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (k * (epsilon**2))
    )
    bag_noise = np.random.normal(0, sigma_privacy, len(bag_label))
    bag_label = bag_label + bag_noise
  instance_labels = np.array(instance_labels)
  event_labels = np.array(event_labels)
  ds = tf.data.Dataset.from_tensor_slices((
      bag_features,
      bag_label,
      y_tilde_labels,
      instance_labels,
      event_labels,
  ))
  ds = ds.shuffle(len(ds))
  loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
  return loader


def generate_label_kmeans_bags(
    X,  # pylint: disable=invalid-name
    y_tilde,
    y,
    k=25,
    batch_size=128,
    loss_type='inst_mir',
    non_private=True,
    epsilon=1,
    delta=1e-5,
):
  """Generates label kmeans bags.

  Args:
    X: Features.
    y_tilde: True labels.
    y: Noisy labels.
    k: Bag size.
    batch_size: Batch size.
    loss_type: Loss type.
    non_private: Non-private - boolean.
    epsilon: Epsilon - privacy parameter.
    delta: Delta - privacy parameter.

  Returns:
    Dataset loader.
  """
  # k - bag size
  m = len(X) // k  # num bags
  print('m', m, 'len(X)', len(X), 'k', k)
  if loss_type in ['inst_mir', 'event_mir'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_term2_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    sigma_cluster_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    cluster_noise = np.random.normal(0, sigma_cluster_privacy, len(y_tilde))
    term2_noise = np.random.normal(0, sigma_term2_privacy, len(y_tilde))
    y_tilde = y_tilde + cluster_noise
    y = y + term2_noise + cluster_noise
  sorted_idx = np.argsort(y)
  X_sorted = X[sorted_idx]  # pylint: disable=invalid-name
  y_sorted = y[sorted_idx]
  y_tilde_sorted = y_tilde[sorted_idx]
  bag_features = []
  bag_label = []
  instance_labels = []
  y_tilde_labels = []
  event_labels = []
  for i in range(0, m):
    il = y_sorted[i * k : (i + 1) * k]
    yt = y_tilde_sorted[i * k : (i + 1) * k]
    bag_features.append(X_sorted[i * k : (i + 1) * k])
    if loss_type in ['inst_mir', 'agg_mir', 'event_mir']:
      bag_label.append(il[np.random.randint(len(il))])
    elif loss_type in ['bag_llp', 'inst_llp', 'event_llp']:
      bag_label.append(sum(il) / len(il))
    instance_labels.append(il)
    y_tilde_labels.append(yt)
    event_labels.append([bag_label[-1]] * len(il))
  bag_features = np.array(bag_features)
  bag_label = np.array(bag_label)
  if loss_type in ['bag_llp'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (k * (epsilon**2))
    )
    bag_noise = np.random.normal(0, sigma_privacy, len(bag_label))
    bag_label = bag_label + bag_noise
  instance_labels = np.array(instance_labels)
  event_labels = np.array(event_labels)
  ds = tf.data.Dataset.from_tensor_slices((
      bag_features,
      bag_label,
      y_tilde_labels,
      instance_labels,
      event_labels,
  ))
  ds = ds.shuffle(len(ds))
  loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
  return loader


def generate_min_var_direction_kmeans_bags(
    X,  # pylint: disable=invalid-name
    y_tilde,
    y,
    k=25,
    batch_size=128,
    loss_type='inst_mir',
    non_private=True,
    epsilon=1,
    delta=1e-5,
):
  """Generates min var direction kmeans bags.

  Args:
    X: Features.
    y_tilde: True labels.
    y: Noisy labels.
    k: Bag size.
    batch_size: Batch size.
    loss_type: Loss type.
    non_private: Non-private - boolean.
    epsilon: Epsilon - privacy parameter.
    delta: Delta - privacy parameter.

  Returns:
    Dataset loader.
  """
  # k - bag size
  m = len(X) // k  # num bags
  print('m', m, 'len(X)', len(X), 'k', k)
  if loss_type in ['inst_mir', 'event_mir'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_term2_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    sigma_cluster_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    cluster_noise = np.random.normal(0, sigma_cluster_privacy, len(y_tilde))
    term2_noise = np.random.normal(0, sigma_term2_privacy, len(y_tilde))
    y_tilde = y_tilde + cluster_noise
    y = y + term2_noise + cluster_noise
  sorted_idx = np.argsort(X.T[0])
  X_sorted = X[sorted_idx]  # pylint: disable=invalid-name
  y_sorted = y[sorted_idx]
  y_tilde_sorted = y_tilde[sorted_idx]
  bag_features = []
  bag_label = []
  instance_labels = []
  y_tilde_labels = []
  event_labels = []
  for i in range(0, m):
    il = y_sorted[i * k : (i + 1) * k]
    yt = y_tilde_sorted[i * k : (i + 1) * k]
    bag_features.append(X_sorted[i * k : (i + 1) * k])
    if loss_type in ['inst_mir', 'agg_mir', 'event_mir']:
      bag_label.append(il[np.random.randint(len(il))])
    elif loss_type in ['bag_llp', 'inst_llp', 'event_llp']:
      bag_label.append(sum(il) / len(il))
    instance_labels.append(il)
    y_tilde_labels.append(yt)
    event_labels.append([bag_label[-1]] * len(il))
  bag_features = np.array(bag_features)
  bag_label = np.array(bag_label)
  if loss_type in ['bag_llp'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (k * (epsilon**2))
    )
    bag_noise = np.random.normal(0, sigma_privacy, len(bag_label))
    bag_label = bag_label + bag_noise
  instance_labels = np.array(instance_labels)
  event_labels = np.array(event_labels)
  ds = tf.data.Dataset.from_tensor_slices((
      bag_features,
      bag_label,
      y_tilde_labels,
      instance_labels,
      event_labels,
  ))
  ds = ds.shuffle(len(ds))
  loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
  return loader


def get_even_clusters(X, cluster_size):  # pylint: disable=invalid-name
  """Generates even clusters."""
  n_clusters = int(np.ceil(len(X) / cluster_size))
  kmeans = KMeans(n_clusters)
  kmeans.fit(X)
  centers = kmeans.cluster_centers_
  centers = (
      centers.reshape(-1, 1, X.shape[-1])
      .repeat(cluster_size, 1)
      .reshape(-1, X.shape[-1])
  )
  distance_matrix = distance.cdist(X, centers)
  clusters = optimize.linear_sum_assignment(distance_matrix)[1] // (
      cluster_size
  )
  return clusters


def generate_feature_kmeans_bags(
    X,  # pylint: disable=invalid-name
    y_tilde,
    y,
    k=25,
    batch_size=128,
    loss_type='inst_mir',
    non_private=True,
    epsilon=1,
    delta=1e-5,
):
  """Generates feature kmeans bags.

  Args:
    X: Features.
    y_tilde: True labels.
    y: Noisy labels.
    k: Bag size.
    batch_size: Batch size.
    loss_type: Loss type.
    non_private: Non-private - boolean.
    epsilon: Epsilon - privacy parameter.
    delta: Delta - privacy parameter.

  Returns:
    Dataset loader.
  """
  # k - bag size
  m = len(X) // k  # num bags  # pylint: disable=unused-variable
  # kmeans = EqualGroupsKMeans(n_clusters=m, random_state=0).fit(X)
  X = X[: len(X) - len(X) % k]
  if loss_type in ['inst_mir', 'event_mir'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_term2_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    sigma_cluster_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    cluster_noise = np.random.normal(0, sigma_cluster_privacy, len(y_tilde))
    term2_noise = np.random.normal(0, sigma_term2_privacy, len(y_tilde))
    y_tilde = y_tilde + cluster_noise
    y = y + term2_noise + cluster_noise
  bag_assignment = get_even_clusters(X, k)
  set_labels = set(bag_assignment)
  bag_features = []
  bag_label = []
  instance_labels = []
  y_tilde_labels = []
  event_labels = []
  for label in set_labels:
    bf, il, yt = [], [], []
    for idx in range(len(bag_assignment)):
      if bag_assignment[idx] == label:
        bf.append(X[idx])
        il.append(y[idx])
        yt.append(y_tilde[idx])
    bag_features.append(bf)
    if loss_type in ['inst_mir', 'agg_mir', 'event_mir']:
      bag_label.append(il[np.random.randint(len(il))])
    elif loss_type in ['bag_llp', 'inst_llp', 'event_llp']:
      bag_label.append(sum(il) / len(il))
    instance_labels.append(il)
    y_tilde_labels.append(yt)
    event_labels.append([bag_label[-1]] * len(il))
  bag_features = np.array(bag_features)
  bag_label = np.array(bag_label)
  if loss_type in ['bag_llp'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (k * (epsilon**2))
    )
    bag_noise = np.random.normal(0, sigma_privacy, len(bag_label))
    bag_label = bag_label + bag_noise
  instance_labels = np.array(instance_labels)
  y_tilde_labels = np.array(y_tilde_labels)
  event_labels = np.array(event_labels)
  ds = tf.data.Dataset.from_tensor_slices((
      bag_features,
      bag_label,
      y_tilde_labels,
      instance_labels,
      event_labels,
  ))
  ds = ds.shuffle(len(ds))
  loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
  return loader


def generate_transformed_feature_kmeans_bags(
    X,  # pylint: disable=invalid-name
    y_tilde,
    y,
    k=25,
    batch_size=128,
    loss_type='inst_mir',
    non_private=True,
    epsilon=1,
    delta=1e-5,
):
  """Generates transformed/scaled feature kmeans bags.

  Args:
    X: Features.
    y_tilde: True labels.
    y: Noisy labels.
    k: Bag size.
    batch_size: Batch size.
    loss_type: Loss type.
    non_private: Non-private - boolean.
    epsilon: Epsilon - privacy parameter.
    delta: Delta - privacy parameter.

  Returns:
    Dataset loader.
  """
  # k - bag size
  m = len(X) // k  # num bags
  X = X[:int(m*k)]
  y_tilde = y_tilde[:int(m*k)]
  y = y[:int(m*k)]
  # kmeans = EqualGroupsKMeans(n_clusters=m, random_state=0).fit(X)
  if loss_type in ['inst_mir', 'event_mir'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_term2_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    sigma_cluster_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (epsilon**2)
    )
    cluster_noise = np.random.normal(0, sigma_cluster_privacy, len(y_tilde))
    term2_noise = np.random.normal(0, sigma_term2_privacy, len(y_tilde))
    y_tilde = y_tilde + cluster_noise
    y = y + term2_noise + cluster_noise
  cov = X.T @ X
  transformed_cov = linalg.fractional_matrix_power(cov, -0.5)
  transformed_X = X @ transformed_cov  # pylint: disable=invalid-name
  bag_assignment = get_even_clusters(transformed_X, k)
  set_labels = set(bag_assignment)
  bag_features = []
  bag_label = []
  instance_labels = []
  y_tilde_labels = []
  event_labels = []
  for label in set_labels:
    bf, il, yt = [], [], []
    for idx in range(len(bag_assignment)):
      if bag_assignment[idx] == label:
        bf.append(X[idx])
        il.append(y[idx])
        yt.append(y_tilde[idx])
    bag_features.append(bf)
    if loss_type in ['inst_mir', 'agg_mir', 'event_mir']:
      bag_label.append(il[np.random.randint(len(il))])
    elif loss_type in ['bag_llp', 'inst_llp', 'event_llp']:
      bag_label.append(sum(il) / len(il))
    instance_labels.append(il)
    y_tilde_labels.append(yt)
    event_labels.append([bag_label[-1]] * len(il))
  bag_features = np.array(bag_features)
  bag_label = np.array(bag_label)
  if loss_type in ['bag_llp'] and not non_private:
    R = 2  # pylint: disable=invalid-name
    sigma_privacy = np.sqrt(
        (R**2) * (np.log(1.25 / delta)) / (k * (epsilon**2))
    )
    bag_noise = np.random.normal(0, sigma_privacy, len(bag_label))
    bag_label = bag_label + bag_noise
  instance_labels = np.array(instance_labels)
  y_tilde_labels = np.array(y_tilde_labels)
  event_labels = np.array(event_labels)
  ds = tf.data.Dataset.from_tensor_slices((
      bag_features,
      bag_label,
      y_tilde_labels,
      instance_labels,
      event_labels,
  ))
  ds = ds.shuffle(len(ds))
  loader = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=8)
  return loader
