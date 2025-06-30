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

"""Main file for training and evaluating models."""

import datetime
import json
import os

from absl import app
from absl import flags
from generate_bags import generate_feature_kmeans_bags
from generate_bags import generate_label_kmeans_bags
from generate_bags import generate_label_kmeans_superbags
from generate_bags import generate_min_var_direction_kmeans_bags
from generate_bags import generate_random_bags
from generate_bags import generate_random_superbags
from generate_bags import generate_transformed_feature_kmeans_bags
from generate_bags import load_data
from generate_bags import load_real_data
from generate_bags import save_isotropic_synthetic_data
from generate_bags import save_synthetic_independent_nonisotropic_data
from generate_bags import save_synthetic_nonisotropic_data
from generate_bags import save_wine_data
from model import LinearModel
from model import loss_fn
from model import test_step
from model import train_step
import pandas as pd
import tensorflow as tf


flags.DEFINE_integer('n', 50000, 'num samples')
flags.DEFINE_integer('d', 32, 'feature dimension')
flags.DEFINE_integer('k', 100, 'bag size')
flags.DEFINE_float('sigma', 0.1, 'sigma')
flags.DEFINE_float('epsilon', 0.1, 'epsilon privacy')
flags.DEFINE_float('delta', 1e-5, 'delta privacy')
flags.DEFINE_integer('ds_num', 1, 'dataset number')
flags.DEFINE_integer('fold', 1, 'fold number')
flags.DEFINE_integer('epochs', 61, 'Number of epochs.')
flags.DEFINE_float('lr', 1e-2, 'Learning rate')
flags.DEFINE_boolean('non_private', True, 'Non private or Private version')
flags.DEFINE_string('loss_fn', 'mse', 'Loss function.')
flags.DEFINE_string('loss_type', 'bag', 'event/instance/bag')
flags.DEFINE_string('bagging', 'random', 'random/label_kmeans/features_kmeans')
flags.DEFINE_string(
    'data_type',
    'isotropic',
    'isotropic/non_isotropic_independent/non_isotropic_non_independent/wine_quality_white/wine_quality_red',
)
flags.DEFINE_string('data_dir', 'data', 'data directory')
flags.DEFINE_string('results_dir', 'results', 'results directory')
flags.DEFINE_bool('save_dataloader_cv', True, 'Save dataloader for CV split.')

args = flags.FLAGS


def main():

  print('----------------------------------------------------')
  print(args)
  eid = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  config_str = json.dumps(args.flag_values_dict(), indent=2)  # pylint: disable=unused-variable
  data_dir = None
  os.makedirs(f'{args.results_dir}', exist_ok=True)

  if args.data_type == 'isotropic':
    data_dir = f'{args.data_dir}/synthetic/isotropic'
    if args.save_dataloader_cv:
      for ds_num in range(1, 4):
        save_isotropic_synthetic_data(
            args.data_dir, args.n, args.d, args.sigma, ds_num
        )

  elif args.data_type == 'non_isotropic_independent':
    data_dir = f'{args.data_dir}/synthetic/non_isotropic/independent'
    if args.save_dataloader_cv:
      for ds_num in range(1, 4):
        save_synthetic_independent_nonisotropic_data(
            args.data_dir, args.n, args.d, args.sigma, ds_num
        )
  elif args.data_type == 'non_isotropic_non_independent':
    data_dir = f'{args.data_dir}/synthetic/non_isotropic/non_independent'
    if args.save_dataloader_cv:
      for ds_num in range(1, 4):
        save_synthetic_nonisotropic_data(
            args.data_dir, args.n, args.d, args.sigma, ds_num
        )
  elif args.data_type == 'wine_quality_white':
    data_dir = f'{args.data_dir}/wine_quality/white'
    if args.save_dataloader_cv:
      save_wine_data(args.data_dir, 'white')
  elif args.data_type == 'wine_quality_red':
    data_dir = f'{args.data_dir}/wine_quality/red'
    if args.save_dataloader_cv:
      save_wine_data(args.data_dir, 'red')
  if args.data_type not in ['wine_quality_white', 'wine_quality_red']:
    (
        train_data,
        train_y_tilde,
        train_y,
        test_data,
        test_y_tilde,
        test_y,
        theta_true,
    ) = load_data(data_dir, args.sigma, ds_num=args.ds_num, fold=args.fold)
    eval_data, eval_y = None, None
  else:
    (
        train_data,
        train_y_tilde,
        train_y,
        test_data,
        test_y_tilde,
        test_y,
        eval_data,
        eval_y,
    ) = load_real_data(data_dir, ds_num=args.ds_num, fold=args.fold)
    theta_true = None
  trainloader, testloader = None, None
  if args.bagging == 'random':
    trainloader = generate_random_bags(
        train_data,
        train_y_tilde,
        train_y,
        args.k,
        batch_size=256,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    testloader = generate_random_bags(
        test_data,
        test_y_tilde,
        test_y,
        args.k,
        batch_size=128,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    if args.data_type in ['real_wine', 'real_criteo']:
      evalloader = generate_random_bags(  # pylint: disable=unused-variable
          eval_data,
          eval_y,
          eval_y,
          args.k,
          batch_size=256,
          loss_type=args.loss_type,
          non_private=args.non_private,
          epsilon=args.epsilon,
          delta=args.delta,
      )
  elif args.bagging == 'label_kmeans':
    trainloader = generate_label_kmeans_bags(
        train_data,
        train_y_tilde,
        train_y,
        args.k,
        batch_size=256,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    testloader = generate_label_kmeans_bags(
        test_data,
        test_y_tilde,
        test_y,
        args.k,
        batch_size=128,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
  elif args.bagging == 'features_kmeans':
    trainloader = generate_feature_kmeans_bags(
        train_data,
        train_y_tilde,
        train_y,
        args.k,
        batch_size=256,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    testloader = generate_feature_kmeans_bags(
        test_data,
        test_y_tilde,
        test_y,
        args.k,
        batch_size=128,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
  elif args.bagging == 'min_var_kmeans':
    trainloader = generate_min_var_direction_kmeans_bags(
        train_data,
        train_y_tilde,
        train_y,
        args.k,
        batch_size=256,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    testloader = generate_min_var_direction_kmeans_bags(
        test_data,
        test_y_tilde,
        test_y,
        args.k,
        batch_size=128,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
  elif args.bagging == 'random_superbags':
    trainloader = generate_random_superbags(
        train_data,
        train_y_tilde,
        train_y,
        args.k,
        batch_size=256,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    testloader = generate_random_superbags(
        test_data,
        test_y_tilde,
        test_y,
        args.k,
        batch_size=128,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
  elif args.bagging == 'label_kmeans_superbags':
    trainloader = generate_label_kmeans_superbags(
        train_data,
        train_y_tilde,
        train_y,
        args.k,
        batch_size=256,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    testloader = generate_random_superbags(
        test_data,
        test_y_tilde,
        test_y,
        args.k,
        batch_size=128,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
  elif args.bagging == 'transformed_feature_kmeans':
    trainloader = generate_transformed_feature_kmeans_bags(
        train_data,
        train_y_tilde,
        train_y,
        args.k,
        batch_size=256,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    testloader = generate_random_superbags(
        test_data,
        test_y_tilde,
        test_y,
        args.k,
        batch_size=128,
        loss_type=args.loss_type,
        non_private=args.non_private,
        epsilon=args.epsilon,
        delta=args.delta,
    )
  valloader = testloader
  model = LinearModel()
  results_df = pd.DataFrame(
      columns=[
          'epoch',
          'ds_num',
          'fold',
          'n',
          'k',
          'd',
          'loss_type',
          'sigma',
          'epsilon',
          'delta',
          'bagging',
          'data_type',
          'train_error',
          'val_error',
          'test_error',
          'theta_error',
      ]
  )
  results_df = results_df.set_index('epoch')
  loss = loss_fn(args.loss_fn)
  optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  val_loss = tf.keras.metrics.Mean(name='val_loss')
  test_loss = tf.keras.metrics.Mean(name='test_loss')
  train_loss_epoch = []
  val_loss_epoch = []
  for epoch in range(args.epochs):
    if epoch % 10 == 0:
      print(f'Epoch {epoch}')
    # Reset the metrics at the start of the next epoch
    train_loss.reset_state()
    val_loss.reset_state()
    train_loss_list = []
    for batch in trainloader:
      epoch_loss = train_step(batch, model, optimizer, loss, args.loss_type)
      train_loss_list.append(epoch_loss)
      train_loss(epoch_loss)
    if epoch % 10 == 0:
      print(f'Train Loss: {sum(train_loss_list)/len(train_loss_list)}')
    train_loss_epoch.append(train_loss.result())
    val_loss_list = []
    val_theta_error = []
    for batch in valloader:
      vloss, theta_error = test_step(
          batch, model, loss, args.loss_type, theta_true
      )
      val_loss(vloss)
      val_theta_error.append(theta_error)
      val_loss_list.append(vloss)
    val_loss_epoch.append(val_loss.result())
    if epoch % 5 == 0:
      print(
          f'Validation Loss: {sum(val_loss_list)/len(val_loss_list)} Val Theta'
          f' Error: {sum(val_theta_error)/len(val_theta_error)}'
      )
      print(
          f'Epoch {epoch}, '
          f'Train Loss: {train_loss.result()}, '
          f'Validation Loss: {val_loss.result()}, '
      )
    if epoch % 5 == 0:
      test_loss_list = []
      test_theta_error = []
      for batch in testloader:
        vloss, theta_error = test_step(
            batch, model, loss, args.loss_type, theta_true
        )
        test_loss(vloss)
        test_theta_error.append(theta_error)
        test_loss_list.append(vloss)
      val_loss_epoch.append(val_loss.result())

      print(
          f'Test Loss: {sum(test_loss_list)/len(test_loss_list)} Test Theta'
          f' Error: {sum(test_theta_error)/len(test_theta_error)}'
      )
      print(
          f'Epoch {epoch}, '
          f'Train Loss: {train_loss.result()}, '
          f'Validation Loss: {val_loss.result()}, '
      )

      results_df.loc[len(results_df)] = {
          'epoch': epoch,
          'ds_num': args.ds_num,
          'fold': args.fold,
          'n': args.n,
          'k': args.k,
          'd': args.d,
          'loss_type': args.loss_type,
          'sigma': args.sigma,
          'epsilon': args.epsilon,
          'delta': args.delta,
          'bagging': args.bagging,
          'data_type': args.data_type,
          'train_error': sum(train_loss_list) / len(train_loss_list),
          'val_error': sum(val_loss_list) / len(val_loss_list),
          'test_error': sum(test_loss_list) / len(test_loss_list),
          'theta_error': sum(test_theta_error) / len(test_theta_error),
      }
  results_df.to_csv(f'{args.results_dir}/{eid}.csv')


if __name__ == '__main__':
  app.run(main)
