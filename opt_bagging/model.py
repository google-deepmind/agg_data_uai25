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

"""Model and loss functions for linear regression."""
import tensorflow as tf


class LinearModel(tf.keras.Model):
  """Linear Regression."""

  def __init__(self):
    super().__init__()
    output_dim = 1
    self.linear_model = tf.keras.layers.Dense(output_dim)

  def call(self, batch, loss_type='bag'):
    bag_features, bag_label, y_tilde_labels, y_labels, event_labels = batch  # pylint: disable=unused-variable
    pred, target = None, None
    if loss_type == 'agg_mir':  # AggMIR
      agg_feature_vector = tf.reduce_mean(bag_features, 1)
      pred = tf.squeeze(self.linear_model(agg_feature_vector), axis=1)
      target = bag_label
    elif loss_type == 'bag_llp':
      data_shape = tf.shape(bag_features).numpy()
      bag_features = tf.reshape(
          bag_features, [data_shape[0] * data_shape[1], data_shape[2]]
      )
      pred = tf.squeeze(self.linear_model(bag_features), axis=1)
      pred = tf.reshape(pred, [data_shape[0], data_shape[1]])
      pred = tf.reduce_mean(pred, 1)
      target = bag_label
    elif loss_type in ['inst_mir', 'event_mir', 'inst_llp', 'event_llp']:
      data_shape = tf.shape(bag_features).numpy()
      bag_features = tf.reshape(
          bag_features, [data_shape[0] * data_shape[1], data_shape[2]]
      )
      pred = tf.squeeze(self.linear_model(bag_features), axis=1)
      if loss_type in ['inst_mir', 'inst_llp']:
        target = tf.reshape(y_labels, [data_shape[0] * data_shape[1], 1])
      elif loss_type in ['event_mir', 'event_llp']:
        target = tf.reshape(event_labels, [data_shape[0] * data_shape[1], 1])
    return target, pred


def loss_fn(loss_fn='mse'):  # pylint: disable=redefined-outer-name
  loss = None
  loss_no_red = None
  if loss_fn == 'mse':
    loss = tf.keras.losses.MeanSquaredError()
    loss_no_red = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE
    )
  return loss, loss_no_red


def get_loss(loss_fn, target, pred):  # pylint: disable=redefined-outer-name
  return loss_fn(target, pred)


def get_theta_error(loss_fn, model, theta_true):  # pylint: disable=redefined-outer-name
  return loss_fn(model.linear_model.weights[0], theta_true)


def train_step(batch, model, optimizer, loss_fn, loss_type):  # pylint: disable=redefined-outer-name
  with tf.GradientTape() as tape:
    target, pred = model(batch, loss_type=loss_type)

    loss = get_loss(loss_fn[0], target, pred)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss


def test_step(batch, model, loss_fn, loss_type, theta_true):  # pylint: disable=redefined-outer-name
  target, pred = model(batch, loss_type=loss_type)
  loss = get_loss(loss_fn[0], target, pred)
  if theta_true is not None:
    theta_error = get_theta_error(loss_fn[0], model, theta_true)
  else:
    theta_error = 0
  return loss, theta_error


def eval_step(batch, model, loss_fn, loss_type):  # pylint: disable=redefined-outer-name
  target, pred = model(batch, loss_type=loss_type)
  loss = get_loss(loss_fn[0], target, pred)
  return loss
