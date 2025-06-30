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

"""Launch script for training on XManager."""
import itertools

from absl import app
from xmanager import xm
from xmanager import xm_local


def main(_) -> None:
  with xm_local.create_experiment(experiment_title='experiment') as experiment:
    spec = xm.PythonContainer(
        # Package the current directory that this script is in.
        path='.',
        base_image='gcr.io/deeplearning-platform-release/tf2-cpu',
        entrypoint=xm.ModuleName('main'),
    )

    [executable] = experiment.package([
        xm.Packageable(
            executable_spec=spec,
            executor_spec=xm_local.Local.Spec(),
        ),
    ])

    # Hyper-parameter definition.
    epochs = [200]  # 51
    lr = [1e-4]  # 1e-2
    n = [5000]
    d = [32]
    k = [50]  # , 200]
    sigma = [0.5]
    epsilon = [0.5]
    delta = [1e-5]
    loss_fn = ['mse']
    loss_type = ['bag_llp']  # ['bag_llp', 'event_mir', 'event_llp', 'agg_mir']
    bagging = ['random']
    # ['random_superbags', 'label_kmeans_superbags','random', 'label_kmeans',
    # 'features_kmeans', 'min_var_kmeans', 'transformed_feature_kmeans']
    non_private = [True]
    ds_num = [1]  # 1,2,3
    fold = [0]
    data_type = [
        'wine_quality_white'
    ]  # 'wine_quality_red', 'wine_quality_white']
    # ['isotropic','non_isotropic_independent', 'non_isotropic_non_independent']
    parameters = [
        {
            'epochs': epochs_hp,
            'ds_num': ds_num_hp,
            'fold': fold_hp,
            'lr': lr_hp,
            'n': n_hp,
            'd': d_hp,
            'k': k_hp,
            'sigma': sigma_hp,
            'epsilon': epsilon_hp,
            'delta': delta_hp,
            'non_private': non_private_hp,
            'loss_fn': loss_fn_hp,
            'loss_type': loss_type_hp,
            'bagging': bagging_hp,
            'data_type': data_type_hp,
            'save_dataloader_cv': False,
        }
        for epochs_hp, ds_num_hp, fold_hp, lr_hp, n_hp, d_hp, k_hp, sigma_hp, epsilon_hp, delta_hp, non_private_hp, loss_fn_hp, loss_type_hp, bagging_hp, data_type_hp in itertools.product(
            epochs,
            ds_num,
            fold,
            lr,
            n,
            d,
            k,
            sigma,
            epsilon,
            delta,
            non_private,
            loss_fn,
            loss_type,
            bagging,
            data_type,
        )
    ]

    for hparams in parameters:
      experiment.add(
          xm.Job(
              executable=executable,
              executor=xm_local.Local(
                  docker_options=xm_local.DockerOptions(
                      volumes={  # pylint: disable=duplicate-key
                          'agg': '/opt_bagging/data',
                          'agg': '/opt_bagging/results',
                      }
                  )
              ),
              args=hparams,
          )
      )


if __name__ == '__main__':
  app.run(main)
