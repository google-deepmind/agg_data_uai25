# Aggregating Data for Optimal and Private Learning

This repository accompanies the publication

> [Aggregating Data for Optimal and Private Learning](https://arxiv.org/abs/2411.19045).
> *UAI 2025 (Main Conference Oral)*

## Installation

-   Install [XManager](https://github.com/google-deepmind/xmanager)
-   Install
    [Docker](https://github.com/google-deepmind/xmanager#install-docker-optional)
-   (Optional, Recommended) Make a virtual environment: `cd <w2s> python -m venv
    .venv source .venv/bin/activate`
-   Run `pip install -r requirements.txt` from opt_bagging/ folder

-   Run the following: `docker volume create agg`

## Usage

Setup:

-   Download the Wine Quality (Red and White) datasets from
    [here](https://archive.ics.uci.edu/dataset/186/wine+quality) and save them as csv files:
    `opt_bagging/data/wine_quality/winequality-{white,red}.csv`

Launch:

-   Define the dataset to be use in xm_launch.py and other hyperparameters
    as well.
-   To create and save cross-validation splits set `save_dataloader_cv` to True.
-   Run `.venv/bin/xmanager launch xm_launch.py`

## Citing this work

```
@inproceedings{
agarwal2025aggregating,
title={Aggregating Data for Optimal Learning},
author={Sushant Agarwal and Yukti Makhija and Rishi Saket and Aravindan Raghuveer},
booktitle={The 41st Conference on Uncertainty in Artificial Intelligence},
year={2025},
url={https://openreview.net/forum?id=85HF0u4lSB}
}

```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
