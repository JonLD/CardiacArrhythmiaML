# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from ecg_train import tune_ecg


def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """
    args_parser = argparse.ArgumentParser()

    # Data files arguments
    args_parser.add_argument(
        '--data-dir',
        help='GCS or local paths to training data',
        nargs='+',
        required=True)

    # Experiment arguments
    args_parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --train-size and --num-epochs are specified,
        --train-steps will be: (train-size/train-batch-size) * num-epochs.\
        """,
        default=50,
        type=int,
    )
    # Feature columns arguments
    args_parser.add_argument(
        '--normalised',
        help="""
        If set to True, the normalised data will be used.
        """,
        type=bool,
        default=True,
    )
    
    args_parser.add_argument(
        '--num-samples',
        help="""\
        Number of times to sample from the hyperparameter space.\
        """,
        default=10,
        type=int,
    )

    return args_parser.parse_args()


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    tune_ecg(args.data_dir, num_epochs=args.num_epochs, normalised=args.normalised, num_samples=args.num_samples)


if __name__ == '__main__':
    main()