# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example implementation of image model in TensorFlow 
that can be trained and deployed on Cloud ML Engine
"""

import argparse
import json
import os

from . import model
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
    "--database_path",
        help = "Path for the database",
        required = True
    )
    parser.add_argument(
        "--dependent_columns",
        help = "Comma seperated list of dependent column names",
        required = False
    )
    parser.add_argument(
        "--target_columns",
        help = "Comma seperated list of target column names",
        required = False
    )
    parser.add_argument(
        "--train_size_value",
        help = "Batch size for training",
        type = float,
        default = 0.8
    )
    parser.add_argument(
        "--random_state_value",
        help = "Value for Random State",
        type = int,
        default = 42
    )
    parser.add_argument(
        "--input_dimensions",
        help = "Input dimensions for dependent variables",
        type = int,
        default = 3
    )
    parser.add_argument(
        "--batch_size",
        help = "Batch size for training steps",
        type = int,
        default = 100
    )
    parser.add_argument(
        "--epochs",
        help = "steps_per_epochs optional",
        type = int,
        default = 100
    )
    parser.add_argument(
        "--steps_per_epoch",
        help = "steps_per_epochs optional",
        type = int,
        default = 100
    )
    parser.add_argument(
        "--validation_steps",
        help = "validation steps optional",
        type = int,
        default = 100
    )
    parser.set_defaults(batch_norm = False)

    args = parser.parse_args()
    hparams = args.__dict__
    
    # Append trial_id to path for hptuning
# =============================================================================
#     output_dir = hparams.pop("output_dir")
#     output_dir = os.path.join(
#         output_dir,
#         json.loads(
#             os.environ.get("TF_CONFIG", "{}")
#         ).get("task", {}).get("trial", "")
#     )  
# =============================================================================

    # Run the training job
    model.train_and_evaluate(hparams)
