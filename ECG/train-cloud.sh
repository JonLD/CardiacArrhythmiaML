#!/bin/bash
# Copyright 2019 Google LLC
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
# ==============================================================================
# This scripts performs cloud training for a PyTorch model.
echo "Training cloud ML model"

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=ecg_job_$(date +%Y%m%d_%H%M%S)

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
REGION=us-central1

IMAGE_URI=gcr.io/ml-projects-314319/mlmodel

# Submit your training job
echo "Submitting the training job"

# These variables are passed to the docker image
# Note: these files have already been copied over when the image was built
DATA_DIR=/root

gcloud beta ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier BASIC_GPU \
    -- \
    --data-dir ${DATA_DIR} \
    --num-epochs=50

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}