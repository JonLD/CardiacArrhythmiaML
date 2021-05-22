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

# Install pytorch
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-8

WORKDIR /root

# Installs pandas, and google-cloud-storage.
RUN pip install pandas google-cloud-storage
# Installs Pytorch Lightninh, Ray, Torchmetrics and scipy
RUN pip install pytorch_lightning ray torchmetrics scipy

# You can modify this by changing the arguments passed into the Dockerfile
ARG DATA_DIR=gs://mldatastorage/trainingset_normalised.mat

# Download the data from the public Google Cloud Storage bucket for this sample
RUN gsutil cp $DATA_DIR ./trainingset_normalised.mat

# Copies the trainer code to the docker image.
COPY model/ecg_data_module.py ./model/ecg_data_module.py
COPY model/ecg_model.py ./model/ecg_model.py
COPY model/ecg_train.py ./model/ecg_train.py
COPY model/ecg_dataset.py ./model/ecg_dataset.py
COPY model/task.py ./model/task.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-u", "model/task.py"]