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

# Installs pandas, and google-cloud-storage, tabulate
RUN pip install pandas google-cloud-storage tabulate

# Installs Pytorch Lightning, Ray, Torchmetrics and scipy
RUN pip install pytorch_lightning torchmetrics scipy ray
RUN pip install "ray[default]"
RUN pip install "ray[tune]"

# You can modify this by changing the arguments passed into the Dockerfile
ARG DATA_DIR=gs://mldatastorage/trainingset_normalised.mat
ARG P1=gs://mldatastorage/ecg_data_module.py
ARG P2=gs://mldatastorage/ecg_model.py
ARG P3=gs://mldatastorage/ecg_train.py
ARG P4=gs://mldatastorage/ecg_dataset.py
ARG P5=gs://mldatastorage/task.py

# Download the data from the public Google Cloud Storage bucket for this sample
RUN gsutil cp $DATA_DIR ./trainingset_normalised.mat
#COPY DataPreProcess/trainingset_normalised.mat ./trainingset_normalised.mat

# Copies the trainer code to the docker image.
#COPY model/ecg_data_module.py ./model/ecg_data_module.py
#COPY model/ecg_model.py ./model/ecg_model.py
#COPY model/ecg_train.py ./model/ecg_train.py
#COPY model/ecg_dataset.py ./model/ecg_dataset.py
#COPY model/task.py ./model/task.py
RUN gsutil cp $P1 ./model/ecg_data_module.py
RUN gsutil cp $P2 ./model/ecg_model.py
RUN gsutil cp $P3 ./model/ecg_train.py
RUN gsutil cp $P4 ./model/ecg_dataset.py
RUN gsutil cp $P5 ./model/task.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-u", "model/task.py"]