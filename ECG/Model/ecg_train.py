import pytorch_lightning as pl
import math
import torch
from ecg_model import MLECG
from ecg_data_module import ECGDataModule
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar

def train_ecg(config, data_dir=None, num_epochs=10, normalised = True, num_gpus=1):
    model = MLECG(config)
    if(normalised):
        model = model.float()
    dm = ECGDataModule(
        data_dir=data_dir, num_workers=8, batch_size=config["batch_size"], normalised=normalised)
    metrics = {"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"}
    bar = ProgressBar()
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        progress_bar_refresh_rate=0,
        callbacks=[EarlyStopping(patience=10, monitor='ptl/val_loss'),bar]
#            TuneReportCheckpointCallback(
#                metrics=metrics,
#                filename="checkpoint",
#                on="validation_end")
#        ]
        )
    trainer.fit(model, dm)
    
def test_ecg(data_dir=None, model_dir=None, h_dir=None, normalised = True, num_gpus=1):
    dm = ECGDataModule(data_dir=data_dir, num_workers=8, batch_size=128, normalised=normalised)
    trainer = pl.Trainer(gpus=math.ceil(num_gpus))
    checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)
    c0 = {
         "lstm_size": 5,
         "lr": 0.001,
         "batch_size": 128
        }
    model = MLECG(c0).float().load_from_checkpoint(
    checkpoint_path=model_dir,
    hparams_file=h_dir,
    map_location=None, config=c0)
    trainer.test(model=model, datamodule=dm)
    


def tune_ecg(data_dir, num_epochs=1, normalised = True, num_samples=10, gpus_per_trial=1):
    config = {
         "lstm_size": tune.choice([2, 3, 4, 5, 32, 64, 128]),
         "lr": tune.loguniform(1e-4, 1e-1),
         "batch_size": tune.choice([32, 64, 128, 256, 512])
        }
#    scheduler = PopulationBasedTraining(
#        time_attr="training_iteration",
#        perturbation_interval=5,
#        hyperparam_mutations={
#            # distribution for resampling
#            "lr": lambda: np.random.uniform(0.0001, 1),
#            # allow perturbations within this set of categorical values
#            "momentum": [0.8, 0.9, 0.99],
#        })

    reporter = CLIReporter(
        parameter_columns=["lstm_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])
    
    trainable = tune.with_parameters(
        train_ecg,
        data_dir=data_dir,
        normalised = normalised,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial)
    
    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 16,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        local_dir="./results",
#        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_ecg")

    print("Best hyperparameters found were: ", analysis.best_config)
c0 = {
         "lstm_size": 3,
         "lr": 0.001,
         "batch_size": 128
        }
c1 = {
         "lstm_size": 2,
         "lr": 0.001,
         "batch_size": 128
        }
c2 = {
         "lstm_size": 4,
         "lr": 0.001,
         "batch_size": 128
        }
c3 = {
         "lstm_size": 5,
         "lr": 0.001,
         "batch_size": 128
        }
list1 = [c0,c1,c2,c3]
if __name__ == '__main__':
#    test_ecg(data_dir="C:/MPhys_project/Istvan_Jon/CardioML/ECG/Model/testset_normalised.mat",
#             model_dir="C:\MPhys_project\Istvan_Jon\CardioML\ECG\Model\lightning_logs\et\checkpoints\epoch=142-step=8007.ckpt",
#             h_dir = "C:\MPhys_project\Istvan_Jon\CardioML\ECG\Model\lightning_logs\et\hparams.yaml")
    for i in list1:
        train_ecg(i, data_dir="C:/MPhys_project/Istvan_Jon/CardioML/ECG/Model/trainingset_normalised.mat", num_epochs=9999, normalised=True);