import pytorch_lightning as pl
import math
from ecg_model import MLECG
from ecg_data_module import ECGDataModule
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

def train_ecg(config, data_dir=None, num_epochs=10, normalised = True, num_gpus=0):
    model = MLECG(config)
    if(normalised):
        model = model.float()
    dm = ECGDataModule(
        data_dir=data_dir, num_workers=1, batch_size=config["batch_size"], normalised=normalised)
    metrics = {"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"}
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        progress_bar_refresh_rate=0,    
        callbacks=[
            TuneReportCheckpointCallback(
                metrics=metrics,
                filename="checkpoint",
                on="validation_end")
        ])
    trainer.fit(model, dm)


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
            "cpu": 1,
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