import sys, os
import yaml
import logging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import numpy as np
import torch
import pytorch_lightning as pl
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger
import argparse
from subprocess import check_output
from argparse import Namespace
from oct_ml_vae_fc import VAE_OCT, classifier_oct, VAE_classifier_OCT
from pl_datamodule_vae_fc import OCT_DM
from load_config import load_config
from oct_ml_module_fc2 import VAE_class
from oct_ml_module_vae import VAE
from IPython import embed
from pytorch_lightning.profiler import PyTorchProfiler
from torch.profiler import profile, record_function, ProfilerActivity

def get_eye_args(config):
    net = config.network_architecture
    convs = net.convolution.parameters
    vae_args = {
                 "input_dim": config.input_dim,
                 "latent_dim": net.latent_dim,
                 "n_classes": config.n_classes,
                 "n_channels": net.convolution.parameters.channels,
                 "kernel_size": net.convolution.parameters.kernel_size,
                 "padding":net.convolution.parameters.padding,
                 "stride":net.convolution.parameters.stride
                  }
    classifier_args = {
                        "n_classes": config.n_classes,
                        "n_channels": net.convolution.parameters.channels,
                        "n_channels_class": net.convolution.parameters.channels_class,
                        "n_channels_concat": net.convolution.parameters.channels_concat,
                      }
    return vae_args, classifier_args

def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def get_datamodule(config):
    dm_classification = OCT_DM(config.dir_imgs_left, config.dir_imgs_right, config.ids_set_class, img_size=config.input_dim[-3:], batch_size=config.optimizer.batch_size)
    return dm_classification

def ml_model1_trainer(config):
     dm_vae, _ = get_datamodule(config)
     vae_oct_args = get_eye_args(config)
     vae_oct = VAE_OCT(**vae_oct_args)
     model1 = VAE(vae_oct, config)
     trainer_vae = pl.Trainer(strategy="deepspeed_stage_3", callbacks=[EarlyStopping(monitor="avg_total_loss_validation", mode="min", patience=7)], max_epochs=1000) #, min_epochs=1)7

     return dm_vae, model1, trainer_vae


def ml_model2_trainer(config):
     dm_classification = get_datamodule(config)
     vae_oct_args, classifier_args = get_eye_args(config)
     oct_vae = VAE_OCT(**vae_oct_args)
     oct_classifier = classifier_oct(**classifier_args)
     vae_oct_classification = VAE_classifier_OCT(oct_vae, oct_classifier)
     # assing weights from pretrained model
     checkpoint=torch.load(config.pretrained_model, map_location= torch.device('cuda:0'))
     _model_pretrained_weights = {k.replace("model.","model.vae_OCT."): v for k, v in checkpoint['state_dict'].items()}
     model2 =  VAE_class(vae_oct_classification, config)
     model2.load_state_dict(_model_pretrained_weights, strict=False)
     early_stopping_callback = [EarlyStopping(monitor="avg_total_loss_validation", mode='min',  patience=5)]  #, ModelSummary(max_depth=-1)]
     trainer = pl.Trainer(accelerator="gpu", callbacks=early_stopping_callback, devices=1, num_nodes=1, max_epochs=0)
     return dm_classification, model2, trainer #, trainer_right


def get_mlflow_parameters(config):

    mlflow_parameters = {
            "platform": check_output(["hostname"]).strip().decode(),
            "w_kl": config.w_kld,
            "w_bce": config.w_bce,
            "latent_dim": config.network_architecture.latent_dim,
            "n_channels": config.network_architecture.convolution.parameters.channels,
            "n_channels_class": config.network_architecture.convolution.parameters.channels_class,
            "n_channels_concat": config.network_architecture.convolution.parameters.channels_concat,
            "batch_size": config.optimizer.batch_size
    }
    return mlflow_parameters


def main(config):
      if config.log_to_mlflow:
          mlflow.pytorch.autolog()
              
          if config.pretrained_model is None:
              exp1 = config.mlflow.pretraining.experiment_name
              exp1 = exp1 if exp1 is not None else "default"
              mlf_logger = MLFlowLogger(experiment_name=exp1, tracking_uri="file:./mlruns")

              try:
                  exp_id = mlflow.create_experiment(exp1)
              except:
                  # If the experiment already exists, we can just retrieve its ID
                  exp_id = mlflow.get_experiment_by_name(exp1).experiment_id

              with mlflow.start_run(run_id=mlf_logger.run_id, experiment_id=exp_id, run_name=config.mlflow.pretraining.run_name) as run:
                   dm_vae, model1, trainer_vae = ml_model1_trainer(config)
                   trainer_vae.logger = mlf_logger
                   for k, v in get_mlflow_parameters(config).items():
                     mlflow.log_param(k, v)
                   trainer_vae.fit(model1, datamodule=dm_vae) #, logger=mlf_logger)            
                   
                   print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))          
          else: 
           exp2 = config.mlflow.classifier_training.experiment_name
           exp2 = exp2 if exp2 is not None else "default"
           mlf_logger_2 = MLFlowLogger(experiment_name=exp2, tracking_uri="file:./mlruns")

           try:
            exp_id = mlflow.create_experiment(exp2)
           except:
            # If the experiment already exists, we can just retrieve its ID
            exp_id = mlflow.get_experiment_by_name(exp2).experiment_id

           with mlflow.start_run(experiment_id=exp_id, run_id=mlf_logger_2.run_id, run_name=config.mlflow.classifier_training.run_name ) as run: 
            for k, v in get_mlflow_parameters(config).items():
                     mlflow.log_param(k, v)
            try: 
                mlflow.log_param("base_model", run_id)
            except:
                mlflow.log_param("base_model", config.pretrained_model)

            dm_classification, model2, trainer = ml_model2_trainer(config)
            ## agregar pesos aqui
            trainer.logger = mlf_logger_2
            trainer.fit(model2, datamodule=dm_classification)
      else:
#        trainer_vae.fit(model1, datamodule=dm_vae )  
         trainer.fit(model2, datamodule=dm_classification)
         
      trainer.test(model2, datamodule=dm_classification)# , trainer=trainer)
#      trainer.predict(model2, datamodule=dm_classification)
#      trainer_vae.test(model1, datamodule=dm_vae)

if __name__ == '__main__':
 
    import argparse

 
    parser = argparse.ArgumentParser(description='VAE_FC OCT')
    parser.add_argument('--config', default = 'config2.yaml')
    parser.add_argument('--latent_dim', type=int, default=None,
                    help='latent dimensionality (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=None, metavar='N',
                    help='batch size for data (default: 1)')
    parser.add_argument('--epochs', type=int, default=None, metavar='E',
                    help='number of epochs to train (default: 5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--dir_imgs', type=str, default = None)
    parser.add_argument('--ids_set', type=str, default = None)
    parser.add_argument('--lr', type=float, default=None,
                    help='the learning rate')
    parser.add_argument('--w_kld', type=float, default=None,
                    help='the weight of the KL term.')
    parser.add_argument('--n_classes', type=int, default=None,
                    help='num of classes')
    parser.add_argument('--channels_class', type=int, default=None,
                    help='num of channels classifier')
    parser.add_argument('--channels_concat', type=int, default=None,
                    help='num of channels concatenate')
    parser.add_argument('--weight_decay', type=float, default=None,
                    help='the weight decay')
#    parser.add_argument("--log_to_mlflow", default=False, action="store_true",
#                    help="Set this flag if you want to log the run's data to MLflow.",)
    parser.add_argument("--disable_mlflow_logging", default=False, action="store_true",
        help="Set this flag if you don't want to log the run's data to MLflow.",)

    args = parser.parse_args()

    ### Load configuration
    if not os.path.exists(args.config):
        logger.error("Config not found" + args.config)


    config = load_config(args.config, args)
    config.log_to_mlflow = not args.disable_mlflow_logging
    main(config)

   




