import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix,  confusion_matrix, roc_curve, RocCurveDisplay, roc_auc_score, classification_report, precision_recall_fscore_support, average_precision_score
from  sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython import embed
from typing import Any, Dict, List, Optional, Type
import pandas as pd
import seaborn as sns
from captum.attr import Occlusion
from captum.attr import visualization as viz
from skimage import color
#import shap

class VAE_class(pl.LightningModule):
    
   def __init__(self, model, params):
      super(VAE_class, self).__init__()
      self.model = model 
      self.params = params
      self.bce = nn.BCELoss() #nn.CrossEntropyLoss(torch.tensor([1.0,0.0,1.0]))  #1.16, 1.86]))  
      self.w_bce = self.params.w_bce
      self.w_kld = self.params.w_kld

   def forward(self, *input, **kwargs): 
       return self.model(*input, **kwargs)


   def on_fit_start(self):
#       self.model.vae_OCT.encoder_layers.requires_grad_(False)
#       self.model.vae_OCT.decoder_layers.requires_grad_(False)
       self.model.vae_OCT.requires_grad_(False)

   def training_step(self, batch, batch_idx):
     x_left, x_right, y, md, ids, maskl, maskr = batch
     y_predict_left, zl = self(x_left,md)
     y_predict_right, zr = self(x_right,md)
     yl = y_predict_left * maskl
     yr = y_predict_right * maskr
     y_predict = (yl + yr)/(1+maskl*maskr)
     bce = self.bce(y_predict, y.double())
     loss = self.w_bce * bce
     acc, precision, sensitivity, specificity = self._shared_eval_step(y_predict, y)
     metrics = {"loss": loss, "Acc":acc, "Precision":precision, "sensitivity": sensitivity, "specificity": specificity}
     self.log_dict(metrics)
     return dict(metrics, **{"y_predict":y_predict.cpu(), "ids":ids, "label":y.cpu(), "zl":zl.cpu(), "zr":zr.cpu()}) #loss_dict  # dict(loss_dict, **{"fpr":fpr, "tpr":tpr}) 


   def training_epoch_end(self, outputs):
        avg_total_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = np.array([x["Acc"] for x in outputs]).mean()
        avg_precision = np.array([x["Precision"] for x in outputs]).mean()
        avg_sensitivity = np.array([x["sensitivity"] for x in outputs]).mean()
        avg_specificity = np.array([x["specificity"] for x in outputs]).mean()
        self._collect_ids(outputs, "train_ids.xlsx")
        self._log_z_vectors(outputs, "latent_vector_train.xlsx")
        self.log_dict(
                {"avg_total_loss":avg_total_loss, "avg_acc_training":avg_acc, "avg_precision_training":avg_precision, "avg_sensitivity_training": avg_sensitivity, "avg_specificity_training":avg_specificity},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
     

   def validation_step(self, batch, batch_idx):
         x_left, x_right, y, md, ids, maskl, maskr = batch
         y_predict_left, zl = self(x_left,md)
         y_predict_right, zr = self(x_right,md)
         yl = y_predict_left * maskl
         yr = y_predict_right * maskr
         y_predict = (yl + yr)/(1+maskl*maskr)
         bce = self.bce(y_predict, y.double())
         val_loss = self.w_bce * bce
         acc, precision, sensitivity, specificity, = self._shared_eval_step(y_predict, y)
         loss_dict = {"val_loss": val_loss, "Acc":acc, "Precision":precision, "sensitivity": sensitivity, "specificity": specificity}
#         self.log_dict(loss_dict)
         return dict(loss_dict, **{"y_predict":y_predict.cpu(), "ids":ids, "label":y.cpu(), "zl":zl.cpu(), "zr":zr.cpu()})  #loss_dict #dict(loss_dict, **{"fpr":fpr, "tpr":tpr})


   def validation_epoch_end(self, outputs):
        avg_total_loss_validation = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = np.array([x["Acc"] for x in outputs]).mean()
        avg_precision = np.array([x["Precision"] for x in outputs]).mean()
        avg_sensitivity = np.array([x["sensitivity"] for x in outputs]).mean()
        avg_specificity = np.array([x["specificity"] for x in outputs]).mean()
        self._collect_ids(outputs, "val_ids.xlsx")
        self._log_z_vectors(outputs, "latent_vector_val.xlsx")
        self.log_dict(
                { "avg_total_loss_validation":avg_total_loss_validation, "avg_acc_validation":avg_acc, "avg_precision_validation":avg_precision, "avg_sensitivity_validation": avg_sensitivity, "avg_specificity_validation":avg_specificity},
            on_epoch=True,
            prog_bar=True,
            logger=True,
          )


   def test_step(self, batch, batch_idx):
       torch.set_grad_enabled(True)
       x_left, x_right, y, md, ids, maskl, maskr = batch
       y_predict_left, zl = self(x_left,md)
       y_predict_right, zr = self(x_right,md)
       yl = y_predict_left * maskl
       yr = y_predict_right * maskr
       y_predict = (yl + yr)/(1+maskl*maskr)
       acc, precision, sensitivity, specificity = self._shared_eval_step(y_predict, y)
       metrics = {"Acc":acc, "Precision":precision, "sensitivity": sensitivity, "specificity": specificity}
       self.log_dict(metrics)
       return dict(metrics, **{"y_predict":y_predict.cpu(), "ids":ids, "label":y.cpu(), "zl":zl.cpu(), "zr":zr.cpu() })#     metrics,y, y_predict 
         

   def test_epoch_end(self, outputs):
        avg_acc = np.array([x["Acc"] for x in outputs]).mean()
        avg_precision = np.array([x["Precision"] for x in outputs]).mean()
        avg_sensitivity = np.array([x["sensitivity"] for x in outputs]).mean()
        avg_specificity = np.array([x["specificity"] for x in outputs]).mean()
        avg_acc = np.array([x["Acc"] for x in outputs]).mean()
        labels=torch.cat([x["label"] for x in outputs])
        y_probas=torch.cat([x["y_predict"] for x in outputs])
        self._collect_ids(outputs, "test_ids.xlsx")
        self._log_z_vectors(outputs, "latent_vector_test.xlsx")
        roc_auc_macro, roc_auc_weight, auc_pr_macro, auc_pr_weigt = self.roc_curve_display(labels, y_probas)
        fpr_tpr = self.roc_curves(labels, y_probas, "fprs.xlsx")
        self.log_dict({"avg_acc": avg_acc, "avg_precision":avg_precision, "avg_sensitivity": avg_sensitivity, "avg_specificity":avg_specificity})

   def _shared_eval_step(self, y_predict, y):
         y = y.cpu()
         y_predict = y_predict.cpu().detach().numpy().round()
         acc, precision, sensitivity, specificity = self.evaluate_metrics(y_predict, y.cpu())
         return acc, precision, sensitivity, specificity 

   def evaluate_metrics(self, y_pred, y_true):
         acc = accuracy_score(y_true, y_pred)
         precision = precision_score(y_true, y_pred, zero_division=0)
         sensitivity = recall_score(y_true, y_pred, zero_division=0)
         tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
         specificity = tn / (tn + fp)
         return acc, precision, sensitivity, specificity

   def roc_curves(self, labels, y_probas, filename=None):
       fpr, tpr, thresholds = roc_curve(labels, y_probas.detach().numpy())
       fpr_tpr = pd.DataFrame(data=np.array([fpr, tpr]).T, columns=["fpr", "tpr"])
       if filename is not None:
           fpr_tpr.to_excel(filename, index=False)
           self.logger.experiment.log_artifact(
                   local_path = filename,
                   artifact_path = "output", run_id=self.logger.run_id)
       return fpr_tpr    


   def _collect_ids(self, outputs, filename=None):
      ids = [x["ids"] for x in outputs]
      ids = [id for sublist in ids for id in sublist]
      y_probas = [x["y_predict"] for x in outputs]
      y_probas = [ p for sublist in y_probas for p in sublist ]
      y_probas = [tensor.detach().numpy() for tensor in y_probas]
      labels = [x["label"] for x in outputs]
      labels = [y for sublist in labels for y in sublist]
      ids_proba = pd.DataFrame(data=np.array([ids, y_probas, labels], dtype=object).T, columns=["IDs", "probas", "labels"])
      if filename is not None:
         ids_proba.to_excel(filename, index=False)
         self.logger.experiment.log_artifact(
                local_path = filename,
                artifact_path = "output", run_id=self.logger.run_id
            )
      return ids

   def _log_z_vectors(self, outputs, filename):    
        zl = torch.concat([x["zl"] for x in outputs])
        zl_columns = [f"z{i:03d}" for i in range(zl.shape[1])] # z001, z002, z003, ...
        zl_df = pd.DataFrame(np.array(zl), columns=zl_columns)
        ids = self._collect_ids(outputs)
        ids = pd.DataFrame(np.array(ids), columns=["IDs"])
        zl_df = pd.concat([ids, zl_df], axis=1)

        zr = torch.concat([x["zr"] for x in outputs])
        zr_columns = [f"z{i:03d}" for i in range(zr.shape[1])] # z001, z002, z003, ...
        zr_df = pd.DataFrame(np.array(zr), columns=zr_columns)
        zr_df = pd.concat([ids, zr_df], axis=1)

        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            zl_df.to_excel(writer, sheet_name="zl", index=False)
            zr_df.to_excel(writer, sheet_name="zr", index=False)

        self.logger.experiment.log_artifact(
            local_path = filename,
            artifact_path = "output", run_id=self.logger.run_id
        ) 

   
   def captum_small_stride(self, x_left, md,ids):
       occlusion = Occlusion(self)
       attributions_occ = occlusion.attribute((x_left,md), strides = ((1, 8, 8),(1,)), sliding_window_shapes=((1,15, 15), (1,)))
       for j in range(self.params.batch_size):
          if attributions_occ[0][j].amax() != 0:
            for i in range(128):
              if attributions_occ[0][j][i].amax() != 0:
                _ = viz.visualize_image_attr_multiple(attributions_occ[0][j][i].cpu().unsqueeze(dim=2).numpy(),
                color.gray2rgb(x_left[j][i].cpu().unsqueeze(dim=2).numpy()),
                ["original_image", "heat_map"],
                ["all", "positive"],
                show_colorbar=True,
                outlier_perc=2,)
                plt.savefig(f'oct_{i}_{ids[j]}_8_15.png')
                plt.cla()
                plt.close('all')
                self.logger.experiment.log_artifact(
                local_path = f'oct_{i}_{ids[j]}_8_15.png',
                artifact_path = "output", run_id=self.logger.run_id)
              else:
                continue
            plt.close('all')   
                
   def captum_big_stride(self, x_left, md,ids):
       occlusion = Occlusion(self)
       attributions_occ = occlusion.attribute((x_left,md), strides = ((1, 50, 50),(1,)), sliding_window_shapes=((1,60, 60), (1,)))
       for j in range(self.params.batch_size):
          if attributions_occ[0][j].amax() != 0:
            for i in range(128):
              if attributions_occ[0][j][i].amax() != 0:
                _ = viz.visualize_image_attr_multiple(attributions_occ[0][j][i].cpu().unsqueeze(dim=2).numpy(),
                color.gray2rgb(x_left[j][i].cpu().unsqueeze(dim=2).numpy()),
                ["original_image", "heat_map"],
                ["all", "positive"],
                show_colorbar=True,
                outlier_perc=2,)
                plt.savefig(f'oct_{i}_{ids[j]}_50_60.png')
                plt.cla()
                plt.close('all')
                self.logger.experiment.log_artifact(
                local_path = f'oct_{i}_{ids[j]}_50_60.png',
                artifact_path = "output", run_id=self.logger.run_id)
              else:
                continue
            plt.close('all')       
                
   def captum_medium_stride(self, x_left, md,ids):
       occlusion = Occlusion(self)
       attributions_occ = occlusion.attribute((x_left,md), strides = ((1, 25, 25),(1,)), sliding_window_shapes=((1,30, 30), (1,)))
       for j in range(self.params.batch_size):
          if attributions_occ[0][j].amax() != 0:
            for i in range(128):
              if attributions_occ[0][j][i].amax() != 0:
                _ = viz.visualize_image_attr_multiple(attributions_occ[0][j][i].cpu().unsqueeze(dim=2).numpy(),
                color.gray2rgb(x_left[j][i].cpu().unsqueeze(dim=2).numpy()),
                ["original_image", "heat_map"],
                ["all", "positive"],
                show_colorbar=True,
                outlier_perc=2,)
                plt.savefig(f'oct_{i}_{ids[j]}_25_30.png')
                plt.cla()
                plt.close('all')
                self.logger.experiment.log_artifact(
                local_path = f'oct_{i}_{ids[j]}_25_30.png',
                artifact_path = "output", run_id=self.logger.run_id)
              else:
                continue
            plt.close('all')   
    
   


   def update_kl(self, w, t):
    t_kl = w*(1.1)**(t-1)
    max_kl = 0.0002
    if t_kl >= max_kl:
       w = max_kl
    else:
       w = t_kl
    return w


   def roc_curve_display(self, y, y_predict):
       macro_roc_auc = roc_auc_score(y, y_predict.detach().numpy())
       weighted_roc_auc = roc_auc_score(y, y_predict.detach().numpy(), average="weighted")
       print(
            "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(weighted)".format(macro_roc_auc, weighted_roc_auc))
       macro_average_precision_score = average_precision_score(y, y_predict.detach().numpy(), average="macro")
       weighted_average_precision_score = average_precision_score( y, y_predict.detach().numpy(),average="weighted")
       print(
               "ROC AUC PR scores:\n{:.6f} (macro),\n{:.6f} "
               "(weighted by prevalence)".format(macro_average_precision_score, weighted_average_precision_score))
       return macro_roc_auc, weighted_roc_auc, macro_average_precision_score, weighted_average_precision_score
       
       

       return macro_roc_auc_ovo, macro_roc_auc_ovr


   def configure_optimizers(self):

         algorithm = self.params.optimizer.algorithm
         algorithm = torch.optim.__dict__[algorithm]
         parameters = vars(self.params.optimizer.parameters)
         optimizer = algorithm(self.model.parameters(), **parameters)
         return optimizer
