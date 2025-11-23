#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multilabel_classifier.py
- Classifier for the multi-label classifier
"""
import os

import fastmri
import numpy as np
import torch
import pytorch_lightning as pl
import torchvision
import torchmetrics
from sklearn.metrics import balanced_accuracy_score

from robustness.model_utils import DummyModel
from robustness.attacker import AttackerModel

import sys

from tqdm import tqdm

sys.path.append('../../')
from util import network_utils, helper
from models.net_builds.build_classifier import build_classifier
from classifier.configs.config_multilabel_classifier import Config


class MultiLabelClassifier(pl.LightningModule):
    
    def __init__(self, config):
        '''

        '''
        super().__init__()
        
        self.save_hyperparameters()
        self.config = config

        # Figure out the size of the inputs
        img_size = config['data_args']['img_size']
        if config['data_args']['challenge'] == 'singlecoil':
            if config['data_args']['complex']:
                # Changed this to include real, imaginary, and magnitude
                self.input_dims = [3, img_size, img_size]
            else:
                self.input_dims = [3, img_size, img_size]
        elif config['data_args']['challenge'] == 'multicoil':
            self.input_dims = [2 * config['data_args']['num_vcoils'], img_size, img_size]
            
        self.network_type = config['net_args']['network_type']
        self.challenge = config['data_args']['challenge']
        self.complex = config['data_args']['complex']

        # Get the acs size
        if 'acs_size' in config['data_args']:
            self.acs_size = config['data_args']['acs_size']
        else:
            self.acs_size = 13

        # Use RSS to combine coils
        self.rss = config['net_args']['rss']

        # Get the number of labels
        self.num_labels = config['data_args']['topk']

        if 'labels' in config['data_args']:
            self.label_names = config['data_args']['labels']
        else:
            self.label_names = None

        # Freeze a set of the features before supervised training
        self.freeze_feats = config['net_args']['freeze_feats']

        # Weight decay
        if 'weight_decay' in config['train_args']:
            self.weight_decay = config['train_args']['weight_decay']
        else:
            self.weight_decay = 0

        # Use adversarial training
        self.adversarial = config['net_args']['adversarial']

        # Define the loss function
        self.bce_weight = config['net_args']['bce_weight']
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.bce_weight, device=self.device))

        # Define adversarial loss parameters
        self.adv_kwargs = config['adversarial_args']
        # Define the custom loss function
        self.adv_loss = AdvLoss([self.bce_weight])
        self.adv_kwargs['custom_loss'] = self.adv_loss.custom_adv_loss

        # Define different metrics to measure
        self.val_accuracy = torchmetrics.Accuracy(task='multilabel', num_labels=self.num_labels, average='macro')
        self.val_accuracy_indiv = torchmetrics.Accuracy(task='multilabel', num_labels=self.num_labels, average=None)
        self.val_precision = torchmetrics.Precision(task='multilabel', num_labels=self.num_labels, average='macro')
        self.val_recall = torchmetrics.Recall(task='multilabel', num_labels=self.num_labels, average='macro')
        self.val_auroc = torchmetrics.AUROC(task='multilabel', num_labels=self.num_labels, average='macro')
        self.val_hamming = torchmetrics.HammingDistance(task='multilabel', num_labels=self.num_labels, average='macro')
        self.f1_score = torchmetrics.F1Score(task='multilabel', num_labels=self.num_labels, average='weighted')

        self.train_auroc = torchmetrics.AUROC(task='multilabel', num_labels=self.num_labels, average='macro')

        # Check if you want to use a pretrained network (i.e. contrastive learning)
        self.pretrained_ckpt = config['net_args']['pretrained_ckpt'] if 'pretrained_ckpt' in config['net_args'] else None

        # Build the network
        self.build()

        # Wrap with adversarial model if needed
        if self.adversarial:
            self.net = AttackerModel(DummyModel(self.net), self.net.t)

        if 'use_posteriors' in config['train_args']:
            self.use_posteriors = config['train_args']['use_posteriors']
            if self.use_posteriors:
                config_file = os.path.join(config['train_args']['cnf_dir'], 'configs.pkl')
                cnf_config = helper.read_pickle(config_file)
                model_type = cnf_config['flow_args']['model_type']
                ckpt_name = 'last.ckpt' if config['train_args']['load_last_ckpt'] else 'best.ckpt'
                ckpt = os.path.join(config['train_args']['cnf_dir'],
                                    'checkpoints',
                                    ckpt_name)
                self.cnf = helper.load_model(model_type, cnf_config, ckpt)
                self.cnf.eval()
                for param in self.cnf.parameters():
                    param.requires_grad = False

        else:
            self.use_posteriors = False
            self.cnf = None


    # Function to build the network
    def build(self):
        # Build the network with pretrained weights
        if self.pretrained_ckpt is not None:
            print('Loading pretrained network')
            ckpt = os.path.join(self.pretrained_ckpt,
                                'checkpoints',
                                'best_val_loss.ckpt')

            # Check if the checkpoint exists, if not, either add or remove '/local' from the path
            if not os.path.exists(ckpt):
                if '/local' in self.pretrained_ckpt:
                    ckpt = ckpt[6:]
                else:
                    ckpt = '/local' + ckpt


            # Get the configuration file
            config_file = os.path.join(self.pretrained_ckpt, 'configs.pkl')
            config_pretrained = helper.read_pickle(config_file)

            # Load the model
            model = helper.load_model(config_pretrained['net_args']['model_type'], config_pretrained, ckpt)

            if config_pretrained['net_args']['model_type'] == 'MAE':
                self.net = model.get_classifier()

            else:
                self.net = model.net

            # Modify the last layer of classification head to have multiple outputs
            self.net.classifier[-1] = torch.nn.Linear(self.net.classifier[-1].in_features, self.num_labels)

        else:
            self.net = build_classifier(self.network_type, input_chans=self.input_dims[0], num_labels=self.num_labels)

        self.transform_mean = self.net.transform_mean
        self.transform_std = self.net.transform_std



    def preprocess_data(self,x, cond, std):
        #Combine the coil images
        if self.rss:
            x = fastmri.rss_complex(network_utils.chans_to_coils(x), dim=1).unsqueeze(1)
        else:
            # Get the maps
            maps = network_utils.get_maps(cond, self.acs_size, std)

            # Get the singlecoil prediction
            x = network_utils.multicoil2single(x, maps)

            # Get the magnitude image
            x = fastmri.complex_abs(x).unsqueeze(1)

        x = self.reformat(x)

        return x



    def reformat(self,x):
        #Expects images to be (batch, 1, img_size, img_size)

        # Repeat the image for RGB channels
        x = x.repeat(1, 3, 1, 1)

        # Normalize to be between 0 and 1
        flattened_imgs = x.view(x.shape[0], -1)
        min_val, _ = torch.min(flattened_imgs, dim=1)
        max_val, _ = torch.max(flattened_imgs, dim=1)
        x = (x - min_val.view(-1, 1, 1, 1)) / (max_val.view(-1, 1, 1, 1) - min_val.view(-1, 1, 1, 1))

        # Define transforms based on pretrained network
        transforms = torchvision.transforms.Normalize(
            mean=self.transform_mean,
            std=self.transform_std
        )

        # Apply the transforms (transforms are applied in adversarial framework so don't apply here)
        if not self.adversarial:
            x = transforms(x)

        return x

    def normalize(self, x):
        # Normalize to be between 0 and 1
        flattened_imgs = x.view(x.shape[0], -1)
        min_val, _ = torch.min(flattened_imgs, dim=1)
        max_val, _ = torch.max(flattened_imgs, dim=1)
        x = (x - min_val.view(-1, 1, 1, 1)) / (max_val.view(-1, 1, 1, 1) - min_val.view(-1, 1, 1, 1))

        return x

    def rim(self, x):
        # Get real, imaginary, and magnitude as the three channels
        mag = fastmri.complex_abs(network_utils.format_multicoil(x, chans=False))

        x = torch.cat([x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1), mag], dim=1)

        return x



    def forward(self, x, cond=None, std=None, target=None, make_adv=False, with_latent=False,
                fake_relu=False, no_relu=False, with_image=True, **attacker_kwargs):


        # if self.complex:
        #     #Preprocess the data
        #     x = self.preprocess_data(x, cond, std).to(self.device)
        if not self.complex:
            x = self.reformat(x).to(self.device)

        # elif self.complex and self.challenge == 'singlecoil':
        #     x = self.normalize(x).to(self.device)

        if self.complex and self.challenge == 'singlecoil':
            x = self.rim(x).to(self.device)

        # Check if this works with complex images
        if self.adversarial:
            if target is not None:
                x, adv = self.net(x, target, make_adv, with_latent, fake_relu, no_relu, with_image=True,
                                  **attacker_kwargs)

                return x, adv

            else:
                if self.complex and self.challenge == 'multicoil':
                    x = self.preprocess_data(x, cond, std)
                x = self.net(x, with_image=False)#.flatten()

        # Non-adversarial network
        else:
            #Get the extracted features
            if self.freeze_feats is not None: #and self.training:
                self.net.feature_extractor.eval()
                with torch.no_grad():
                    feats = self.net.get_features(x)

            else:
                feats = self.net.get_features(x)

            #Classify using the features
            x = self.net.classify(feats)

        # Returns the logits
        return x
    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(),
                                        lr=self.config['train_args']['lr'],
                                        weight_decay=self.weight_decay
                                        )

        # schedulers = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=40, gamma=0.1
        # )

        schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['train_args']['epochs'], eta_min=self.config['train_args']['lr'] / 50
        )

        return [optimizer], [schedulers]

                
    def training_step(self, batch, batch_idx):

        #loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4], device=self.device))

        if self.challenge == 'multicoil' and not self.use_posteriors:
            cond = batch[0]
            x = batch[1]
            std = batch[4]
            y = batch[-1]

            # Unnormalize the image first
            #x = network_utils.unnormalize(x, std)

            if self.adversarial:
                x = self.projected_gradient_descent(x, y, loss_fn=self.loss_fn,
                        num_steps=10, step_size=0.0001, eps=0.000025, eps_norm=2, step_norm=2,
                        cond = cond, std=std)

            #Get the prediction
            ypred = self(x, cond, std)
        elif self.challenge == 'multicoil' and self.use_posteriors:
            c = batch[0].to(self.device)
            #x = batch[1].to(self.device)
            masks = batch[2].to(self.device)
            norm_val = batch[3].to(self.device)
            y = batch[-1]

            samples = self.cnf.reconstruct(c,
                                        num_samples=1,
                                        temp=1.0,
                                        check=True,
                                        maps=None,
                                        mask=masks,
                                        norm_val=norm_val,
                                        split_num=4,
                                        multicoil=False,
                                        rss=True)

            x = torch.cat(samples,dim=0)

            if self.adversarial:
                # Do the adversarial prediction
                # Note: since targeted is False by default in adv_kwargs, this does gradient ascent to maximize the loss
                ypred, im_adv = self(x, target=y, make_adv=True, **self.adv_kwargs)

            else:
                #Get the prediction
                ypred = self(x)
        else:
            x = batch[0]
            y = batch[-1]



            if self.adversarial:
                # Do the adversarial prediction
                # Note: since targeted is False by default in adv_kwargs, this does gradient ascent to maximize the loss
                ypred, im_adv = self(x, target=y, make_adv=True, **self.adv_kwargs)

            else:
                #Get the prediction
                ypred = self(x)


        #ypred = ypred.flatten()

        loss = self.loss_fn(ypred,y.float())

        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.train_auroc(ypred.detach(), y.int())
        self.log('Train AUROC', self.train_auroc, on_step=True)

        return loss


    
    def validation_step(self, batch, batch_idx):

        #loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4], device=self.device))

        if self.challenge == 'multicoil':
            cond = batch[0]
            x = batch[1]
            std = batch[4]
            y = batch[-1]

            # Unnormalize the image first
            #x = network_utils.unnormalize(x, std)

            # if self.adversarial:
            #     x = self.projected_gradient_descent(x, y, loss_fn=loss_fn,
            #             num_steps=40, step_size=0.0001, eps=0.000025, eps_norm=2, step_norm=2,
            #             cond = cond, std=std)

            # Get the prediction
            ypred = self(x, cond, std)

        elif self.challenge == 'multicoil' and self.use_posteriors:
            c = batch[0].to(self.device)
            #x = batch[1].to(self.device)
            masks = batch[2].to(self.device)
            norm_val = batch[3].to(self.device)
            y = batch[-1]

            samples = self.cnf.reconstruct(c,
                                        num_samples=1,
                                        temp=1.0,
                                        check=True,
                                        maps=None,
                                        mask=masks,
                                        norm_val=norm_val,
                                        split_num=4,
                                        multicoil=False,
                                        rss=True)

            x = torch.cat(samples,dim=0)

            if self.adversarial:
                # Do the adversarial prediction
                # Note: since targeted is False by default in adv_kwargs, this does gradient ascent to maximize the loss
                ypred, im_adv = self(x, target=y.unsqueeze(1), make_adv=True, **self.adv_kwargs)

            else:
                #Get the prediction
                ypred = self(x)
        else:
            x = batch[0]
            y = batch[-1]

            # if self.adversarial:
            #     x = self.projected_gradient_descent(x, y, loss_fn=loss_fn,
            #             num_steps=40, step_size=0.0001, eps=0.000025, eps_norm=2, step_norm=2)

            # Get the prediction
            ypred = self(x)

        #ypred = ypred.flatten()


        loss = self.loss_fn(ypred, y.float())

        self.log('val_loss', loss, on_epoch=True)

        self.val_accuracy(ypred.detach(),y.int())
        self.log('Val Accuracy', self.val_accuracy, on_epoch=True)
        self.val_precision(ypred.detach(),y.int())
        self.log('Val Precision', self.val_precision, on_epoch=True)
        self.val_recall(ypred.detach(),y.int())
        self.log('Val Recall', self.val_recall, on_epoch=True)
        self.val_auroc(ypred.detach(),y.int())
        self.log('Val AUROC', self.val_auroc, on_epoch=True)
        self.val_hamming(ypred.detach(),y.int())
        self.log('Val Hamming', self.val_hamming, on_epoch=True)
        self.f1_score(ypred.detach(),y.int())
        self.log('Val F1 Score', self.f1_score, on_epoch=True)

        #self.val_accuracy_indiv(ypred.detach(),y.int())
        #self.log('Val Accuracy Indiv', self.val_accuracy_indiv, on_epoch=True)

        # Find the balanced accuracy
        # bal_acc = balanced_accuracy_score(y.cpu().int(), torch.round(torch.sigmoid(ypred.detach())).cpu().int())
        # self.log('Val Bal Accuacy', bal_acc, on_epoch=True)




    # Evaluate the model with different metrics and save the results to the save dir
    def test(self, data, save_dir):

        test_loader = data.val_dataloader()
        label_names = data.val.label_names

        # Set the model to evaluation
        self.eval()

        # Initialize the metrics
        self.test_accuracy = torchmetrics.Accuracy(task='multilabel', num_labels=self.num_labels, average='macro').to(self.device)
        self.test_precision = torchmetrics.Precision(task='multilabel', num_labels=self.num_labels, average='macro').to(self.device)
        self.test_recall = torchmetrics.Recall(task='multilabel', num_labels=self.num_labels, average='macro').to(self.device)
        self.test_auroc = torchmetrics.AUROC(task='multilabel', num_labels=self.num_labels, average='macro').to(self.device)

        # Initialize the metrics for each individual label
        self.test_accuracy_indiv = torchmetrics.Accuracy(task='multilabel', num_labels=self.num_labels, average=None).to(self.device)
        self.test_precision_indiv = torchmetrics.Precision(task='multilabel', num_labels=self.num_labels, average=None).to(self.device)
        self.test_recall_indiv = torchmetrics.Recall(task='multilabel', num_labels=self.num_labels, average=None).to(self.device)
        self.test_auroc_indiv = torchmetrics.AUROC(task='multilabel', num_labels=self.num_labels, average=None).to(self.device)

        print('Testing')

        for i, batch in tqdm(enumerate(test_loader)):
            if self.challenge == 'multicoil':
                cond = batch[0]
                x = batch[1]
                std = batch[4]
                y = batch[-1]

                # Unnormalize the image first
                # x = network_utils.unnormalize(x, std)

                # if self.adversarial:
                #     x = self.projected_gradient_descent(x, y, loss_fn=loss_fn,
                #             num_steps=40, step_size=0.0001, eps=0.000025, eps_norm=2, step_norm=2,
                #             cond = cond, std=std)

                # Get the prediction
                ypred = self(x, cond, std)

            elif self.challenge == 'multicoil' and self.use_posteriors:
                c = batch[0].to(self.device)
                # x = batch[1].to(self.device)
                masks = batch[2].to(self.device)
                norm_val = batch[3].to(self.device)
                y = batch[-1]

                samples = self.cnf.reconstruct(c,
                                               num_samples=1,
                                               temp=1.0,
                                               check=True,
                                               maps=None,
                                               mask=masks,
                                               norm_val=norm_val,
                                               split_num=4,
                                               multicoil=False,
                                               rss=True)

                x = torch.cat(samples, dim=0)

                if self.adversarial:
                    # Do the adversarial prediction
                    # Note: since targeted is False by default in adv_kwargs, this does gradient ascent to maximize the loss
                    ypred, im_adv = self(x, target=y.unsqueeze(1), make_adv=True, **self.adv_kwargs)

                else:
                    # Get the prediction
                    ypred = self(x)
            else:
                x = batch[0].to(self.device)
                y = batch[-1].to(self.device)

                # if self.adversarial:
                #     x = self.projected_gradient_descent(x, y, loss_fn=loss_fn,
                #             num_steps=40, step_size=0.0001, eps=0.000025, eps_norm=2, step_norm=2)

                # Get the prediction
                ypred = self(x)

            # ypred = ypred.flatten()

            # Get the metrics
            self.test_accuracy.update(ypred.detach(), y.int())
            self.test_precision.update(ypred.detach(), y.int())
            self.test_recall.update(ypred.detach(), y.int())
            self.test_auroc.update(ypred.detach(), y.int())

            self.test_accuracy_indiv.update(ypred.detach(), y.int())
            self.test_precision_indiv.update(ypred.detach(), y.int())
            self.test_recall_indiv.update(ypred.detach(), y.int())
            self.test_auroc_indiv.update(ypred.detach(), y.int())




        # Compute the metrics
        accuracy = self.test_accuracy.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        auroc = self.test_auroc.compute()

        accuracy_indiv = self.test_accuracy_indiv.compute()
        precision_indiv = self.test_precision_indiv.compute()
        recall_indiv = self.test_recall_indiv.compute()
        auroc_indiv = self.test_auroc_indiv.compute()

        # Save the metrics to the save dir in a table with columns as the metrics and rows as the individual labels and average
        metrics = np.array([accuracy.cpu().numpy(), precision.cpu().numpy(), recall.cpu().numpy(), auroc.cpu().numpy()])
        metrics_indiv = np.vstack([accuracy_indiv.cpu().numpy(), precision_indiv.cpu().numpy(), recall_indiv.cpu().numpy(), auroc_indiv.cpu().numpy()])
        print(metrics_indiv.shape)
        print(metrics.shape)
        metrics = np.vstack([metrics, metrics_indiv.transpose()])
        metrics = np.round(metrics, 4)
        metrics = np.vstack([['Accuracy', 'Precision', 'Recall', 'AUROC'], metrics])

        labels = np.array(['Label'] + ['Average'] + [f'{label_names[i]}' for i in range(self.num_labels)])

        metrics = np.hstack([labels.reshape(-1, 1), metrics])

        np.savetxt(os.path.join(save_dir, 'metrics.csv'), metrics, delimiter=',', fmt='%s')






class AdvLoss:

    def __init__(self, weights):
        self.weights = weights

    def custom_adv_loss(self, model, inp, target):
        #Need to return a loss for each input so reduction = 'none', mean is taken later in the robustness autograd
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.array(self.weights), device=target.device), reduction='none')
        pred = model(inp)
        loss = criterion(pred, target.float())
        #print(loss.shape)
        #loss = self.loss_fn(pred, target.float())

        return loss, None


if __name__ == '__main__':
    #Get the configurations
    config = Config()
    config = config.config
    config['train_args']['freeze_feats'] = 'all'

    # Set the input dimensions
    img_size = config['data_args']['img_size']
    input_dims = [3, img_size, img_size]

    # Initialize the network
    model = MultiLabelClassifier(config).cuda()


    x = torch.zeros(5,1,320,320).cuda()
    y = torch.zeros(5,5).cuda()

    #model = AttackerModel(model)


    #ypred = model(x)

    ypred, im_adv = model(x, target=y, make_adv=True, **model.adv_kwargs)