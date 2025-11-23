'''
tasks.py
- Classes to define the downstream tasks
'''

import os
import math
import torch
import numpy as np
from DISTS_pytorch import DISTS
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import sys

sys.path.append("../")

from util import helper



def get_tasks(task_names, cf):
    task_types = []

    for name in task_names:
        if 'metric' in name:
            task = QualityMetricsTask(task_names, cf)
            task_types.append(task)
            break
    for name in task_names:
        if 'classify' in name:
            task = ClassifyTask(task_names, cf)
            task_types.append(task)
            break

    return task_types


# Generic class for tasks
class GeneralTask():
    def __init__(self, task_names, cf):
        self.task_names = task_names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cf = cf


    def compute_task_output(self, reference, posteriors, gt):
        # Output should be
        pass



# Class to compute the quality metrics
class QualityMetricsTask(GeneralTask):
    def __init__(self, task_names, cf):
        super().__init__(task_names, cf)

        self.metrics = {}

        # Define the metrics
        for name in task_names:
            if name not in ['metric_dists', 'metric_ssim', 'metric_psnr', 'metric_lpips']:
                continue
            self.metrics[name] = self.find_metric(name)


    def find_metric(self, name):
        if name == 'metric_dists':
            metric = DISTS().to(self.device)
        elif name == 'metric_ssim':
            metric = StructuralSimilarityIndexMeasure(reduction=None).to(self.device)
        elif name == 'metric_psnr':
            metric = PSNR(device=self.device)
            #metric = PeakSignalNoiseRatio(reduction=None).to(self.device)
        elif name == 'metric_lpips':
            metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg',normalize=True).to(self.device)
        else:
            raise ValueError(f'Metric {name} not recognized')

        return metric


    def compute_task_output(self, reference, posteriors, gt):


        # Get the number of posterior samples
        p = posteriors.shape[0]

        # Repeat the reference to match the number of posteriors
        reference_repeat = reference.repeat(p, 1, 1, 1)

        # Store all the outputs
        outputs = {}

        # Compute the all the metrics
        for metric_name, metric in self.metrics.items():

            # For some reason, LPIPS either sums or averages in the batch dimension so do it separately
            if metric_name == 'metric_lpips': # or metric_name == 'metric_psnr':
                metrics_p = []
                for i in range(p):
                    metrics_p.append(self.compute_metric(reference_repeat[i].unsqueeze(0), posteriors[i].unsqueeze(0),
                                                         metric_name,
                                                         metric))
                    metric_p = np.stack(metrics_p, axis=0)
            else:
                metric_p = self.compute_metric(reference_repeat, posteriors, metric_name, metric)

            #print(metric_p.shape)


            # Compute the ground truth metric
            if len(gt.shape) < 4:
                gt = gt.unsqueeze(0)
            metric_gt = self.compute_metric(reference, gt, metric_name, metric)

            #print(metric_gt.shape)

            # Store the metric outputs
            outputs[metric_name] = {'posteriors': metric_p, 'gt': metric_gt}


        return outputs




    # Compute the metric
    def compute_metric(self, recons, gt, metric_name, metric):
        if metric_name == 'metric_dists' or metric_name == 'metric_lpips':
            return self.compute_rgb_metric(recons, gt, metric)
        elif metric_name == 'metric_ssim' or metric_name == 'metric_psnr':
            return self.compute_single_channel_metric(recons, gt, metric)
        else:
            raise ValueError('Metric not implemented')

    def compute_rgb_metric(self, recon, gt, metric):
        """
        Compute the DISTS or LPIPS metric between the reconstruction and the ground truth
        :param recon: (b, 1, h, w) tensor of the reconstruction
        :param gt: (b, 1, h, w) tensor of the ground truth
        :return: DISTS metric
        """

        # Repeat the channel dimensions
        if recon.shape[1] == 1:
            recon = to_rgb(recon)
            gt = to_rgb(gt)

        # Normalize to be between 0 and 1
        recon = normalize(recon)
        gt = normalize(gt)

        return metric(recon, gt).detach().cpu().numpy()

    def compute_single_channel_metric(self, recon, gt, metric):
        """
        Compute the SSIM or PSNR metric between the reconstruction and the ground truth
        :param recon: (b, 1, h, w) tensor of the reconstruction
        :param gt: (b, 1, h, w) tensor of the ground truth
        :return: SSIM or PSNR metric
        """

        return metric(recon, gt).detach().cpu().numpy()




def normalize(x):
    # Normalize to be between 0 and 1
    flattened_imgs = x.view(x.shape[0], -1)
    min_val, _ = torch.min(flattened_imgs, dim=1)
    max_val, _ = torch.max(flattened_imgs, dim=1)
    x = (x - min_val.view(-1, 1, 1, 1)) / (max_val.view(-1, 1, 1, 1) - min_val.view(-1, 1, 1, 1))

    return x

def to_rgb(x):
    '''
    Repeat the channel dimensions to make the tensor RGB
    :param x: (b, 1, h, w) tensor
    :return:
    '''

    # Repeat the channel dimensions
    x = x.repeat(1, 3, 1, 1)
    return x

class PSNR:
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, recon, gt):
        '''
        Compute the PSNR between the reconstruction and the ground truth
        :param recon: (n, c, h, w) tensor of the reconstruction
        :param gt: (n, c, h, w) tensor of the ground truth
        :return:
        '''

        if isinstance(recon, np.ndarray):
            recon = torch.tensor(recon)
        if isinstance(gt, np.ndarray):
            gt = torch.tensor(gt)

        recon = recon.to(self.device)
        gt = gt.to(self.device)

        se = (recon - gt) ** 2
        mse = torch.mean(se, dim=(1, 2, 3))
        max_intensity = torch.max(gt.reshape(gt.shape[0], -1), dim=1)[0]
        psnr = 10 * torch.log10(max_intensity ** 2 / mse)

        #print(psnr.shape)

        return psnr.squeeze()




class ClassifyTask(GeneralTask):
    def __init__(self, task_names, cf):
        super().__init__(task_names, cf)

        self.classifier = self.load_classifier(cf.model.classifier_load_ckpt_dir)


    # Load the classifier
    def load_classifier(self, classifier_dir):

        # Load the previous configuration
        ckpt = os.path.join(classifier_dir,
                            'checkpoints',
                            'best_val_loss.ckpt') # TODO: Changed this to the best val auroc
        config = helper.read_pickle(os.path.join(classifier_dir, 'configs.pkl'))


        # Load the classifer model
        classifier = helper.load_model('MultiLabelClassifier', config, ckpt)

        return classifier




    def compute_task_output(self, reference, posteriors, gt):

        # Prep the classifier for evaluation
        self.classifier.eval()
        self.classifier.to(self.device)

        # Put posterior and gt on the device
        posteriors = posteriors.to(self.device)
        gt = gt.to(self.device)

        # Get the posterior predictions from the classifier
        # Potentially split up posterior samples so that they can fit in memory
        if posteriors.shape[0] > 64:
            ypred = self.classifier(posteriors[:64])
            yhat = torch.sigmoid(ypred).detach().cpu()
            for k in range(1, math.ceil(posteriors.shape[0] / 64)):
                ypred = self.classifier(posteriors[k * 64:(k + 1) * 64])
                yhat = torch.cat((yhat, torch.sigmoid(ypred).detach().cpu()), dim=0)

        else:
            ypred = self.classifier(posteriors)
            yhat = torch.sigmoid(ypred).detach().cpu()

        # Get the ground truth predictions
        ypred = self.classifier(gt)
        yhat_gt = torch.sigmoid(ypred).detach().cpu()

        # Store all the outputs
        outputs = {}

        # Compute the all the metrics
        for l, label_name in enumerate(self.classifier.label_names):

            # Store the metric outputs
            outputs[label_name] = {'posteriors': yhat[:,l], 'gt': yhat_gt[:,l]}


        return outputs

