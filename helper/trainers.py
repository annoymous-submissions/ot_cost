
ROOT_DIR = ''
import pandas as pd
import os
import copy
import torch
import torch.nn as nn
import sys
import numpy as np
sys.path.append(f'{ROOT_DIR}/code/helper')
import data_preprocessing as dp
import importlib
importlib.reload(dp)
from sklearn import metrics
import torch.nn.functional as F
from scipy.spatial.distance import cosine

SQUEEZE = ['Synthetic', 'Credit']
LONG = ['EMNIST', 'CIFAR', 'ISIC']
CLASS_ADJUST = ['EMNIST', 'CIFAR']
TENSOR = ['IXITiny']
CONTINUOUS_OUTCOME = ['Weather']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer:
    def __init__(self, EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, device, patience = 20):
        self.DATASET = DATASET
        self.model = model
        self.best_model = copy.deepcopy(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.device = device
        self.best_loss = float('inf')
        self.counter = 0
        self.EPOCHS = EPOCHS
        self.WARMUP_STEPS = WARMUP_STEPS
        self.pfedme = False # for subclass trainers
        self.ditto = False # for subclass trainers

    def set_loader(self, site_data):
        self.train_loader, self.val_loader, self.test_loader = site_data
        return

    def get_objects(self, site_number = None, data = True):
        if site_number == None:
            if data:
                return self.model, self.criterion, self.optimizer, self.lr_scheduler, self.train_loader, self.val_loader, self.test_loader
            else:
                return self.model, self.criterion, self.optimizer, self.lr_scheduler
        else:
            #Federated setting
            if isinstance(site_number, int):
                if site_number == 1:
                    if data:
                        return (self.model_1, self.criterion_1, self.optimizer_1, self.lr_scheduler_1, self.train_loader_1, self.val_loader_1, self.test_loader_1)
                    else: 
                        (self.model_1, self.criterion_1, self.optimizer_1, self.lr_scheduler_1)
                else:
                    if data:
                        return (self.model_2, self.criterion_2, self.optimizer_2, self.lr_scheduler_2, self.train_loader_2, self.val_loader_2, self.test_loader_2)
                    else:
                        return (self.model_2, self.criterion_2, self.optimizer_2, self.lr_scheduler_2)
            else:
                #Ditto model - this is the shared model and not the personal one
                if site_number == '1s':
                    if data:
                        return (self.model_1_s, self.criterion_1_s, self.optimizer_1_s, self.lr_scheduler_1_s,self.train_loader_1, self.val_loader_1, self.test_loader_1 )
                    else:
                        return (self.model_1_s, self.criterion_1_s, self.optimizer_1_s, self.lr_scheduler_1_s)
                else:
                    if data:
                        return (self.model_2_s, self.criterion_2_s, self.optimizer_2_s, self.lr_scheduler_2_s, self.train_loader_2, self.val_loader_2, self.test_loader_2)
                    else:
                        return (self.model_2_s, self.criterion_2_s, self.optimizer_2_s, self.lr_scheduler_2_s)

    def combined_scheduler(self, scheduler):
        warmup_scheduler, exp_scheduler = scheduler
        if self.step < self.WARMUP_STEPS:
            warmup_scheduler.step()
        else:
            exp_scheduler.step()

    def train_one_epoch(self, site = None):
        model, criterion, optimizer, lr_scheduler, train_loader, _, _ = self.get_objects(site)
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            outputs = model(x)
            if self.DATASET in SQUEEZE:
                y = y.unsqueeze(1)
            elif self.DATASET in LONG:
                y = y.long()
            loss = criterion(outputs, y)
            if self.pfedme:
                loss = self.pfedme_loss(site, loss)
            loss.backward()
            optimizer.step()
            self.combined_scheduler(lr_scheduler)
            self.step +=1
            train_loss += loss.item()
        return train_loss / len(train_loader)

    def validate(self, site = None):
        model, criterion, _, _, _, val_loader, _ = self.get_objects(site)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                if self.DATASET in SQUEEZE:
                    y = y.unsqueeze(1)
                elif self.DATASET in LONG:
                    y = y.long()
                loss = criterion(outputs, y)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def check_early_stopping(self, val_loss, val_losses, epoch, site = None):
        model, _, _, _, _, _, _ = self.get_objects(site)
        val_loss_ab = abs(val_loss)
        if val_loss_ab < self.best_loss:
            self.best_loss = val_loss_ab
            self.best_model = copy.deepcopy(model)
            self.counter = 0
        elif epoch < 5: # Let the model have at least 5 epochs of training
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def get_metric(self):
        metric_mapping = {
            'Synthetic': metrics.f1_score,
            'Credit': metrics.f1_score,
            'Weather': metrics.r2_score,
            'EMNIST': metrics.accuracy_score,
            'CIFAR': metrics.accuracy_score,
            'IXITiny': get_dice_score,
            'ISIC': metrics.balanced_accuracy_score}
        return metric_mapping[self.DATASET]

    def test(self, site = None):
        _, criterion, _, _, _, _, test_loader = self.get_objects(site)
        test_loss = 0
        self.best_model.eval()
        with torch.no_grad():
            predictions_list = []
            true_labels_list = []
            test_loss = 0
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.best_model(x)
                if self.DATASET in SQUEEZE:
                    y = y.unsqueeze(1)
                elif self.DATASET in LONG:
                    y = y.long()
                loss = criterion(predictions, y)
                test_loss += loss.item()
                predictions_list.extend(predictions.cpu().numpy())
                true_labels_list.extend(y.cpu().numpy())
            test_loss /= len(test_loader)
            predictions_array = np.array(predictions_list)
            if self.DATASET in CONTINUOUS_OUTCOME :
                predictions_array = np.clip(predictions_array, -2, 2) #clipping of values well off normalised labels
            elif self.DATASET in SQUEEZE:
                predictions_array = (predictions_array >= 0.5).astype(int) # set predictions for F1 score
            elif self.DATASET in LONG:
                    predictions_array = predictions_array.argmax(axis = 1) 
            true_labels_array = np.array(true_labels_list)
            metric_assess = self.get_metric()
            if self.DATASET in TENSOR:
                score = metric_assess(torch.tensor(true_labels_array, dtype=torch.float32), torch.tensor(predictions_array,dtype=torch.float32))
            else:
                score = metric_assess(true_labels_array, predictions_array)
            return test_loss, score

    def run(self):
        train_losses = []
        val_losses = []
        self.step = 0
        for epoch in range(self.EPOCHS):
            train_loss = self.train_one_epoch()
            train_losses.append(train_loss)
            val_loss = self.validate()
            val_losses.append(val_loss)
            if self.check_early_stopping(val_loss, val_losses, epoch):
                break
        print(f'Stopping at epoch: {epoch}')
        test_loss, score = self.test()
        return score, test_loss, val_losses, train_losses

class FederatedModelTrainer(ModelTrainer):
    def __init__(self, EPOCHS, WARMUP_STEPS, DATASET,  model, criterion, optimizer, lr_scheduler, device, pfedme = False, pfedme_reg =1e-1):
        super().__init__(EPOCHS, WARMUP_STEPS, DATASET, model,criterion, optimizer, lr_scheduler, device)
        self.model_1, self.criterion_1, self.optimizer_1, self.lr_scheduler_1 = self.clone_model()
        self.model_2, self.criterion_2, self.optimizer_2, self.lr_scheduler_2 = self.clone_model()
        self.pfedme = pfedme
        self.pfedme_reg = pfedme_reg
        self.ROUNDS = 1
        self.gradient_diversity_metric = []
        self.gradient_diversity_cosine = []

    def set_loader(self, site_1_data, site_2_data):
        self.train_loader_1, self.val_loader_1, self.test_loader_1 = site_1_data
        self.train_loader_2, self.val_loader_2, self.test_loader_2 = site_2_data
        #Set weights for fedavg
        total_samples_1, total_samples_2 = len(self.train_loader_1.dataset), len(self.train_loader_2.dataset)
        self.weight_1 = total_samples_1 / (total_samples_1 + total_samples_2)
        self.weight_2 = total_samples_2 / (total_samples_1 + total_samples_2)
        assert abs((self.weight_1 + self.weight_2) - 1) < 1e-2, "Issue with the weights: They don't sum up to approximately 1."
        return
    
    def clone_model(self):
        model_clone = copy.deepcopy(self.model)
        criterion_clone = copy.deepcopy(self.criterion)
        optimizer_clone = type(self.optimizer)(model_clone.parameters(), **self.optimizer.defaults)
        lr_scheduler_clone = (type(self.lr_scheduler[0])(optimizer_clone, lr_lambda=self.lr_scheduler[0].lr_lambdas[0]), 
                              type(self.lr_scheduler[1])(optimizer_clone, gamma = self.lr_scheduler[1].gamma))
        return model_clone, criterion_clone, optimizer_clone, lr_scheduler_clone

    def compare_state_dicts(self, dict1, dict2):
        for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
            if k1 != k2 or not torch.allclose(v1, v2):
                return False
        return True

    def fed_avg(self, site_1, site_2):
        model, _, _, _, = self.get_objects(None, data = False) # server model
        model_1, _, _, _, _, _, _ = self.get_objects(site_1) # site 1
        model_2, _, _, _, _, _, _ = self.get_objects(site_2) # site 2
        
        # fed avg using site weights
        tmp_model = copy.deepcopy(model) 
        for name, param in model.named_parameters():
            weighted_avg_param = (self.weight_1 * model_1.state_dict()[name] + self.weight_2 * model_2.state_dict()[name])
            tmp_model.state_dict()[name].copy_(weighted_avg_param)
        #update server model
        model.load_state_dict(tmp_model.state_dict())
        #if basic fedavg then update local models
        if not self.pfedme:
            if not self.ditto:
                self.gradient_diversity(model_1, model_2)
            model_1.load_state_dict(model.state_dict())
            model_2.load_state_dict(model.state_dict())  
            assert self.compare_state_dicts(model.state_dict(), model_1.state_dict()) and self.compare_state_dicts(model.state_dict(), model_2.state_dict()), "The model weights are not identical."
        return 

    def pfedme_loss(self, site, loss):
        regularization_loss = 0
        model_c, _, _, _, = self.get_objects(None, data = False) # server model
        model_p, _, _, _, _, _, _ = self.get_objects(site) # site model
        for p, g_p in zip(model_p.parameters(), model_c.parameters()):
            regularization_loss += torch.norm(p - g_p)
        regularization_loss = self.pfedme_reg * regularization_loss
        loss += regularization_loss
        return loss
    
    def gradient_diversity(self, model_1, model_2):
        #Get model gradients
        grads = {'model_1': [], 'model_2':[]}
        for name, param in model_1.named_parameters():
                if param.requires_grad:
                    grads['model_1'].append(param.grad.clone().detach())
        for name, param in model_2.named_parameters():
                if param.requires_grad:  
                    grads['model_2'].append(param.grad.clone().detach())
        #wrangle
        model_1_g = torch.cat([w.flatten() for w in grads['model_1']])
        model_2_g = torch.cat([w.flatten() for w in grads['model_2']])
        #calculate diversity
        num = torch.norm(model_1_g)**2  + torch.norm(model_2_g)**2
        denom = torch.norm(model_1_g + model_2_g)**2
        gradient_diversity_metric = (num/ denom).cpu().numpy().reshape(1)[0]
        self.gradient_diversity_metric.append(gradient_diversity_metric)
        cosine_gradient_diversity = cosine(model_1_g.cpu().numpy().astype(float), model_2_g.cpu().numpy().astype(float))
        self.gradient_diversity_cosine.append(cosine_gradient_diversity)
        return

    def run(self):
        train_losses = []
        val_losses = []
        self.step = 0
        for epoch in range(self.EPOCHS):
            train_loss = self.train_one_epoch(1)
            train_loss +=  self.train_one_epoch(2)
            train_losses.append(train_loss)
            self.fed_avg(1,2)
            val_loss = self.validate(1)
            val_loss += self.validate(2)
            val_losses.append(val_loss)
            if self.check_early_stopping(val_loss, val_losses, epoch, 1):
                break
        print(f'Stopping at epoch: {epoch}')
        test_loss, score = self.test(1)
        return score, test_loss, val_losses, train_losses

class DittoModelTrainer(FederatedModelTrainer):
    def __init__(self, EPOCHS, WARMUP_STEPS, DATASET,  model, criterion, optimizer, lr_scheduler, device, reg = 1e-1):
        super().__init__(EPOCHS, WARMUP_STEPS, DATASET, model,criterion, optimizer, lr_scheduler, device)
        self.model_1_s, self.criterion_1_s, self.optimizer_1_s, self.lr_scheduler_1_s = self.clone_model() # models sent server
        self.model_2_s, self.criterion_2_s, self.optimizer_2_s, self.lr_scheduler_2_s = self.clone_model()
        self.ditto = True
        self.reg = reg

    def train_site_personal_epoch(self, site):
        model_site_p, criterion_p, optimizer_site_p, lr_scheduler_site_p, train_loader, _ , _  = self.get_objects(site) # personal model
        model_site, _, _, _, _, _, _ = self.get_objects(f'{site}s') # server sent model
        train_loss = 0
        for i in range(self.ROUNDS):
            model_site_p.train()
            for x, y in train_loader:
                model_site_p.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                if self.DATASET in SQUEEZE:
                    y = y.unsqueeze(1)
                elif self.DATASET in LONG:
                    y = y.long()
                outputs = model_site_p(x.float())
                loss = criterion_p(outputs, y)
                loss = self.ditto_loss(model_site, model_site_p, loss)
                train_loss += loss.item()
                loss.backward()
                optimizer_site_p.step()
                self.combined_scheduler(lr_scheduler_site_p)
                self.step += 1
            train_loss /= len(train_loader)
        return train_loss
    
    def ditto_loss(self, model_site, model_site_p, loss):
        regularization_loss = 0
        for p, g_p in zip(model_site_p.parameters(), model_site.parameters()):
            regularization_loss += torch.norm(p - g_p)
        regularization_loss = self.reg * regularization_loss
        loss += regularization_loss
        return loss

    def run(self):
        train_losses = []
        val_losses = []
        self.step = 0
        self.step_sent = 0
        for epoch in range(self.EPOCHS):
            _ = self.train_one_epoch('1s')
            _ =  self.train_one_epoch('2s')
            self.fed_avg('1s', '2s' )
            train_loss = self.train_site_personal_epoch(1)
            train_loss +=  self.train_site_personal_epoch(2)
            train_losses.append(train_loss)
            val_loss = self.validate(1)
            val_loss += self.validate(2)
            if self.check_early_stopping(val_loss, val_losses, epoch, 1):
                break
            val_losses.append(val_loss)
        print(f'Stopping at epoch: {epoch}')
        test_loss, score = self.test(1)
        return score, test_loss, val_losses, train_losses
    
def get_dice_score(output, target, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(axis=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(axis=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(axis=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score.mean().item()

def get_soft_dice_metric(y_pred, y_true, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon=1e-9):
    """
    Soft Dice coefficient
    """
    intersection = (y_pred * y_true).sum(axis=SPATIAL_DIMENSIONS)
    union = (0.5 * (y_pred + y_true)).sum(axis=SPATIAL_DIMENSIONS)
    dice = intersection / (union + epsilon)
    # If both inputs are empty the dice coefficient should be equal 1
    dice[union == 0] = 1
    return np.mean(dice)