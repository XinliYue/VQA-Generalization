# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import torch.optim as optim

from data_loader import VideoDataset_images_with_motion_features
from utils import performance_fit
from utils import L1RankLoss



from model import UGC_BVQA_model

from torchvision import transforms
import time

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr

class AWP:
    def __init__(self, model, criterion, optimizer, adv_param="weight", gamma=0.0001):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.gamma = gamma
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, video, feature_3D, labels):
        if self.gamma == 0:
            return
        self._save()
        self._attack_step()

        outputs = self.model(video, feature_3D)
        adv_loss = self.criterion(labels, outputs)
        self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.gamma * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.gamma * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

import os
import random
import sys

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import logging
def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(handler)

setup_logger()
logger = logging.getLogger()

def main(config):

    seed = 42
    logger.info(f'seed: {seed}')
    set_seed(seed)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    if config.model_name == 'UGC_BVQA_model':
        logger.info(f'The current model is {config.model_name}')
        model = UGC_BVQA_model.resnet50(pretrained=True)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)
    

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.conv_base_lr, weight_decay = 0.0000001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    if config.loss_type == 'L1RankLoss':
        criterion = L1RankLoss()

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    logger.info('Trainable params: %.2f million' % (param_num / 1e6))
       

    if config.database == "konvid":
        videos_dir = '/path/to/konvid1k_image'
        feature_dir = '/path/to/konvid1k_SlowFast_feature'
        datainfo_train = '/path/to/labels_train.txt'
        datainfo_test = '/path/to/labels_test.txt'
        name = 'KoNViD-1k'
    elif config.database == "livevqc":
        videos_dir = '/path/to/livevqc_image'
        feature_dir = '/path/to/livevqc_SlowFast_feature'
        datainfo_train = '/path/to/labels_train.txt'
        datainfo_test = '/path/to/labels_test.txt'
        name = 'LIVE_VQC'
    elif config.database == "youtubeugc":
        videos_dir = '/path/to/youtubeugc_image'
        feature_dir = '/path/to/youtubeugc_SlowFast_feature'
        datainfo_train = '/path/to/labels_train.txt'
        datainfo_test = '/path/to/labels_test.txt'
        name = 'YouTubeUGC'
    transformations_train = transforms.Compose([transforms.Resize(config.resize), transforms.RandomCrop(config.crop_size), transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose([transforms.Resize(config.resize),transforms.CenterCrop(config.crop_size),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        
    trainset = VideoDataset_images_with_motion_features(videos_dir, feature_dir, datainfo_train, transformations_train, name, config.crop_size, 'SlowFast')
    testset = VideoDataset_images_with_motion_features(videos_dir, feature_dir, datainfo_test, transformations_test, name, config.crop_size, 'SlowFast')


    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)


    best_test_criterion = -1  # SROCC min
    best_test = []

    logger.info('Starting training:')
    awp = AWP(model, criterion, optimizer, adv_param="weight", gamma=config.gamma)
    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        train_labels, pred_labels = [], []
        for i, (video, feature_3D, mos, _) in enumerate(train_loader):
            

            video = video.to(device)
            feature_3D = feature_3D.to(device)
            labels = mos.to(device).float()
            
            outputs = model(video, feature_3D)
            optimizer.zero_grad()
            
            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()
            if config.awp:
                loss = awp.attack_backward(video, feature_3D, labels)
                logger.info(f"Epoch {epoch}, Iteration {i + 1}/{len(train_loader)}: AWP Loss: {loss.item():.4f}")
                loss.backward()
                awp._restore()
            optimizer.step()

            pred_labels.extend(list(outputs.detach().cpu().numpy()))
            train_labels.extend(list(labels.detach().cpu().numpy()))

            if (i+1) % (config.print_samples//config.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples//config.train_batch_size)
                logger.info('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                    (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, \
                        avg_loss_epoch))
                batch_losses_each_disp = []
                logger.info('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
        logger.info('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()
        logger.info('The current learning rate is {:.06f}'.format(lr[0]))

        train_srcc = spearmanr(train_labels, pred_labels)[0]
        train_plcc = pearsonr(train_labels, pred_labels)[0]
        train_krocc = kendallr(train_labels, pred_labels)[0]
        train_rmse = np.sqrt(((np.array(train_labels) - np.array(pred_labels)) ** 2).mean())
        logger.info(f"Epoch {epoch + 1}, Train: SROCC: {train_srcc:.4f}, PLCC: {train_plcc:.4f}, KROCC: {train_krocc:.4f}, RMSE: {train_rmse:.4f}")

        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            label = np.zeros([len(testset)])
            y_output = np.zeros([len(testset)])
            for i, (video, feature_3D, mos, _) in enumerate(test_loader):
                
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                label[i] = mos.item()
                outputs = model(video, feature_3D)

                y_output[i] = outputs.item()
            
            test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(label, y_output)
            
            logger.info('Epoch {} completed. The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                test_SRCC, test_KRCC, test_PLCC, test_RMSE))
                
            if test_SRCC > best_test_criterion:
                logger.info("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC
                best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                logger.info('Saving model...')
                
            save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                config.database + '_' + config.loss_type + '_NR_v'+ str(config.exp_version) \
                    + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC))
            torch.save(model.state_dict(), save_model_name)


    logger.info('Training completed.')
    logger.info('The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        best_test[0], best_test[1], best_test[2], best_test[3]))

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default = 0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default = 1000)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=520)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=10)
    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='L1RankLoss')
    parser.add_argument('--awp', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=0.0001)
    
    config = parser.parse_args()
    logger.info(config)
    main(config)