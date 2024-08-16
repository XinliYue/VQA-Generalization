import torch
import fastvqa.models as models
import fastvqa.datasets as datasets

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

import math
import timeit
import yaml

from functools import reduce

class WP:
    def __init__(self, model, criterion, optimizer, adv_param="weight", gamma=0.0001, awp=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.gamma = gamma
        self.backup = {}
        self.backup_eps = {}
        self.awp = awp

    def attack_backward(self, inputs, labels):
        if self.gamma == 0:
            return
        self._save()
        self._attack_step()

        y_preds = self.model(inputs, inference=False,
                                reduce_scores=False)[0].mean((-3, -2, -1))
        adv_loss = self.criterion(y_preds, labels)
        self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if not self.awp:
                    noise = torch.randn_like(param.data)
                    norm1 = torch.norm(noise)
                    norm2 = torch.norm(param.data.detach())
                    if norm1 != 0 and not torch.isnan(norm1):
                        r_at = self.gamma * noise / (norm1 + e) * (norm2 + e)
                        param.data.add_(r_at)
                        param.data = torch.min(
                            torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                        )
                else:
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

def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def loss_fn(y_pred, y):
    p_loss = plcc_loss(y_pred, y)
    r_loss = rank_loss(y_pred, y)
    return p_loss + 0.3 * r_loss

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["fragments"]

def finetune_epoch(ft_loader, model, model_ema, optimizer, scheduler, device, epoch=-1, wp=None):
    model.train()
    tic = timeit.default_timer()
    train_labels, pred_labels = [], []
    for i, data in enumerate(ft_loader):
        optimizer.zero_grad()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
        
        y = data["gt_label"].float().detach().to(device).unsqueeze(-1)

        scores = model(video, inference=False,
                                reduce_scores=False) 
        if len(scores) > 1:
            y_pred = reduce(lambda x,y:x+y, scores)
        else:
            y_pred = scores[0]
        y_pred = y_pred.mean((-3, -2, -1))
        
        loss = loss_fn(y_pred, y)
        logger.info(f"Epoch {epoch}, Iteration {i + 1}/{len(ft_loader)}: Clean Loss: {loss.item():.4f}")
        loss.backward()
        if wp is not None:
            loss = wp.attack_backward(video, y)
            logger.info(f"Epoch {epoch}, Iteration {i + 1}/{len(ft_loader)}: WP Loss: {loss.item():.4f}")
            loss.backward()
            wp._restore()
        optimizer.step()
        scheduler.step()
        
        pred_labels.extend(list(y_pred.view(-1).detach().cpu().numpy()))
        train_labels.extend(list(y.view(-1).detach().cpu().numpy()))
        
        if model_ema is not None:
            model_params = dict(model.named_parameters())
            model_ema_params = dict(model_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999
                )

    pred_labels = rescale(pred_labels, train_labels)
    train_srcc = spearmanr(train_labels, pred_labels)[0]
    train_plcc = pearsonr(train_labels, pred_labels)[0]
    train_krocc = kendallr(train_labels, pred_labels)[0]
    train_rmse = np.sqrt(((train_labels - pred_labels) ** 2).mean())

    toc = timeit.default_timer()

    minutes = int((toc - tic) / 60)
    seconds = int((toc - tic) % 60)
    logger.info('Epoch-{:02d}, time elapsed {:02d}m {:02d}s.'.format(epoch, minutes, seconds))
    logger.info('backbone_lr = {:.2e}, head_lr = {:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'],
                                                          optimizer.state_dict()['param_groups'][-1]['lr']))
    logger.info(f"Epoch {epoch}, Train: SROCC: {train_srcc:.4f}, PLCC: {train_plcc:.4f}, KROCC: {train_krocc:.4f}, RMSE: {train_rmse:.4f}")
    model.eval()

def inference_set(inf_loader, model, device, best_, save_model=False, suffix='s', save_name="divide", epoch=-1):

    results = []
    tic = timeit.default_timer()
    best_s, best_p, best_k, best_r = best_
 
    for i, data in enumerate(inf_loader):
        result = dict()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
            
        with torch.no_grad():
            result["pr_labels"] = model(video).cpu().numpy()
                
        result["gt_label"] = data["gt_label"].item()
        del video
        results.append(result)
        logger.info(f"Processed {i+1}/{len(inf_loader)} videos.")

    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)
    
    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

        
    del results, result
    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "validation_results": best_,
            },
            f"pretrained_weights/train/{save_name}/{suffix}_best.pth",
        )
    if save_model:
        state_dict = model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "validation_results": best_,
            },
            f"pretrained_weights/train/{save_name}/{suffix}_{epoch}.pth",
        )

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )


    toc = timeit.default_timer()
    minutes = int((toc - tic) / 60)
    seconds = int((toc - tic) % 60)

    logger.info(
        f"For {len(gt_labels)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )
    logger.info('time elapsed {:02d}m {:02d}s.'.format(minutes, seconds))

    return best_s, best_p, best_k, best_r

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

def main():
    seed = 42
    logger.info(f'seed: {seed}')
    set_seed(seed)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/divide/mradd.yml", help="the option file"
    )
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('--aua', action='store_true', default=False)
    parser.add_argument('--ra', action='store_true', default=False)
    parser.add_argument('--ta', action='store_true', default=False)
    parser.add_argument('--rwp', action='store_true', default=False)
    parser.add_argument('--awp', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=0.0001)
    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    opt["optimizer"]["wd"] = args.wd
    for key in opt["data"]:
        if key == 'train':
            if args.aua:
                opt["data"][key]["args"]["aua"] = True
            if args.ra:
                opt["data"][key]["args"]["ra"] = True
            if args.ta:
                opt["data"][key]["args"]["ta"] = True
    logger.info(opt)
    save_dir = f'pretrained_weights/train/{opt["name"]}'
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    
    if opt.get("split_seed", -1) > 0:
        num_splits = 10
    else:
        num_splits = 1
    
    for split in range(num_splits):
        
        val_datasets = {}
        for key in opt["data"]:
            if key.startswith("val"):
                val_datasets[key] = getattr(datasets, 
                                            opt["data"][key]["type"])(opt["data"][key]["args"])


        val_loaders = {}
        for key, val_dataset in val_datasets.items():
            val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
            )

        train_datasets = {}
        for key in opt["data"]:
            if key.startswith("train"):
                train_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])
                train_datasets[key] = train_dataset
        
        train_loaders = {}
        for key, train_dataset in train_datasets.items():
            train_loaders[key] = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=True,
            )
        
        
        if "load_path" in opt:
            state_dict = torch.load(opt["load_path"], map_location=device)

            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                from collections import OrderedDict

                i_state_dict = OrderedDict()
                for key in state_dict.keys():
                    if "head" in key:
                        continue
                    if "cls" in key:
                        tkey = key.replace("cls", "vqa")
                    elif "backbone" in key:
                        i_state_dict[key] = state_dict[key]
                        i_state_dict["fragments_"+key] = state_dict[key]
                        i_state_dict["resize_"+key] = state_dict[key]
                    else:
                        i_state_dict[key] = state_dict[key]
            t_state_dict = model.state_dict()
            for key, value in t_state_dict.items():
                if key in i_state_dict and i_state_dict[key].shape != value.shape:
                    i_state_dict.pop(key)
            
            logger.info(model.load_state_dict(i_state_dict, strict=False))

        if opt["ema"]:
            from copy import deepcopy
            model_ema = deepcopy(model)
        else:
            model_ema = None

        param_groups=[]

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"] * opt["optimizer"]["backbone_lr_mult"]}]
            else:
                param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"]}]

        optimizer = torch.optim.AdamW(lr=opt["optimizer"]["lr"], params=param_groups,
                                      weight_decay=opt["optimizer"]["wd"],
                                     )
        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda,
        )

        bests = {}
        bests_n = {}
        for key in val_loaders:
            bests[key] = -1,-1,-1,1000
            bests_n[key] = -1,-1,-1,1000
        wp = WP(model, loss_fn, optimizer, gamma=args.gamma, awp=args.awp) if args.awp or args.rwp else None
        for epoch in range(opt["num_epochs"]):
            logger.info(f"Finetune Epoch {epoch}:")

            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader, model, model_ema, optimizer, scheduler, device, epoch, wp
                )
            for key in val_loaders:
                bests[key] = inference_set(
                    val_loaders[key],
                    model_ema if model_ema is not None else model,
                    device, bests[key], save_model=opt["save_model"], save_name=opt["name"],
                    suffix=key+"_s",
                    epoch=epoch,
                )
                if model_ema is not None:
                    bests_n[key] = inference_set(
                        val_loaders[key],
                        model,
                        device, bests_n[key], save_model=opt["save_model"], save_name=opt["name"],
                        suffix = key+'_n',
                        epoch=epoch,
                    )
                else:
                    bests_n[key] = bests[key]
                    
        if opt["num_epochs"] > 0:
            for key in val_loaders:
                logger.info(
                    f"""For the finetuning process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-s is as follows:
                    SROCC: {bests[key][0]:.4f}
                    PLCC:  {bests[key][1]:.4f}
                    KROCC: {bests[key][2]:.4f}
                    RMSE:  {bests[key][3]:.4f}."""
                )

                logger.info(
                    f"""For the finetuning process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-n is as follows:
                    SROCC: {bests_n[key][0]:.4f}
                    PLCC:  {bests_n[key][1]:.4f}
                    KROCC: {bests_n[key][2]:.4f}
                    RMSE:  {bests_n[key][3]:.4f}."""
                )
            
if __name__ == "__main__":
    main()
