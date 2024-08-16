import torch
import fastvqa.models as models
import fastvqa.datasets as datasets
import argparse
import sys
import random
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

import timeit

import yaml

import warnings

warnings.filterwarnings("ignore")

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["fragments"]

def inference_set(inf_loader, model, device, best_, suffix='s', epoch=-1):

    results = []

    tic = timeit.default_timer()
    gt_labels, pr_labels = [], []
    best_s, best_p, best_k, best_r = best_
    for i, data in enumerate(inf_loader):
        result = dict()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 

        with torch.no_grad():
            result["pr_labels"] = model(video).cpu().numpy()
                
        result["gt_label"] = data["gt_label"].item()

        results.append(result)
        logger.info(f"Processed {i+1}/{len(inf_loader)} videos.")

    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)
    
    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
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
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}_{epoch}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )
    logger.info('time elapsed {:02d}m {:02d}s.'.format(minutes, seconds))
    del results, result
    torch.cuda.empty_cache()
    return best_s, best_p, best_k, best_r

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
    logger.info(f'seed:{seed}')
    set_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, 
                        default="./options/fast-SAMA-test.yml", help="the option file")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()
    
    logger.info(args)
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    logger.info(opt)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if sys.gettrace():
        logger.info('in DEBUGE mode.')
        opt["name"] = "DEBUG"
        opt['test_num_workers']=0

    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)

    stype = opt['stype'] if opt['stype'] in ['sama', 'sama-c', 'sama-mix', 'sama+spm', 'sama+swm'] else 'fragments'
        
    val_datasets = {}
    for key in opt["data"]:
        if key.startswith("val"):
            val_datasets[key] = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"], stype=stype)
            logger.info('dataset=[{}], with {} samples.'.format(key, len(val_datasets[key])))

    val_loaders = {}
    for key, val_dataset in val_datasets.items():
        val_loaders[key] = torch.utils.data.DataLoader(val_dataset, 
                                                        batch_size=opt["test_batch_size"], 
                                                        num_workers=opt["test_num_workers"], 
                                                        pin_memory=False,
                                                        shuffle=False,
                                                        drop_last=False)
    bests = {}
    for key in val_loaders:
            bests[key] = -1,-1,-1,1000
    for epoch in range(30):
        model_path = args.model_dir + "/val-kv1k_s_{}.pth".format(epoch)
        logger.info(model_path)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict['state_dict'] , strict=False)
                
        logger.info(f"evaluation epoch {epoch}...")

        for key in val_loaders:
            bests[key] = inference_set(
                val_loaders[key],
                model,
                device,
                bests[key],
                suffix=key,
                epoch=epoch,
            )                  
    for key in val_loaders:
        logger.info(
            f"""For the finetuning process on {key} with {len(val_datasets[key])} videos,
            the best validation accuracy of the model-s is as follows:
            SROCC: {bests[key][0]:.4f}
            PLCC:  {bests[key][1]:.4f}
            KROCC: {bests[key][2]:.4f}
            RMSE:  {bests[key][3]:.4f}."""
        )

if __name__ == "__main__":
    main()
