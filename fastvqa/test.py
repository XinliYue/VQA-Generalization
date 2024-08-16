import torch
import sys
import random
import fastvqa.models as models
import fastvqa.datasets as datasets

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from tqdm import tqdm

import yaml


def rescale(pr, gt=None):
    if gt is None:
        logger.info("mean", np.mean(pr), "std", np.std(pr))
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        logger.info(np.mean(pr), np.std(pr), np.std(gt), np.mean(gt))
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["fragments"]

def inference_set(inf_loader, model, device, best_, save_model=False, suffix='s', set_name="na"):
    logger.info(f"Validating for {set_name}.")
    results = []

    best_s, best_p, best_k, best_r = best_
    
    keys = []

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video = {}
        for key in sample_types:
            if key not in keys:
                keys.append(key)
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
        with torch.no_grad():
            labels = model(video,reduce_scores=False)
            labels = [np.mean(l.cpu().numpy()) for l in labels]
            result["pr_labels"] = labels
        result["gt_label"] = data["gt_label"].item()
        result["name"] = data["name"]
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)
        if i % 100 == 0:
            logger.info(f"Processed {i+1}/{len(inf_loader)} videos.")

    
    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = 0
    pr_dict = {}
    for i, key in zip(range(len(results[0]["pr_labels"])), keys):
        key_pr_labels = np.array([np.mean(r["pr_labels"][i]) for r in results])
        pr_dict[key] = key_pr_labels
        pr_labels += rescale(key_pr_labels)
        
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
    
    
    results = sorted(results, key=lambda x: x["pr_labels"])

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    logger.info(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )

    return best_s, best_p, best_k, best_r, pr_labels

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
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/fast/fast-b.yml", help="the option file"
    )
    parser.add_argument("--model_dir", type=str, default="")
    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    logger.info(opt)
    
    ## adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    ## defining model and loading checkpoint
    
    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    
    for key in opt["data"].keys():
        for epoch in range(30):
            model_path = args.model_dir + "/val-kv1k_s_{}.pth".format(epoch)
            logger.info(model_path)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict['state_dict'] , strict=False)
            if "val" not in key and "test" not in key:
                continue
            
            val_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])

            val_loader =  torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
            )

            # test the model
            logger.info(len(val_loader))

            best_ = -1, -1, -1, 1000

            best_ = inference_set(
                val_loader,
                model,
                device, best_,
                set_name=key,
            )

            logger.info(
                f"""Testing result on: [{len(val_loader)}] videos:
                SROCC: {best_[0]:.4f}
                PLCC:  {best_[1]:.4f}
                KROCC: {best_[2]:.4f}
                RMSE:  {best_[3]:.4f}."""
            )

if __name__ == "__main__":
    main()
