import argparse
import os

import numpy as np

import torch
import torch.nn
from torchvision import transforms

from model import UGC_BVQA_model

from utils import performance_fit

from data_loader import VideoDataset_images_with_motion_features

import sys
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.database == 'LSVQ_test':
        datainfo_test = 'data/LSVQ_whole_test.csv'
        videos_dir = os.path.join(config.data_path, 'LSVQ_image')
        feature_dir = os.path.join(config.data_path, 'LSVQ_SlowFast_feature/')
    elif config.database == 'KoNViD-1k':
        datainfo_test = './data/konvid_labels.txt'
        videos_dir = os.path.join(config.data_path, 'konvid1k_image')
        feature_dir = os.path.join(config.data_path, 'konvid1k_SlowFast_feature/')
    elif config.database == 'LIVE_VQC':
        datainfo_test = './data/livevqc_labels.txt'
        videos_dir = os.path.join(config.data_path, 'livevqc_image')
        feature_dir = os.path.join(config.data_path, 'livevqc_SlowFast_feature/')
    elif config.database == 'YouTubeUGC':
        datainfo_test = './data/youtubeugc_labels.txt'
        videos_dir = os.path.join(config.data_path, 'youtubeugc_image')
        feature_dir = os.path.join(config.data_path, 'youtubeugc_SlowFast_feature/')

    transformations_test = transforms.Compose([transforms.Resize(520),transforms.CenterCrop(448),\
        transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
  
    testset = VideoDataset_images_with_motion_features(videos_dir, feature_dir, datainfo_test, \
        transformations_test, config.database, 448, config.feature_type)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    for root, dirs, files in os.walk(config.model_dir):
        for file in files:
            config.trained_model = os.path.join(root, file)
            logger.info(config.trained_model)

            if config.model_name == 'UGC_BVQA_model':
                logger.info(f'The current model is {config.model_name}')
                model = UGC_BVQA_model.resnet50(pretrained=False)

            # model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)

            # load the trained model
            logger.info('loading the trained model')
            model.load_state_dict(torch.load(config.trained_model))
            with torch.no_grad():
                model.eval()
                label = np.zeros([len(testset)])
                y_output = np.zeros([len(testset)])
                videos_name = []
                for i, (video, feature_3D, mos, video_name) in enumerate(test_loader):
                    videos_name.append(video_name)
                    video = video.to(device)
                    feature_3D = feature_3D.to(device)
                    label[i] = mos.item()
                    outputs = model(video, feature_3D)

                    y_output[i] = outputs.item()
                    if i % 100 == 0:
                        logger.info(f"Processed {i+1}/{len(test_loader)} videos.")
                val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(label, y_output)
                
                logger.info('The result on the databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(\
                    val_SRCC, val_KRCC, val_PLCC, val_RMSE))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='KoNViD-1k')
    parser.add_argument('--train_database', type=str, default='LSVQ')
    parser.add_argument('--model_name', type=str, default='UGC_BVQA_model')

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str, default='ckpts')
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--feature_type', type=str, default='SlowFast')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--model_dir', type=str)
    
    config = parser.parse_args()

    main(config)



