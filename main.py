import argparse
from model.TransMIL import TransMIL
import torch
import os
from torch.utils.data import DataLoader
from dataset import *
import cv2
import numpy as np
parser = argparse.ArgumentParser(description='Create heatmaps')
#use clam create_feature_fp.py you can get .h5 file
parser.add_argument('--h5_path', type=str, default='h5-files/')
#thumbnail img dir
parser.add_argument('--thumbnail_path', type=str, default='images/')
#mean min max
parser.add_argument('--head_fusion', type=str, default= 'mean')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--model_name', type=str, default='TransMIL')
parser.add_argument('--device', type=str, default='cuda:0')
#Determine the maximum size of the wsi using qupath, divide this size by the thumbnail size
#int
parser.add_argument('--downsample', type=int, default=64)
#patch-size
parser.add_argument('--patch_size', type=int, default=512)
args = parser.parse_args()

def load_model(model_name, mode_path):
    if model_name == 'TransMIL':
        param = torch.load(mode_path)['state_dict']
        new_param = {k[6:]: v for k, v in param.items()}
        model = TransMIL(n_classes=2, head_fusion=args.head_fusion)
        model.load_state_dict(new_param)
    return model
def main(args):
    model = load_model(args.model_name, args.model_path)
    attn_dataset = Attn_Dateset(args.h5_path, args.thumbnail_path)
    attn_dataloader = DataLoader(attn_dataset, batch_size=1, shuffle=False)
    model.to(device=args.device)
    model.eval()
    for batch in attn_dataloader:
        coords, feature, img_path = batch
        img_path = img_path[0]
        feature = feature.to(args.device)
        with torch.no_grad():
            results_dict, attns = model(feature)
        result = torch.ones(attns[0].shape).to(attns[0].device)
        for i, attn in enumerate(attns):
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            attn = attn / (attn.max())
            result = ((attn * result) + result) / 2
        attns = result[0, 1:].to('cpu')
        if int(results_dict['Y_hat']) == 1:
            epsilon = 1e-10
            attns = attns + epsilon
            attns = attns.exp()
            min_val = attns.min()
            max_val = attns.max()
            attns = (attns - min_val) / (max_val - min_val)
        else:
            # attns = attns.max()
            attns = attns * 0.1
        downsample = args.downsample
        downsample_patchsize = int(args.patch_size//downsample)
        img = cv2.imread(img_path)
    
        mask = np.zeros((int(img.shape[0]),int(img.shape[1])))
        mask1 = np.ones((int(downsample_patchsize),int(downsample_patchsize)))
        coords = coords.numpy()[0]
        for i in range(coords.shape[0]):
            x = int(coords[i][1]//downsample)
            y = int(coords[i][0]//downsample)
            if x+downsample_patchsize < mask.shape[0] and y+downsample_patchsize < mask.shape[1]:
                mask[x:x+downsample_patchsize,y:y+downsample_patchsize] = attns[i]*mask1
        print(mask.max())
        img = np.float32(img)/255
        mask = cv2.resize(mask,(img.shape[1],img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255*mask),cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        savepath1 = "out/out1/" + img_path[7:]
        savepath2 = "out/out2/" + img_path[7:]
        cv2.imwrite(savepath1, np.uint8(255 * cam))
        cv2.imwrite(savepath2, heatmap*255)
        print("finish")
if __name__ == '__main__':
    main(args)
    pass
