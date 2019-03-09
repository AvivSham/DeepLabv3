import torch
import torch.nn as nn
from utils import *
from models.deeplabv3 import DeepLabv3
import sys
import os
import time
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def predict(FLAGS):
    # Check if the pretrained model is available
    if not FLAGS.m.endswith('.pth'):
        raise RuntimeError('Unknown file passed. Must end with .pth')
    if FLAGS.image_path is None or not os.path.exists(FLAGS.image_path):
        raise RuntimeError('An image file path must be passed')
    
    h = FLAGS.resize_height
    w = FLAGS.resize_width
    
    print ('[INFO]Loading Checkpoint...')
    checkpoint = torch.load(FLAGS.m,  map_location='cpu')
    print ('[INFO]Checkpoint Loaded')
    
    # Assuming the dataset is camvid
    deeplabv3 = DeepLabv3(FLAGS.num_classes)
    deeplabv3.load_state_dict(checkpoint['model_state_dict'])
    print ('[INFO]Initiated model with pretraiend weights.')

    tmg_ = np.array(Image.open(FLAGS.image_path))
    tmg_ = cv2.resize(tmg_, (w, h), cv2.INTER_NEAREST)
    tmg = torch.tensor(tmg_).unsqueeze(0).float()
    tmg = tmg.transpose(2, 3).transpose(1, 2)
    
    print ('[INFO]Starting inference...')
    deeplabv3.eval()
    s = time.time()
    out1 = deeplabv3(tmg.float()).squeeze(0)
    o = time.time()
    deeplabv3.train()
    print ('[INFO]Inference complete!')
    print ('[INFO]Time taken: ', o - s)
    
    out2 = out1.squeeze(0).cpu().detach().numpy()

    b_ = out1.data.max(0)[1].cpu().detach().numpy()
    
    b = decode_segmap_cscapes(b_)
    print ('[INFO]Got segmented results!')

    plt.title('Input Image')
    plt.axis('off')
    plt.imshow(tmg_)
    plt.show()

    plt.title('Output Image')
    plt.axis('off')
    plt.imshow(b)
    plt.show()

    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(9, 4)
    gs.update(wspace=0.025, hspace=0.005)

    label = 0
    for ii in range(34):
        plt.subplot(gs[ii])
        plt.axis('off')
        plt.imshow(out2[label, :, :])
        label += 1
    plt.show()
    
    if FLAGS.save:
        cv2.imwrite('seg.png', b)
        print ('[INFO]Segmented image saved successfully!')

    print ('[INFO] Prediction complete successfully!')
