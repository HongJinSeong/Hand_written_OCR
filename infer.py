import re
from utils import *
from datasets import *
from models import *

import torch
from torch.utils.data import DataLoader

import random
import json
### 윈도우 버전
import pickle
### 리눅스 버전
# import pickle5 as pickle
import torchmetrics as tmetric
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict

# 시드(seed) 설정
RANDOM_SEED = 199002
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imgs = glob('datasets/test_prepro_sharp/', '*')
imgs.sort()
labels = pd.read_csv('datasets/train.csv')
labels = labels['label'].values

lbl_str = ''

for st in labels:
    lbl_str += st

lbl_str = sorted(list(set(list(lbl_str))))


train_str=''
for st in lbl_str:
    train_str += st

test_ds = hand_written_ds(imgs, None, train_str, None, False)



test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

model = RCNN_OCR()
model.load_state_dict(torch.load('outputs_CUSTOM/0fold_141epoch_classifier.pt'))
model = model.to(device)

csv_sol = pd.read_csv('datasets/sample_submission.csv')


for idx, (imgs) in enumerate(tqdm(iter(test_loader))):
    batch_size = imgs.size(0)
    imgs = imgs.to(device)

    outputs = model(imgs)
    
    prob = torch.nn.functional.softmax(outputs,2)
    

    maxval, outputs_index = prob.max(2)
    

    outputs_str = test_ds.decode(outputs_index.detach().cpu().numpy())
    
    if len(outputs_str[0]) >= 4:
        outputs_str = ''.join(OrderedDict.fromkeys(outputs_str[0]))
    else:
        outputs_str = outputs_str[0]
    print(outputs_str)
    
    csv_sol['label'][idx] = outputs_str
    
csv_sol.to_csv('CUSTOM_DD.csv', index=False)

print('ggg')