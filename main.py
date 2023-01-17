# 128에서 멈춤
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 시드(seed) 설정
RANDOM_SEED = 199002
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

EPOCH = 200
BATCH_SIZE = 128

imgs = glob('datasets/train/', '*')
imgs.sort()
imgs = np.array(imgs)


tds = pd.read_csv('datasets/train.csv')

tds['len'] = tds['label'].str.len()
train_UNIQUE = tds[tds['len'] == 1]  ## 모든 문자열이 포함되어 있음으로 사실상 필수 데이터

aug_set=[]
for llen in range(1,6):
    aug_set.append(tds[tds['len'] == llen].values.tolist())

unique_img = train_UNIQUE['img_path'].values
unique_label = train_UNIQUE['label'].values

labels = tds['label'].values

lbl_str = ''

for st in labels:
    lbl_str += st

lbl_str = sorted(list(set(list(lbl_str))))


train_str=''
for st in lbl_str:
    train_str += st

# 10-fold 로 setting
# folder 기준으로 10fold 나누고 나눈 10-fold 에서 폴더 별로 풀어서 train set / valid set 정의
kf = StratifiedKFold(n_splits=5, shuffle=True)

for kfold_idx, (index_kf_train, index_kf_validation) in enumerate(kf.split(labels, tds['len'])):
    train_imgs = imgs[index_kf_train]
    train_lbls = labels[index_kf_train]

    valid_imgs = imgs[index_kf_validation]
    valid_labels = labels[index_kf_validation]

    train_ds = hand_written_ds(train_imgs.tolist() , train_lbls.tolist(), train_str, aug_set, True)
    valid_ds = hand_written_ds(valid_imgs.tolist(), valid_labels.tolist(), train_str, None, False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=0)

    model = RCNN_OCR()
    model = model.to(device)

    # criterion = torch.nn.CTCLoss().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)

    for ep in range(EPOCH):
        model.train()
        train_loss = []
        train_CER = []
        torch.cuda.empty_cache()
        for ep_idx, (imgs, labels, label_str) in enumerate(tqdm(iter(train_loader))):

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.contiguous().view(-1).long())

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            _, outputs_index = outputs.max(2)
            preds_str = train_ds.decode(outputs_index.detach().cpu().numpy())


            CER = tmetric.functional.char_error_rate(preds_str, label_str)

            train_CER.append(CER)

            if (ep_idx + 1) % 700 == 0:
                print('train loss : ' + str(np.mean(np.array(train_loss))))
                print('train CER : ' + str(np.mean(np.array(train_CER))))
                print(preds_str[:5])
                print(label_str[:5])

        model.eval()
        valid_loss = []
        valid_CER = []
        with torch.no_grad():
            for imgs, labels, label_str in tqdm(iter(valid_loader)):
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)

                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.contiguous().view(-1).long())

                valid_loss.append(loss.item())

                _, outputs_index = outputs.max(2)


                outputs_str = valid_ds.decode(outputs_index.detach().cpu().numpy())


                CER = tmetric.functional.char_error_rate(outputs_str, label_str)

                valid_CER.append(CER)

                writecsv('outputs_CUSTOM/' + str(kfold_idx) + 'fold_' + str(ep) + 'EP_predoutputs.csv',
                         [outputs_str[0], label_str[0]])

        torch.save(model.state_dict(), 'outputs_CUSTOM/' + str(kfold_idx) + 'fold_' + str(ep) + 'epoch_classifier.pt')
        writecsv('outputs_CUSTOM/CER_LOSS.csv',
                 [kfold_idx, ep, np.mean(np.array(train_loss)), np.mean(np.array(train_CER)),
                  np.mean(np.array(valid_loss)), np.mean(np.array(valid_CER))])
    break
