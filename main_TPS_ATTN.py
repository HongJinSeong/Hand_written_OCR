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

EPOCH = 60
BATCH_SIZE = 12

imgs = glob('datasets/train/', '*')
imgs.sort()
imgs = np.array(imgs)

labels = pd.read_csv('datasets/train.csv')

labels['len'] = labels['label'].str.len()
train_UNIQUE = labels[labels['len'] == 1]  ## 모든 문자열이 포함되어 있음으로 사실상 필수 데이터
train_UNIQUE['img_path'] = train_UNIQUE['img_path'].str.replace('./train', 'datasets/train/')

unique_img = train_UNIQUE['img_path'].values
unique_label = train_UNIQUE['label'].values

labels = labels['label'].values

lbl_str = ''

for st in labels:
    lbl_str += st

lbl_str = sorted(list(set(list(lbl_str))))


train_str=''
for st in lbl_str:
    train_str += st

# 10-fold 로 setting
# folder 기준으로 10fold 나누고 나눈 10-fold에서 폴더 별로 풀어서 train set / valid set 정의
kf = StratifiedKFold(n_splits=10, shuffle=True)

for kfold_idx, (index_kf_train, index_kf_validation) in enumerate(kf.split(labels, labels)):
    train_imgs = imgs[index_kf_train]
    train_lbls = labels[index_kf_train]

    valid_imgs = imgs[index_kf_validation]
    valid_labels = labels[index_kf_validation]

    train_ds = hand_written_ds(train_imgs.tolist()+unique_img.tolist(), train_lbls.tolist()+unique_label.tolist(), train_str, True)
    valid_ds = hand_written_ds(valid_imgs.tolist(), valid_labels.tolist(), train_str, False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=0)

    model = TPS_RES_ATTN()
    model = model.to(device)

    # criterion = torch.nn.CTCLoss().to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)

    for ep in range(EPOCH):
        model.train()
        train_loss = []
        train_CER = []
        torch.cuda.empty_cache()
        for ep_idx, (imgs, labels, label_len, label_str) in enumerate(tqdm(iter(train_loader))):

            imgs = imgs.to(device)
            labels = labels.to(device)
            label_len = label_len.to(device)

            optimizer.zero_grad()

            outputs = model(imgs, labels[:, :-1])
            labels = labels[:, 1:]
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.contiguous().view(-1))

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())


            if (ep_idx + 1) % 3500 == 0:
                print('train loss : ' + str(np.mean(np.array(train_loss))))

        model.eval()
        valid_loss = []
        valid_CER = []
        with torch.no_grad():
            for imgs, labels, label_len, label_str in tqdm(iter(valid_loader)):
                batch_size = imgs.size(0)
                imgs = imgs.to(device)
                labels = labels.to(device)
                label_len = label_len.to(device)

                ## max seqence length 72
                length_for_pred = torch.IntTensor([72] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, 72 + 1).fill_(0).to(device)

                outputs = model(imgs, text_for_pred, is_train=False)

                outputs = outputs[:, :labels.shape[1] - 1, :]
                labels = labels[:, 1:]  # without [GO] Symbol

                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.contiguous().view(-1))

                valid_loss.append(loss.item())

                _, outputs_index = outputs.max(2)
                outputs_str = valid_ds.decode(outputs_index, length_for_pred)
                labels = valid_ds.decode(labels, label_len)
                CER = tmetric.functional.char_error_rate(labels[0].split('[s]')[0], outputs_str[0].split('[s]')[0])

                valid_CER.append(CER)

                writecsv('outputs_TPS/' + str(kfold_idx) + 'fold_' + str(ep) + 'EP_predoutputs.csv',
                         [outputs_str[0], label_str[0]])

        torch.save(model.state_dict(), 'outputs_TPS/' + str(kfold_idx) + 'fold_' + str(ep) + 'epoch_classifier.pt')
        writecsv('outputs_TPS/CER_LOSS.csv',
                 [kfold_idx, ep, np.mean(np.array(train_loss)), np.mean(np.array(train_CER)),
                  np.mean(np.array(valid_loss)), np.mean(np.array(valid_CER))])
    break
