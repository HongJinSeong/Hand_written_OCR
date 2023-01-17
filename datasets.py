from utils import *
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import skimage.io as iio
from skimage.transform import resize
from PIL import Image


class AddGaussianNoise(object):
    def __init__(self, mean=0.):
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * torch.rand(1) + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0})'.format(self.mean)

class hand_written_ds_ATTN(Dataset):
    def __init__(self, input_paths, targets, train_str, train_TF):
        self.input_paths = input_paths
        self.targets = targets
        self.train_TF = train_TF

        self.tform = transforms.Compose([transforms.ToTensor()
                                       , transforms.Resize(size=(100, 300))
                                       , transforms.RandomRotation(degrees=(-10, 10), fill=1)  ## 0~1로  scailing 이후에 흰색으로 채워넣음 (글자가 검은색임)
                                       , transforms.RandomApply([transforms.GaussianBlur(3)])
                                       , transforms.RandomAdjustSharpness(2, p=0.5)
                                       , transforms.RandomApply([AddGaussianNoise()])])

        self.tform_TV = transforms.Compose([transforms.ToTensor()
                                       , transforms.Resize(size=(100, 300))])


        self.train_str = train_str

        self.cls = train_str

        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(train_str)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i



    def __len__(self):
            return len(self.input_paths)

    def encode(self, text, batch_max_length=72):
        """ convert text-label into text-index.
                input:
                    text: text labels of each image. [batch_size]
                    batch_max_length: max length of text label in the batch. 25 by default
                output:
                    text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                        text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
                    length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
                """
        length = len(text) + 1 # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(batch_max_length + 1).fill_(0)

        text = list(text)
        text.append('[s]')
        text = [self.dict[char] for char in text]
        batch_text[1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token

        return batch_text, length


    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

    def __getitem__(self, idx):
        img = iio.imread(self.input_paths[idx])

        if len(img.shape)==2:
            img = np.expand_dims(img, 2)
            img = np.concatenate((img, img, img), axis=2)

        img = Image.fromarray(img)

        if self.train_TF == True:
            img = self.tform(img)
        else:
            img = self.tform_TV(img)

        if self.targets!=None:
            label_str = self.targets[idx]

            lbl, length = self.encode(label_str)

            return img, lbl, length, label_str
        else:
            return img



### 텍스트 증강
class hand_written_ds(Dataset):
    def __init__(self, input_paths, targets, train_str, aug_ls, train_TF):
        self.input_paths = input_paths
        self.targets = targets
        self.train_TF = train_TF

        self.tform = transforms.Compose([transforms.ToTensor()
                                       , transforms.Resize(size=(100, 320))
                                       , transforms.RandomRotation(degrees=(-10, 10), fill=1)  ## 0~1로  scailing 이후에 흰색으로 채워넣음 (글자가 검은색임)
                                       , transforms.RandomApply([transforms.ColorJitter(brightness=(0.4, 2), contrast=(0.4, 2))])
                                       , transforms.RandomApply([transforms.GaussianBlur(5)])
                                       , transforms.RandomApply([AddGaussianNoise()])])

        self.tform_TV = transforms.Compose([transforms.ToTensor()
                                        , transforms.Resize(size=(100, 320))])

        self.train_str = train_str

        self.aug_ls = aug_ls

        self.cls_array = ['']
        for str in self.train_str:
            self.cls_array.append(str)
        self.cls_array = np.array(self.cls_array)

    def __len__(self):
        if self.train_TF == True:
            return len(self.input_paths) + 125000
        else:
            return len(self.input_paths)

    def decode(self, labels):
        output_ls=[]

        for label in labels:
            output_str_ls = self.cls_array[label]
            output_str_ls = output_str_ls.tolist()
            pred_str = "".join(output_str_ls)
            output_ls.append(pred_str)

        return output_ls

    def load_train_valid(self, idx):
        label_str = self.targets[idx]

        ### 문자길이 최대 100으로 두고 없는 문자는 0 class / 나머지 존재하는거는 self.cls index + 1
        lbl = np.zeros(shape=(38))

        for lbl_idx, char in enumerate(label_str):
            l_char = self.train_str.find(char)
            if l_char != -1:
                lbl[lbl_idx] = l_char + 1

        return lbl, label_str

    def load_image(self,img_path):
        real_img = (iio.imread(img_path, as_gray=True) * 255).astype(np.uint8)

        img = Image.fromarray(real_img)
        return img
    def load_mixed_image(self):
        k = random.randint(2, 6)
        N = random.randint(1, k-1)
        lbl_str = ''

        tr = random.choice(self.aug_ls[N-1])
        lbl_str += tr[-2]
        tarpath = tr[1].replace('./train/', 'datasets/train_prepro_rotate/')
        imgset = iio.imread(tarpath, as_gray=True)
        imgset = imgset.astype(np.uint8)

        while len(lbl_str) != k:
            N = random.randint(1, k - len(lbl_str))
            tr = random.choice(self.aug_ls[N - 1])
            lbl_str += tr[-2]
            tarpath = tr[1].replace('./train/', 'datasets/train_prepro_rotate/')
            catimg = iio.imread(tarpath, as_gray=True)
            catimg = catimg.astype(np.uint8)
            imgset = np.concatenate((imgset, catimg), axis=1)

        img = Image.fromarray(imgset)

        ### 문자길이 최대 100으로 두고 없는 문자는 0 class / 나머지 존재하는거는 self.cls index + 1
        lbl = np.zeros(shape=(38))

        for lbl_idx, char in enumerate(lbl_str):
            l_char = self.train_str.find(char)
            if l_char != -1:
                lbl[lbl_idx] = l_char + 1

        return self.tform(img), lbl, lbl_str



    def __getitem__(self, idx):
        if idx < len(self.input_paths):
            idx = idx % len(self.input_paths)

            if self.targets != None:
                img_path1 = self.input_paths[idx].replace('\\', '/').replace('datasets/train/', 'datasets/train_prepro_rotate/')
            else:
                img_path1 = self.input_paths[idx]

            img1 = self.load_image(img_path1)
        else:
            img, lbl, lbl_str = self.load_mixed_image()
            return img, lbl, lbl_str


        if self.targets!=None:
            if self.train_TF ==True:
                img = self.tform(img1)
            else:
                img = self.tform_TV(img1)

            lbl, lbl_str = self.load_train_valid(idx)
            return img, lbl, lbl_str
        else:
            img = self.tform_TV(img1)
            return img
