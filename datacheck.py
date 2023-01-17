from utils import *

import json
import pickle

jsonpath = 'outer_ds/hand_written_data/handwriting_data_info_clean.json'


with open(jsonpath, "r", encoding='UTF8') as st_json:

    st_python = json.load(st_json)

imgs_path = glob('outer_ds/hand_written_data/01_handwriting_sentence_images/1_sentence', '*')
imgs_path.sort()

new_ls =[]
end_flag = False
for anno in st_python['annotations']:
    id = anno['id']

    for img_idx, pth in enumerate(imgs_path):
        pth = pth.replace('\\', '/')
        img_id = pth.split('/')[-1][:-4]
        if id == img_id:
            new_ls.append(anno)
            if img_idx == len(imgs_path)-1:
                end_flag = True
            break

    if end_flag == True:
        break

with open('outer_ds/labels.pickle', 'wb') as f:
    pickle.dump(new_ls, f, pickle.HIGHEST_PROTOCOL)

print('aaa')
