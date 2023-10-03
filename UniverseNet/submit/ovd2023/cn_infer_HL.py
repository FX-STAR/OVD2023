import json
import os
import sys
from toolz.curried import get, groupby
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import mmcv
import torch 
from PIL import Image
import cv2

import cn_clip.clip as clip
from cn_clip.eval.data import _preprocess_text
from cn_clip.clip import load_from_name, available_models

curr_dir = os.path.dirname(__file__)

import sys
sys.path.insert(0,f'{curr_dir}')
from cn_templates import templates

classnames=json.load(open(f'{curr_dir}/classes_cn.json'))


def zero_shot_classifier(model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [_preprocess_text(template(classname)) for template in templates]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = model(None, texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

# print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model_H, preprocess_H = load_from_name("ViT-H-14", device=device, download_root=os.path.join(curr_dir))
model_H.eval()

model_L, preprocess_L = load_from_name("ViT-L-14-336", device=device, download_root=os.path.join(curr_dir))
model_L.eval()


ann_json = '../Data/json_final_contest/test_fg.json'
img_dir='../Data/data_final_contest/test'

pid=int(sys.argv[1])
num_p=int(sys.argv[2])

cocoGt = COCO(ann_json)
cocoDt = cocoGt.loadRes(f'{curr_dir}/results/wbf.json')
imgs = list(cocoDt.imgs.keys())

num=len(imgs)//num_p+1
imgs=imgs[pid*num:(pid+1)*num]
res = []

weights=None
classifier_H = zero_shot_classifier(model_H, classnames, templates, device)
classifier_L = zero_shot_classifier(model_L, classnames, templates, device)
# classifier_HH = zero_shot_classifier(model_HH, classnames, templates, device)

def tta(img, bbox, fs = [0.1, 0.2]):
    x1,y1,w,h = bbox
    x1,y1,x2,y2=int(x1),int(y1),int(x1+w),int(y1+h)
    im_h, im_w = img.shape[:2]
    # imgs = [img, cv2.flip(img, 1)]
    imgs = [img]
    if isinstance(fs, int):
        exp_w, exp_h = fs, fs
        lx = x1-exp_w if x1-exp_w>0 else 0
        ty = y1-exp_h if y1-exp_h>0 else 0
        rx = x2+exp_w if x2+exp_w<im_w else im_w-1
        by = y2+exp_h if y2+exp_h<im_h else im_h-1
        img_exp = imgs[0][ty:by,lx:rx,:]
        # img_f_exp = imgs[1][ty:by,lx:rx,:]
        # imgs+=[img_exp, img_f_exp]
        imgs+=[img_exp]
    else:
        for f in fs:
            exp_w, exp_h = int(w*f), int(h*f)
            lx = x1-exp_w if x1-exp_w>0 else 0
            ty = y1-exp_h if y1-exp_h>0 else 0
            rx = x2+exp_w if x2+exp_w<im_w else im_w-1
            by = y2+exp_h if y2+exp_h<im_h else im_h-1
            img_exp = imgs[0][ty:by,lx:rx,:]
            # img_f_exp = imgs[1][ty:by,lx:rx,:]
            # imgs+=[img_exp, img_f_exp]
            imgs+=[img_exp]
    # imgs = [imgs[2]]
    ret_imgs = []
    for img in imgs:
        try:
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            ret_imgs.append(img)
        except:
            continue
    return ret_imgs



for img_id in tqdm(imgs):
    ann_ids = cocoDt.getAnnIds(imgIds = [img_id])
    anns = [cocoDt.anns[_] for _ in ann_ids]
    img_name = cocoDt.imgs[img_id]['file_name']
    img = cv2.imread(f'{img_dir}/{img_name}')
    im_h, im_w = img.shape[:2]
    for ann in anns:
        bbox=ann['bbox']
        if ann['score']<0.005: continue
        try:
            # tta_imgs = tta(img, bbox, fs=[0.1, 0.2])
            # tta_imgs = tta(img, bbox, fs=30)
            tta_imgs = tta(img, bbox, fs=[0,30])
        except:
            continue

        with torch.no_grad(): 
            try:
                tta_imgs_H = [preprocess_H(xx).to(device) for xx in tta_imgs]
                tta_imgs_L = [preprocess_L(xx).to(device) for xx in tta_imgs]
            except:
                continue
            
            tta_imgs_H = torch.stack(tta_imgs_H)
            image_features = model_H(tta_imgs_H, None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits_H = (image_features @ classifier_H)

            tta_imgs_L = torch.stack(tta_imgs_L)
            image_features = model_L(tta_imgs_L, None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits_L = (image_features @ classifier_L)


            logits = logits_H*0.7+logits_L*0.3

            probs = logits.mean(dim=0).softmax(dim=-1).cpu().numpy()
            # print(logits.shape, tta_imgs.shape, image_features.shape, classifier.shape)
            label = probs.argmax()
            score = probs[label]

        res.append({
            'image_id': img_id,
            'bbox': ann['bbox'],
            'category_id': int(label+1),
            'score': float(score*ann['score'])
        })
with open(f'{curr_dir}/results/cn_clip_{pid}.json', 'w') as f:
    json.dump(res, f)
