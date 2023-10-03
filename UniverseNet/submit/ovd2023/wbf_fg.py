import json
import os
import sys
from ensemble_boxes import weighted_boxes_fusion
from toolz.curried import get, groupby
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import mmcv

curr_dir = os.path.dirname(__file__)

ann_json = '../Data/json_final_contest/test_fg.json'
img_prefix = '../Data/data_final_contest/test/'

save_dir = os.path.join(curr_dir, 'results')

cfg_dir = 'configs_cmp/ovdet2023/'
pth_dir = 'work_dirs/ovdet2023/'
cfg_pths = [
    ('cbv2_l_fg.py', 'cbv2_l_fg/bsl_3x/epoch_12.pth', 1),
    ('cas_hornet_gf_fg.py', 'cas_hornet_gf_fg/bsl_3x/epoch_12.pth', 1),
    ('cas_convnext_l_fg.py', 'cas_convnext_l_fg/bsl_3x/epoch_12.pth', 1),
    ('detectors_r101_fg.py', 'detectors_r101_fg/bsl_3x/epoch_12.pth', 1),
    ('cas_x101_fg.py', 'cas_x101_fg/bsl_3x/epoch_12.pth', 0.9),
    ('vfnet_x101_fg.py', 'vfnet_x101_fg/bsl_3x/epoch_12.pth', 0.7),
]


weights = [i[-1] for i in cfg_pths]

for i, (cfg, pth, _) in enumerate(cfg_pths):
    res_json = pth.replace('/','_')
    cfg = f'{cfg_dir}/{cfg}'
    pth = f'{pth_dir}/{pth}'

    cmd = f'PORT=2059 bash tools/dist_test.sh {cfg} {pth} 8 --format-only --cfg-options data.test.img_prefix={img_prefix} data.test.ann_file={ann_json} --eval-options "jsonfile_prefix={save_dir}/{res_json}"'
    if not os.path.exists(f'{save_dir}/{res_json}.bbox.json'):
        os.system(cmd)

with open(ann_json) as f:
    ann = json.load(f)

outs = []
for i, (cfg, pth, _) in enumerate(cfg_pths):
    res_json = pth.replace('/','_')
    outs.append(json.load(open(f'{save_dir}/{res_json}.bbox.json', 'r')))

gss = [groupby(get('image_id'), d) for d in outs]

annotations = []
for index, info in tqdm(enumerate(ann['images']), ncols=40):
    height = info['height']
    width = info['width']
    name = info['file_name']
    image_id = info['id']
    all_boxes, all_scores, all_labels = [], [], []
    for g in gss:
        boxes, scores, labels = [], [], []
        if image_id in g:
            for r in g[image_id]:
                b = r['bbox']
                b = [b[0], b[1], b[0]+b[2], b[1]+b[3]]
                boxes.append([b[0]/width, b[1]/height, b[2]/width, b[3]/height])
                scores.append(r['score'])
                labels.append(r['category_id'])
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
    boxes, scores, labels = weighted_boxes_fusion(all_boxes, all_scores, 
            all_labels, weights=weights, iou_thr=0.5, skip_box_thr=0.001, conf_type='max')
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        box = [box[0]*width, box[1]*height, box[2]*width, box[3]*height]
        box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        r = {
            'image_id': image_id,
            'category_id': int(label),
            'bbox': box,
            'score': score
        }
        annotations.append(r)
        if score>1: print(score)



result_file = f'{save_dir}/wbf.json'
mmcv.dump(annotations, result_file)
