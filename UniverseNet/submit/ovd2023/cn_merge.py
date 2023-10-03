from glob import glob
import json, os

curr_dir = os.path.dirname(__file__)

res = []
for i in (glob(f'{curr_dir}/results/cn_clip_*.json')):
    print(i)
    data = json.load(open(i))
    for d in data:
        res+=[{"image_id": d["image_id"], "category_id":d["category_id"], "bbox": d["bbox"], "score": d["score"]}]

os.makedirs(f'{curr_dir}/results', exist_ok=True)

with open(f'{curr_dir}/results/cn_result_bbox_clip.json', 'w') as fp:
    json.dump(res, fp, indent=1, separators=(',', ': '))

os.system(f'rm {curr_dir}/results/cn_results_fg.zip')
cmd = f'cd {curr_dir}/results && zip cn_results_fg.zip cn_result_bbox_clip.json'
os.system(cmd)