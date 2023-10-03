import json

for subset in ['train', 'test']:
    ff = f'./json_final_contest/{subset}.json'
    inp = json.load(open(ff))

    categories = [{'id': 1, 'name': 'fg'}]

    inp['categories'] = categories

    if 'annotations' in inp:
        anns = inp['annotations']
        for i in range(len(anns)):
            anns[i]['category_id']=1
        inp['annotations']=anns

    with open(ff.replace('.json', '_fg.json'), 'w') as f:
        json.dump(inp, f)
