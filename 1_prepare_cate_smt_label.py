import numpy as  np
import json
from pycocotools.coco import COCO
path = 'dataset/NSD/'

train_annotation_file = path + 'annots/instances_train2017.json'
coco= COCO(train_annotation_file)

images = coco.loadImgs(coco.getImgIds())


train_image_dict = {}

for img in images:
    img_id = img['id'] 
    ann_ids = coco.getAnnIds(imgIds=img_id)  
    annotations = coco.loadAnns(ann_ids)  

    annot_dict = []
    for i, annot in enumerate(annotations):
        area = annot['area']
        category_id = annot['category_id']
        category_info = coco.loadCats(category_id)[0]
        supercategory = category_info['supercategory']
        name = category_info['name']
        annot_dict.append({
            'area': area,
            'supercategory': supercategory,
            'name': name
        })
    train_image_dict[img_id] = annot_dict

test_annotation_file = path + 'annots/instances_val2017.json'
coco= COCO(test_annotation_file)

images = coco.loadImgs(coco.getImgIds())  

test_image_dict = {}

for img in images:
    img_id = img['id']  
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    annot_dict = []
    for i, annot in enumerate(annotations):
        area = annot['area']
        category_id = annot['category_id']
        category_info = coco.loadCats(category_id)[0]
        supercategory = category_info['supercategory']
        name = category_info['name']
        annot_dict.append({
            'area': area,
            'supercategory': supercategory,
            'name': name
        })
    test_image_dict[img_id] = annot_dict
image_dict = train_image_dict | test_image_dict
np.save(path + 'processed_data/image_info.npy', image_dict)

# =============================================================
vocabs_cate, vocabs_smt = {}, {}
path = r'dataset/NSD/annots/instances_val2017.json'

coco_cat = json.load(open(path, "r", encoding="utf-8"))
coco_cat = coco_cat['categories']
# -----------------------------------------------------------
id2cat = {0: 'person', 1: 'animal', 2: 'appliance', 3: 'vehicle', 4: 'food', 5: 'furniture',
                     6: 'electronic', 7: 'indoor', 8: 'outdoor', 9: 'kitchen', 10: 'sports', 11: 'accessory'}
cat2id = [(v, k) for k, v in id2cat.items()]
cat2id = dict(cat2id)

vocabs_cate['coco_cat'] = coco_cat
vocabs_cate['id2cat'] = id2cat
vocabs_cate['cat2id'] = cat2id
vocabs_cate['cat_size'] = len(cat2id)

# -----------------------------------------------------------
id2smt = []
[id2smt.append(x['name']) for x in coco_cat if x['name'] not in id2smt]  # 'supercategory', 'name'

id2smt = [(i, x) for i, x in enumerate(id2smt)]
id2smt = dict(id2smt)

smt2id = [(v, k) for k, v in id2smt.items()]
smt2id = dict(smt2id)

vocabs_smt['coco_cat'] = coco_cat
vocabs_smt['id2smt'] = id2smt
vocabs_smt['smt2id'] = smt2id
vocabs_smt['smt_size'] = len(smt2id)

json.dump(vocabs_cate, open(f"dataset/NSD/processed_data/vocabs_cate.json", 'w', encoding='utf-8'))
json.dump(vocabs_smt, open(f"dataset/NSD/processed_data/vocabs_smt.json", 'w', encoding='utf-8'))