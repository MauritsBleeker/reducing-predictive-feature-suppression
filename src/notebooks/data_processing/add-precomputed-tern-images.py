import pickle
import os
import numpy as np


root = '/image_caption_retrieval/datasets/'
train_pickle = 'training_set_ltd_l.pickle'
feats_data_path = '//datasets/TERN/data/coco/features_36/cocobu_att'
box_data_path = '/datasets/TERN/data/coco/features_36/cocobu_box'

def add_precomputed_features_tern(pickle_file):
    print("start loading data")
    dataset = pickle.load(open(os.path.join(root, 'coco', pickle_file),'rb'))
    print("data loaded")
    for i, _id in enumerate(dataset['images'].keys()):
        cocoid = dataset['images'][_id]['cocoid']

        img_feat_path = os.path.join(feats_data_path, '{}.npz'.format(cocoid))
        img_box_path = os.path.join(box_data_path, '{}.npy'.format(cocoid))

        image = np.load(img_feat_path)['feat']
        img_boxes = np.load(img_box_path)

        dataset['images'][_id]['image'] = image
        dataset['images'][_id]['img_boxes'] = img_boxes

        if i % 100 == 0:
            print(i)
            print(cocoid)

    return dataset


train_tern = add_precomputed_features_tern(train_pickle)
print("start dumping pickle")
print(len(train_tern["images"].keys()))
pickle.dump(train_tern, open(os.path.join(root, 'coco', 'train_set_ltd_l_tern.pickle'),'wb'))
print("done dumping pickle")
