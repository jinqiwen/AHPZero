from os.path import join

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import io
from .random_dataset import RandDataset
from .episode_dataset import EpiDataset, CategoriesSampler, DCategoriesSampler
from .test_dataset import TestDataset
import pandas as pd
from .transforms import data_transform

from models.utils.comm import get_world_size

class ImgDatasetParam(object):
    ## update dir 'imgroot', "dataroot"
    DATASETS = {
        "imgroot": '/home/star/HAPZero-main/datasets',
        "dataroot": '/home/star/HAPZero-main/data/xlsa19',
        "image_embedding": 'images_label',
        "class_embedding": 'att'
    }

    @staticmethod
    def get(dataset):
        attrs = ImgDatasetParam.DATASETS
        attrs["imgroot"] = join(attrs["imgroot"], dataset)
        args = dict(
            dataset=dataset
        )
        args.update(attrs)
        return args

def build_dataloader(cfg, is_distributed=False):
    args = ImgDatasetParam.get(cfg.DATASETS.NAME)
    imgroot = args['imgroot']
    dataroot = args['dataroot']
    image_embedding = args['image_embedding']
    dataset = args['dataset']

    matcontent = io.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")

    img_files =np.squeeze(matcontent['image_files'])
    new_img_files = []
    for img_file in img_files:
        img_path = img_file[0]
        img_path = join(imgroot+"/"+"images/", img_path)
        new_img_files.append(img_path)

    new_img_files = np.array(new_img_files)
    label = matcontent['labels'].astype(int).squeeze()
    if dataset == 'GroupA':
        matcontent = io.loadmat(dataroot + "/split/splits_group_A.mat")
        trainvalloc = matcontent['trainval_loc'].squeeze()
        test_seen_loc = matcontent['test_seen_loc'].squeeze()
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze()
    elif dataset == 'GroupB':
        matcontent = io.loadmat(dataroot + "/split/splits_group_B.mat")
        trainvalloc = matcontent['trainval_loc'].squeeze()
        test_seen_loc = matcontent['test_seen_loc'].squeeze()
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze()
    else:
        matcontent = io.loadmat(dataroot + "/split/splits_group_C.mat")
        trainvalloc = matcontent['trainval_loc'].squeeze()
        test_seen_loc = matcontent['test_seen_loc'].squeeze()
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze()
    att_name = 'att'
    matcontent2 = io.loadmat(dataroot + "/allclasses_names.mat")
    cls_name = matcontent2['allclasses_names']

    matcontent3 = io.loadmat(dataroot + "/att/att_value.mat")
    matcontent3[att_name] = matcontent3[att_name]
    attribute = matcontent3[att_name][label]
    train_img = new_img_files[trainvalloc]
    train_label = label[trainvalloc].astype(int)
    train_att = attribute[train_label]

    train_id, idx = np.unique(train_label, return_inverse=True)
    train_att_unique = matcontent3[att_name][train_id]
    train_clsname = cls_name[train_id]

    num_train = len(train_id)
    train_label = idx
    train_id = np.unique(train_label)

    test_img_unseen = new_img_files[test_unseen_loc]
    test_label_unseen = label[test_unseen_loc].astype(int)
    test_id, idx = np.unique(test_label_unseen, return_inverse=True)
    att_unseen = matcontent3[att_name][test_id]
    test_clsname = cls_name[test_id]
    test_label_unseen = idx + num_train
    test_id = np.unique(test_label_unseen)

    train_test_att = np.concatenate((train_att_unique, att_unseen))
    train_test_id = np.concatenate((train_id, test_id))

    test_img_seen = new_img_files[test_seen_loc]
    test_label_seen = label[test_seen_loc].astype(int)
    _, idx = np.unique(test_label_seen, return_inverse=True)
    test_label_seen = idx

    att_unseen = torch.from_numpy(att_unseen).float()
    test_label_seen = torch.tensor(test_label_seen)
    test_label_unseen = torch.tensor(test_label_unseen)
    train_label = torch.tensor(train_label)
    att_seen = torch.from_numpy(train_att_unique).float()
    trian_class_counts = torch.bincount(train_label).float()
    seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy()))  # (150,)
    unseenclasses = torch.from_numpy(np.unique(test_label_unseen.cpu().numpy()))  # (50,)
    res = {
        'train_label': train_label,
        'train_att': train_att,
        'test_label_seen': test_label_seen,
        'test_label_unseen': test_label_unseen,
        'att_unseen': att_unseen,
        'att_seen': att_seen,
        'train_id': train_id,
        'test_id': test_id,
        'train_test_id': train_test_id,
        'train_clsname': train_clsname,
        'test_clsname': test_clsname,
        'seenclasses': seenclasses,
        'unseenclasses': unseenclasses,
        'trian_class_counts':trian_class_counts

    }
    if cfg.VISUAL:
        attr_path = './attribute/IP102/new_des.csv'
        attr_name = pd.read_csv(attr_path)['new_des']
        res['attr_name'] = attr_name
        res['train_img'] = train_img
        res['test_img_seen'] = test_img_seen
        res['test_img_unseen'] = test_img_unseen
    # train dataloader
    ways = cfg.DATASETS.WAYS#16
    shots = cfg.DATASETS.SHOTS#2
    data_aug_train = cfg.SOLVER.DATA_AUG#'resize_random_crop'
    img_size = cfg.DATASETS.IMAGE_SIZE#224
    transforms = data_transform(data_aug_train, size=img_size)
    if cfg.DATALOADER.MODE == 'random':
        dataset = RandDataset(train_img, train_att, train_label, transforms)

        if not is_distributed:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
            batch = ways*shots
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch, drop_last=True)
            tr_dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=8,
                batch_sampler=batch_sampler,
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            batch = ways * shots
            tr_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=8)


    elif cfg.DATALOADER.MODE == 'episode':
        n_batch = cfg.DATALOADER.N_BATCH#300
        ep_per_batch = cfg.DATALOADER.EP_PER_BATCH#1
        dataset = EpiDataset(train_img, train_att, train_label, transforms)
        if not is_distributed:
            sampler = CategoriesSampler(
                train_label,
                n_batch,
                ways,
                shots,
                ep_per_batch
            )
        else:
            sampler = DCategoriesSampler(
                train_label,
                n_batch,
                ways,
                shots,
                ep_per_batch
            )
        tr_dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)

    data_aug_test = cfg.TEST.DATA_AUG#'resize_crop'
    transforms = data_transform(data_aug_test, size=img_size)
    test_batch_size = cfg.TEST.IMS_PER_BATCH#100

    if not is_distributed:
        tu_data = TestDataset(test_img_unseen, test_label_unseen, transforms)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)

        ts_data = TestDataset(test_img_seen, test_label_seen, transforms)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)
    else:
        tu_data = TestDataset(test_img_unseen, test_label_unseen, transforms)
        tu_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tu_data, shuffle=False)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, sampler=tu_sampler,
            num_workers=4, pin_memory=False)

        ts_data = TestDataset(test_img_seen, test_label_seen, transforms)
        ts_sampler = torch.utils.data.distributed.DistributedSampler(dataset=ts_data, shuffle=False)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, sampler=ts_sampler,
            num_workers=4, pin_memory=False)

    return tr_dataloader, tu_loader, ts_loader, res

