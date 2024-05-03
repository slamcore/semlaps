__copyright__ = """
    SLAMcore Limited
    All Rights Reserved.
    (C) Copyright 2024

    NOTICE:

    All information contained herein is, and remains the property of SLAMcore
    Limited and its suppliers, if any. The intellectual and technical concepts
    contained herein are proprietary to SLAMcore Limited and its suppliers and
    may be covered by patents in process, and are protected by trade secret or
    copyright law.
"""

__license__ = "CC BY-NC-SA 3.0"

import argparse
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import numpy as np
from pprint import pprint
import time
import pandas
import os

unknown_label = 255
unknown_instance = 0
num_things_classes = 7

def export_labels(labels, out_path):
    labels_out = open(out_path, 'w')
    print('there are these classes in predictions:')
    print(np.unique(labels))
    for l in labels:
        labels_out.write(str(int(l)) + '\n')

def export_instances(instances, labels, scene_id, out_folder):
    instance_ids = np.unique(instances)
    os.makedirs(out_folder + '/pred_mask', exist_ok=True)
    all_inst_out = open(out_folder + '/' + scene_id + '.txt', 'w')
    inst_written_cnt = 0
    for inst_id in instance_ids:
        if inst_id == unknown_instance:
            continue
        inst_mask = (instances == inst_id)
        inst_labels = np.unique(labels[inst_mask])
        if len(inst_labels) > 1:
            print('multiple labels per instance: ' + str(inst_labels))
        inst_label = inst_labels[0]
        print('instance ' + str(inst_id) +' label ' + str(inst_label) + ' size ' + str(np.sum(inst_mask)))
        if inst_label >= num_things_classes:
            continue
        inst_fname = scene_id + '_' + str(inst_written_cnt + 1) + '.txt'
        mask_file_path = out_folder + '/pred_mask/' + inst_fname
        inst_mask_out = open(mask_file_path, 'w')
        for vert_inst in inst_mask:
            inst_mask_out.write(str(int(vert_inst)) + '\n')

        all_inst_out.write('pred_mask/' + inst_fname + ' ' + str(int(inst_label)) + ' ' + '1.0\n')
        inst_written_cnt += 1

def read_ply(ply_path):
    verts = None
    labels = None
    instances = None
    with open(ply_path, 'rb') as f:
        plydata = PlyData.read(f)
        for el in plydata.elements:
            if el.name == 'vertex':
                verts = np.stack([el.data['x'], el.data['y'], el.data['z']], axis=1)
                is_label = False
                is_instance = False
                for pr in el.properties:
                    if pr._name == 'label':
                        is_label = True
                    if pr._name == 'instance':
                        is_instance = True
                if is_label:
                    labels = el.data['label']
                if is_instance:
                    instances = el.data['instance']

    return verts, labels, instances, plydata

def associate_verts(verts_rec, verts_gt, assoc_thr):
    t_start = time.time()
    tree = KDTree(verts_rec, leaf_size=2)
    t_fin = time.time()
    dist, ind = tree.query(verts_gt, k=1)
    bad_inds = (dist[:, 0] > assoc_thr)
    t_q = time.time()
    print('tree construction took ' + str(t_fin-t_start) + ' query took ' + str(t_q - t_fin))
    return ind[:, 0], bad_inds, dist

def compute_chamfer(verts_gt, verts_rec, assoc_thr, exp_lbl, out_path):
    ind, bad_inds, dist = associate_verts(verts_rec, verts_gt, assoc_thr)
    dist_thr = 0.01
    while dist_thr <= assoc_thr:
        n_ok = np.sum(dist[:, 0] < dist_thr)
        print('{} m : {}'.format(dist_thr, n_ok * 100.0 / dist.shape[0]))
        dist_thr += 0.01
    good_inds = (dist[:, 0] <= assoc_thr)
    mean_chamfer = np.mean(dist[good_inds, 0])
    print('Mean Chamfer ' + str(mean_chamfer))
    if len(out_path) > 0:
        os.makedirs(out_path, exist_ok=True)
        report_dict = {
            'chamfer': [],
            'good': [],
            'bad' : []
        }
        report_dict['chamfer'].append(mean_chamfer)
        report_dict['good'].append(np.sum(good_inds))
        report_dict['bad'].append(np.sum(bad_inds))
        # create DataFrame
        df = pandas.DataFrame(report_dict)
        # save it
        report_file = os.path.join(out_path, exp_lbl + '.csv')
        columns = ["chamfer", "good", "bad"]
        header = ["#mean Chamfer", "inlier points", "outlier points"]
        df.to_csv(report_file, sep=",", index=False, columns=columns, header=header)
    return ind, bad_inds

def compute_iou(labels_gt_mapped, labels_rec_masked, rec_id2label, ply_gt, out_path):
    print('Class IoUs')
    report_dict = {
        "class": [],
        "iou": [],
        "gt": [],
        "rec": [],
        "int": [],
        "uni": []
    }

    rec_id2label[unknown_label] = 'unknown'

    for l in rec_id2label:
        inter = np.sum((labels_gt_mapped == l) & (labels_rec_masked == l))
        uni = np.sum((labels_gt_mapped == l) | (labels_rec_masked == l)) + 1e-6
        iou = 100.0 * inter / uni
        gt_num = np.sum(labels_gt_mapped == l)
        rec_num = np.sum(labels_rec_masked == l)
        print('{} ({}): {}, {} / {}'.format(rec_id2label[l], l, iou, inter, uni))
        report_dict['class'].append(l)
        report_dict['iou'].append(iou)
        report_dict['gt'].append(gt_num)
        report_dict['rec'].append(rec_num)
        report_dict['int'].append(inter)
        report_dict['uni'].append(uni)

    if len(out_path) > 0:
        # create DataFrame
        df = pandas.DataFrame(report_dict)
        # save it
        report_file = out_path + '_iou.csv'
        columns = ["class", "iou", "gt", "rec", "int", "uni"]
        header = ["#class id", "IoU [%]", "GT points", "rec points", "intersection", "union"]
        df.to_csv(report_file, sep=",", index=False, columns=columns, header=header)

        col_map = get_class_colours()
        colorize(ply_gt, col_map, labels_gt_mapped)
        ply_gt.write(out_path + '_gt.ply')
        colorize(ply_gt, col_map, labels_rec_masked)
        ply_gt.write(out_path + '_rec.ply')

def get_class_colours():
    return np.asarray([
        [255, 0, 255],
        [33, 138, 33],
        [138, 43, 255],
        [188, 142, 142],
        [199, 32, 133],
        [123, 252, 0],
        [0, 0, 138],
        [215, 215, 0],
        [139, 0, 0],
        [98, 150, 236],
        [128, 128, 128],
        [254, 228, 201]
    ])

def colorize(plydata, colour_map, labels):
    for el in plydata.elements:
        if el.name == 'vertex':
            label_inds = (labels >= 0) & (labels < 255)
            el.data['red'][label_inds] = colour_map[labels[label_inds], 0]
            el.data['green'][label_inds] = colour_map[labels[label_inds], 1]
            el.data['blue'][label_inds] = colour_map[labels[label_inds], 2]
            bad_label_inds = (labels == 255)
            el.data['red'][bad_label_inds] = 255
            el.data['green'][bad_label_inds] = 255
            el.data['blue'][bad_label_inds] = 255

def read_label_map(csv_path):
    csv_data = pandas.read_csv(csv_path, header=None)

    gt_ids = csv_data.values[:, 0]
    rec_ids = csv_data.values[:, 2]
    rec_labels = csv_data.values[:, 3]

    rec_id2label = {}
    for i in range(0, len(rec_ids)):
        rec_id2label[ rec_ids[i] ] = rec_labels[i]
    gt2rec = {}
    for i in range(0, len(gt_ids)):
        gt2rec[ gt_ids[i] ] = rec_ids[i]

    gt2rec[0] = unknown_label
    rec_id2label[unknown_label] = 0

    return gt2rec, rec_id2label

def remap_gt_labels(labels_gt):
    uniq = np.unique(labels_gt)
    labels_gt_mapped = unknown_label * np.ones_like(labels_gt)
    for i in range(0, len(uniq)):
        l = uniq[i]
        l_inds = np.nonzero(labels_gt == l)[0]
        if not l in gt2rec:
            labels_gt_mapped[l_inds] = unknown_label
            print('{} -> {}'.format(l, unknown_label))
        else:
            labels_gt_mapped[l_inds] = gt2rec[l]
            print('{} -> {}'.format(l, gt2rec[l]))
    return labels_gt_mapped

if __name__ == '__main__':
    #point association threshold, meters
    assoc_thr = 0.05
    is_instance_test = True

    parser = argparse.ArgumentParser()
    parser.description = "Match two point clouds"
    parser.add_argument(
        "-s",
        "--scannet",
        required=True,
        help=("Path to the ScanNet folder"),
    )
    parser.add_argument(
        "-c",
        "--scenecode",
        required=True,
        help=("Unique ID of the scene"),
    )
    parser.add_argument(
        "-r",
        "--rec",
        default="",
        required=True,
        help=("Path to the reconstructed point cloud in PLY format"),
    )
    parser.add_argument(
        "-l",
        "--labmap",
        default="",
        required=False,
        help=("Path to the label map in the CSV format"),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        required=False,
        help=("Path to the output dir to store colored meshes in PLY"),
    )

    # parse cmdline args
    parser_args = vars(parser.parse_args())
    scannet_path = parser_args['scannet']
    scene_code = parser_args['scenecode']
    out_path = parser_args["output"]
    gt_path = scannet_path + '/reconstructions/' + scene_code + '_vh_clean_2.labels.ply'
    rec_path = parser_args["rec"]
    labmap_path = scannet_path + '/configurations/map_indoor_12.csv'

    if len(labmap_path) > 0:
        gt2rec, rec_id2label = read_label_map(labmap_path)
        # print(gt2rec)
        # print(rec_id2label)

    verts_gt, labels_gt, inst_gt, ply_gt = read_ply(gt_path)
    print('Read GT:')
    print('Unique labels: ' + str(len(np.unique(labels_gt))))
    for l in np.unique(labels_gt):
        print('{}: {}'.format(l, np.sum(labels_gt == l)))

    verts_rec, labels_rec, inst_rec, ply_rec = read_ply(rec_path)
    print('Read Rec:')
    print('Rec Unique labels: ' + str(len(np.unique(labels_rec))))
    for l in np.unique(labels_rec):
        print('{}: {}'.format(l, np.sum(labels_rec == l)))


    print('GT -> REC')
    chamfer_folder = out_path + '/chamfer/'
    rec_inds, bad_inds = compute_chamfer(verts_gt, verts_rec, assoc_thr, scene_code +'_g2r', chamfer_folder)

    print('REC -> GT')
    compute_chamfer(verts_rec, verts_gt, assoc_thr, scene_code + '_r2g', chamfer_folder)

    if labels_rec is not None:
        #remap rec labels to a gt point cloud
        labels_rec = labels_rec[rec_inds]
        labels_rec[bad_inds] = unknown_label
        #remap gt labels to our indoor 12 label scheme
        labels_gt_mapped = remap_gt_labels(labels_gt)
        #mask out recognized labels for which gt is not known
        gt_mask = (labels_gt_mapped == unknown_label)
        labels_rec[gt_mask] = unknown_label

        semantic_details_folder = out_path + '/semantic_details/'
        os.makedirs(semantic_details_folder, exist_ok=True)
        compute_iou(labels_gt_mapped, labels_rec, rec_id2label, ply_gt, semantic_details_folder + scene_code)
        if is_instance_test:
            inst_rec = inst_rec[rec_inds]
            inst_rec[bad_inds] = unknown_instance
            inst_rec[gt_mask] = unknown_instance
        semantic_folder = out_path + '/semantic/'
        os.makedirs(semantic_folder, exist_ok=True)
        export_labels(labels_rec, semantic_folder + scene_code + '.txt')
        instance_folder = out_path + '/instance/'
        os.makedirs(instance_folder, exist_ok=True)
        if is_instance_test:
            assert(len(inst_rec) == len(labels_rec))
            export_instances(inst_rec, labels_rec, scene_code, instance_folder)
