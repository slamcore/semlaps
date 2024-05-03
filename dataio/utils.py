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

from collections import OrderedDict
from copy import deepcopy
import numpy as np
import plyfile


color_encoding_nyu40 = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('wall', (174, 199, 232)),
    ('floor', (152, 223, 138)),
    ('cabinet', (31, 119, 180)),
    ('bed', (255, 187, 120)),
    ('chair', (188, 189, 34)),
    ('sofa', (140, 86, 75)),
    ('table', (255, 152, 150)),
    ('door', (214, 39, 40)),
    ('window', (197, 176, 213)),
    ('bookshelf', (148, 103, 189)),
    ('picture', (196, 156, 148)),
    ('counter', (23, 190, 207)),
    ('blinds', (178, 76, 76)),
    ('desk', (247, 182, 210)),
    ('shelves', (66, 188, 102)),
    ('curtain', (219, 219, 141)),
    ('dresser', (140, 57, 197)),
    ('pillow', (202, 185, 52)),
    ('mirror', (51, 176, 203)),
    ('floormat', (200, 54, 131)),
    ('clothes', (92, 193, 61)),
    ('ceiling', (78, 71, 183)),
    ('books', (172, 114, 82)),
    ('refrigerator', (255, 127, 14)),
    ('television', (91, 163, 138)),
    ('paper', (153, 98, 156)),
    ('towel', (140, 153, 101)),
    ('showercurtain', (158, 218, 229)),
    ('box', (100, 125, 154)),
    ('whiteboard', (178, 127, 135)),
    ('person', (120, 185, 128)),
    ('nightstand', (146, 111, 194)),
    ('toilet', (44, 160, 44)),
    ('sink', (112, 128, 144)),
    ('lamp', (96, 207, 209)),
    ('bathtub', (227, 119, 194)),
    ('bag', (213, 92, 176)),
    ('otherstructure', (94, 106, 211)),
    ('otherfurniture', (82, 84, 163)),
    ('otherprop', (100, 85, 144)),
])

color_encoding_scannet20 = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('wall', (174, 199, 232)),
    ('floor', (152, 223, 138)),
    ('cabinet', (31, 119, 180)),
    ('bed', (255, 187, 120)),
    ('chair', (188, 189, 34)),  # coco62 -> scannet5
    ('sofa', (140, 86, 75)),  # coco63 -> scannet6
    ('table', (255, 152, 150)),  # coco67, 189 -> scannet7
    ('door', (214, 39, 40)),  # coco112 -> scannet8
    ('window', (197, 176, 213)),  # coco180, 181 -> scannet9
    ('bookshelf', (148, 103, 189)),
    ('picture', (196, 156, 148)),
    ('counter', (23, 190, 207)),  # coco107 -> scannet12
    ('desk', (247, 182, 210)),
    ('curtain', (219, 219, 141)),  # coco109 -> scannet14
    ('refrigerator', (255, 127, 14)),  # coco82 -> scannet15
    ('showercurtain', (158, 218, 229)),
    ('toilet', (44, 160, 44)),  # coco70 -> scannet17
    ('sink', (112, 128, 144)),  # coco81 -> scannet18
    ('bathtub', (227, 119, 194)),
    ('otherfurniture', (82, 84, 163)),
])


stuff_color = (31, 119, 180)

color_encoding_scannetX = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('wall', (174, 199, 232)),  # coco171, 175, 176, 177, 199 -> scannet1
    ('floor', (152, 223, 138)),  # coco118, 190 -> scannet2
    ('cabinet', stuff_color),  # coco188 -> scannet3
    ('bed', stuff_color),  # coco65 -> scannet4
    ('chair', stuff_color),  # coco62 -> scannet5
    ('sofa', stuff_color),  # coco63 -> scannet6
    ('table', stuff_color),  # coco67, 189 -> scannet7
    ('door', (214, 39, 40)),  # coco112 -> scannet8
    ('window', (197, 176, 213)),  # coco180, 181 -> scannet9
    ('bookshelf', stuff_color),
    ('picture', stuff_color),
    ('counter', stuff_color),  # coco107 -> scannet12
    ('desk', stuff_color),
    ('curtain', stuff_color),  # coco109 -> scannet14
    ('refrigerator', stuff_color),  # coco82 -> scannet15
    ('showercurtain', stuff_color),
    ('toilet', stuff_color),  # coco70 -> scannet17
    ('sink', stuff_color),  # coco81 -> scannet18
    ('bathtub', stuff_color),
    ('otherfurniture', stuff_color),
])


def nyu40_to_scannet20(label):
    """Remap a label image from the 'nyu40' class palette to the 'scannet20' class palette """

    # Ignore indices 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26. 27. 29. 30. 31. 32, 35. 37. 38, 40
    # Because, these classes from 'nyu40' are absent from 'scannet20'. Our label files are in
    # 'nyu40' format, hence this 'hack'. To see detailed class lists visit:
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt ('nyu40' labels)
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt ('scannet20' labels)
    # The remaining labels are then to be mapped onto a contiguous ordering in the range [0,20]

    # The remapping array comprises tuples (src, tar), where 'src' is the 'nyu40' label, and 'tar' is the
    # corresponding target 'scannet20' label
    remapping = [(0, 0), (13, 0), (15, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (25, 0), (26, 0), (27, 0),
                 (29, 0), (30, 0), (31, 0), (32, 0), (35, 0), (37, 0), (38,
                                                                        0), (40, 0), (14, 13), (16, 14), (24, 15), (28, 16), (33, 17),
                 (34, 18), (36, 19), (39, 20)]
    for src, tar in remapping:
        label[np.where(label == src)] = tar
    return label


def scannet20_to_nyu40(label):
    remapping = [(0, 0), (13, 14), (14, 16), (15, 24), (16, 28),
                 (17, 33), (18, 34), (19, 36), (20, 39)]
    label_nyu40 = deepcopy(label)
    for src, tar in remapping:
        label_nyu40[np.where(label == src)] = tar
    return label_nyu40


def create_label_image(output, color_palette):
    """Create a label image, given a network output (each pixel contains class index) and a color palette.
    Args:
    - output (``np.array``, dtype = np.uint8): Output image. Height x Width. Each pixel contains an integer,
    corresponding to the class label of that pixel.
    - color_palette (``OrderedDict``): Contains (R, G, B) colors (uint8) for each class.
    """

    label_image = np.zeros(
        (output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_palette):
        label_image[output == idx] = color_palette[color]
    return label_image


def vert_label_to_color(vert_label, color_palette):
    """Create a label image, given a network output (each pixel contains class index) and a color palette.
    Args:
    - output (``np.array``, dtype = np.uint8): Output image. Height x Width. Each pixel contains an integer,
    corresponding to the class label of that pixel.
    - color_palette (``OrderedDict``): Contains (R, G, B) colors (uint8) for each class.
    """

    vert_color = np.zeros((vert_label.shape[0], 3))
    for idx, color in enumerate(color_palette):
        vert_color[vert_label == idx] = color_palette[color]
    return vert_color


def class_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:
        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.
    References: https://arxiv.org/abs/1606.02147
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.
    """
    class_count = 0
    total = 0
    for batch_data in dataloader:
        label = batch_data["label"]
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def read_ply(ply_path):
    verts = None
    colors = None
    normals = None
    labels = None
    faces = None
    with open(ply_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)
        for el in plydata.elements:
            if el.name == "vertex":
                # vertices [V, 3]
                verts = np.stack([el.data["x"], el.data["y"], el.data["z"]], axis=1).astype(np.float32)
                
                # vertex_colors [V, 3]
                if "red" in el.data.dtype.names:
                    colors = np.stack([el.data["red"], el.data["green"], el.data["blue"]], axis=1).astype(np.float32) / 255.
                
                # vertex_normals [V, 3]
                if 'Nx' in el.data.dtype.names:
                    normals = np.stack([el.data["Nx"], el.data["Ny"], el.data["Nz"]], axis=1)
                
                # vertex_labels [V,]
                is_label = False
                for pr in el.properties:
                    if pr._name == "label":
                        is_label = True
                if is_label:
                    labels = el.data["label"]
                
            if el.name == "face":
                flist = []
                for f in el.data:
                    flist.append(f[0])
                faces = np.asarray(flist)
    
    mesh_dict = {
        "verts": verts,
        "colors": colors,
        "normals": normals,
        "labels": labels,
        "faces": faces,
        "plydata": plydata
    }

    return mesh_dict


def get_frames_of_a_scene(scene_filename):
    frames = []
    with open(scene_filename, "r") as f:
        pairs = f.readlines()
        for p in pairs:
            frames.append(int(p.strip()))
    return frames


def get_multi_views_of_a_scene(scene_filename):
    """
    Get a list of multi-view frame_ids from pre-computed file, in the format of:
    0, 20, 40
    20, 0, 40
    40, 20, 60
    ...
    the first one is the reference view (major view that features from all other views are warped to)
    :param scene_filename: .txt file that saves the multi-view ids
    :return: list of all multi-view frames
    """
    multi_views = []
    with open(scene_filename, "r") as f:
        views = f.readlines()
        for v in views:
            ids = v.strip().split(" ")
            multi_views.append([int(x) for x in ids])
    return multi_views


def get_scene_list(scene_file):
    scene_list = []
    with open(scene_file, "r") as f:
        scenes = f.readlines()
        for scene in scenes:
            scene = scene.strip()  # remove \n
            scene_list.append(scene)
    return scene_list


def save_scene_list(scene_list, save_file):
    with open(save_file, "w") as f:
        for i, scene in enumerate(scene_list):
            line = scene
            if i < len(scene_list) - 1:
                line += "\n"
            f.write(line)
