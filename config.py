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


import os
import copy
import socket
from addict import Dict
import yaml


scannet_root_dict = {
    "jingwen": "/media/jingwen/Data/scannet/scans",
    "jingwen-slamcore": "/home/jingwen/data/scannet_val_minimal_skip_5/scans",
    "sasha-desktop": "/data/data/semlaps/scannet_root/"
}

slamcore_root_dict = {
    "jingwen-slamcore": "/home/jingwen/data/slamcore_data_sasha_gt",
    "sasha-desktop": "/data/data/semlaps/slamcore_root/"
}


def get_scannet_root():
    hostname = socket.gethostname()
    if hostname in scannet_root_dict:
        return scannet_root_dict[hostname]
    else:
        # raise NotImplementedError
        print("Default data path not defined!!! Need to mannually specify data path!!!")
        return None


def get_scannet_test_root():
    return get_scannet_root() + "_test"


def get_slamcore_root():
    return slamcore_root_dict[socket.gethostname()]


class ForceKeyErrorDict(Dict):
    def __missing__(self, key):
        raise KeyError(key)


def load_yaml(path):
    with open(path, encoding='utf8') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = ForceKeyErrorDict(**config_dict)
    return config


def save_config(datadict: ForceKeyErrorDict, path: str):
    datadict = copy.deepcopy(datadict)
    with open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(datadict.to_dict(), outfile, default_flow_style=False, sort_keys=False)
