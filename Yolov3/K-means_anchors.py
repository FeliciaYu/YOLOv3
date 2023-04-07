import glob
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x*y
    area1 = box[0]*box[1]
    area2 = cluster[:,0]*cluster[:,1]

    return intersection/(area1+area2-intersection)

