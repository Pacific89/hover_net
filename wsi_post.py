from matplotlib import pyplot as plt
import cv2

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import json
import openslide

from misc.wsi_handler import get_file_handler
from misc.viz_utils import visualize_instances_dict


# wsi_path = "/home/user/Documents/Master/tcga_test/gdc_download_20211215_084557.672401/9baa25bd-c9d1-4280-bc17-43b90fafd4e0/three.svs"

wsi_path = "/home/user/Documents/Master/data/one/one.svs"

wsi_basename = "one"
wsi_json_path = "/home/user/Documents/Master/server_hover_recieve/one/"

wsi_one = openslide.open_slide(wsi_path)
print(wsi_one.dimensions)



mask_path_wsi = wsi_json_path + 'mask/' + wsi_basename + '.png'
thumb_path_wsi = wsi_json_path + 'thumb/' + wsi_basename + '.png'

thumb = cv2.cvtColor(cv2.imread(thumb_path_wsi), cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(cv2.imread(mask_path_wsi), cv2.COLOR_BGR2RGB)

# plot the low resolution thumbnail along with the tissue mask

# plt.figure(figsize=(15,8))

# plt.subplot(1,2,1)
# plt.imshow(thumb)
# plt.axis('off')
# plt.title('Thumbnail', fontsize=25)

# plt.subplot(1,2,2)
# plt.imshow(mask)
# plt.axis('off')
# plt.title('Mask', fontsize=25)

# plt.show()


json_path_wsi = wsi_json_path + 'json/' + wsi_basename + '.json'

bbox_list_wsi = []
centroid_list_wsi = []
contour_list_wsi = [] 
type_list_wsi = []

# add results to individual lists
with open(json_path_wsi) as json_file:
    data = json.load(json_file)
    mag_info = data['mag']
    nuc_info = data['nuc']
    for inst in nuc_info:
        inst_info = nuc_info[inst]
        inst_centroid = inst_info['centroid']
        centroid_list_wsi.append(inst_centroid)
        inst_contour = inst_info['contour']
        contour_list_wsi.append(inst_contour)
        inst_bbox = inst_info['bbox']
        bbox_list_wsi.append(inst_bbox)
        inst_type = inst_info['type']
        type_list_wsi.append(inst_type)


keys = nuc_info.keys()
print(nuc_info[list(keys)[-1]])
print(mag_info)

# define the region to select
x_tile = 4000
y_tile = 3000
w_tile = 15000
h_tile = 20000

coords = (x_tile, y_tile)
patch_level = -1
# load the wsi object and read region
wsi_ext =".svs"
# wsi_obj = get_file_handler(wsi_path, wsi_ext)
# wsi = openslide.open_slide(wsi_path)
# print(wsi.dimensions)
# wsi_tile = wsi.read_region(coords, patch_level, tuple([w_tile, h_tile]))
wsi_obj = get_file_handler(wsi_path, wsi_ext)
wsi_obj.prepare_reading(read_mag=mag_info)
wsi_tile = wsi_obj.read_region((x_tile,y_tile), (w_tile,h_tile))

coords_xmin = x_tile
coords_xmax = x_tile + w_tile
coords_ymin = y_tile
coords_ymax = y_tile + h_tile

tile_info_dict = {}
count = 0
for idx, cnt in enumerate(contour_list_wsi):
    cnt_tmp = np.array(cnt)
    cnt_tmp = cnt_tmp[(cnt_tmp[:,0] >= coords_xmin) & (cnt_tmp[:,0] <= coords_xmax) & (cnt_tmp[:,1] >= coords_ymin) & (cnt_tmp[:,1] <= coords_ymax)] 
    label = str(type_list_wsi[idx])
    if cnt_tmp.shape[0] > 0:
        cnt_adj = np.round(cnt_tmp - np.array([x_tile,y_tile])).astype('int')
        tile_info_dict[idx] = {'contour': cnt_adj, 'type':label}
        count += 1



type_info = {
    "0" : ["nolabe", [0  ,   0,   0]], 
    "1" : ["neopla", [255,   0,   0]], 
    "2" : ["inflam", [0  , 255,   0]], 
    "3" : ["connec", [0  ,   0, 255]], 
    "4" : ["necros", [255, 255,   0]], 
    "5" : ["no-neo", [255, 165,   0]] 
}

fig = plt.figure(figsize=(100,80))
overlaid_output = visualize_instances_dict(wsi_tile, tile_info_dict, type_colour=type_info, line_thickness=3)
plt.imshow(overlaid_output)
plt.axis('off')
plt.title('Segmentation Overlay')

for i in type_info:
    label = type_info[i][0]
    color = np.array(type_info[i][1])/255
    plt.plot([0,0], label = label, color=tuple(color))

plt.legend()
plt.savefig("hovernet_line2_dpi.jpg", dpi='figure')
# plt.show()