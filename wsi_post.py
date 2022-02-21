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
from skimage.measure import label, regionprops

from misc.wsi_handler import get_file_handler
from misc.viz_utils import visualize_instances_dict


# wsi_path = "/home/user/Documents/Master/tcga_test/gdc_download_20211215_084557.672401/9baa25bd-c9d1-4280-bc17-43b90fafd4e0/three.svs"

wsi_path = "/media/user/easystore/HRD-Subset/DigitalSlide_A1M_2S_1_20190127153640117/DigitalSlide_A1M_2S_1_20190127153640117.svs"

wsi_basename = wsi_path.split("/")[-1].split(".svs")[0]
print(wsi_basename)
wsi_json_path = "/media/user/easystore/HRD-Subset/DigitalSlide_A1M_2S_1_20190127153640117/results/d7ca4f688ae04bad9627b5b14956881d"

wsi_one = openslide.open_slide(wsi_path)
print(wsi_one.dimensions)


wsi_png = wsi_basename + ".png"
mask_path_wsi = os.path.join(wsi_json_path, 'mask', wsi_png)
thumb_path_wsi = os.path.join(wsi_json_path, 'thumb', wsi_basename + '.png')

thumb = cv2.cvtColor(cv2.imread(thumb_path_wsi), cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(cv2.imread(mask_path_wsi), cv2.COLOR_BGR2RGB)

label_mask = label(mask)
props = regionprops(label_mask)

areas = []
for prop in props:
    areas.append(prop.area)

# get largest object
max_prop = props[np.argmax(areas)]
bbox = max_prop.bbox
print(bbox)

top_left = [bbox[0], bbox[1]]
bot_right = [bbox[3], bbox[4]]

y_mask_ratio = top_left[0] / mask.shape[0]
y_original = int(wsi_one.dimensions[1]*y_mask_ratio)

y_original += 15000

x_mask_ratio = top_left[1] / mask.shape[1]
x_original = int(wsi_one.dimensions[0]*x_mask_ratio)

x_original += 16000

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


json_path_wsi = os.path.join(wsi_json_path, 'json', wsi_basename + '.json')

bbox_list_wsi = []
centroid_list_wsi = []
contour_list_wsi = [] 
type_list_wsi = []
patch_size = 1000

# add results to individual lists
with open(json_path_wsi) as json_file:
    data = json.load(json_file)
    mag_info = data['mag']
    nuc_info = data['nuc']
    for inst in nuc_info:
        inst_info = nuc_info[inst]
        inst_centroid = inst_info['centroid']
        if inst_centroid[0] > x_original and inst_centroid[1] > y_original and inst_centroid[0] < x_original+patch_size and inst_centroid[1] < y_original+patch_size:
            centroid_list_wsi.append(inst_centroid)
            inst_contour = inst_info['contour']
            contour_list_wsi.append(inst_contour)
            inst_bbox = inst_info['bbox']
            bbox_list_wsi.append(inst_bbox)
            inst_type = inst_info['type']
            type_list_wsi.append(inst_type)


# keys = nuc_info.keys()
print("Kept Nuclei: ", len(centroid_list_wsi))
print(mag_info)

# define the region to select
x_tile = x_original
y_tile = y_original
w_tile = patch_size
h_tile = patch_size

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
overlaid_output = visualize_instances_dict(wsi_tile, tile_info_dict, type_colour=type_info, line_thickness=2)
plt.imshow(overlaid_output)
plt.axis('off')
plt.title('Segmentation Overlay')

for i in type_info:
    label = type_info[i][0]
    color = np.array(type_info[i][1])/255
    plt.plot([0,0], label = label, color=tuple(color))

plt.legend()
# plt.savefig("hovernet_line2_dpi.jpg", dpi='figure')
fig2 = plt.figure(figsize=(100,80))
plt.imshow(wsi_tile)
plt.axis('off')
plt.title('Original')
plt.show()