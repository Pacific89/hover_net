# get paths for nuc files
# for each nuc file: 
#   get polygone
#   get regionsprops
#   add to dataframe

import os
import json
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
from scipy import ndimage
import cv2
import pandas as pd
from tqdm import tqdm


class nuCheck():

    def __init__(self, parent_folder, patch_size=256) -> None:
        self.parent_folder = parent_folder
        self.patch_size = patch_size
        self.nuc_files = self.get_nuc_files()



    def get_nuc_files(self):

        nuc_files = []

        if not os.path.isdir(self.parent_folder):
            print("Dir not found!")
            return []

        for root, dirs, files in os.walk(self.parent_folder):
            if "json" in root:
                for file in files:
                    # print(os.path.join(root, file))
                    nuc_files.append(os.path.join(root, file))


        # self.file_name = root.split("results")[0].split("\\")[-2]

        nuc_files_short = nuc_files[:1000]
        # print("Nucleus Files: ", nuc_files)
        # print("Keeping: ", len(nuc_files_short))

        return nuc_files_short

    def init_nuc_frame(self):
        self.property_list = ["area", "major_axis_length", "minor_axis_length", "eccentricity", "equivalent_diameter", "euler_number", "extent", "inertia_tensor", "inertia_tensor_eigvals", "moments", "moments_central", "moments_hu", "moments_normalized", "perimeter", "solidity"]
        dict_columns = self.property_list.copy()
        dict_columns.insert(0, "nuc_type")
        dict_columns.insert(0, "file_name")
        self.nucleus_frame = pd.DataFrame(columns=dict_columns)

    def work_patches(self):

        for nuc_file in self.nuc_files:
            self.read_json_to_pandas(nuc_file)

        print(self.nucleus_frame)

    def read_json_to_pandas(self, nuc_file):

        with open(nuc_file) as json_file:
            nuc_data = json.load(json_file)

        print("Patch: ", nuc_file)
        print("Nuclei: ", len(nuc_data["nuc"].items()))

        nuc_count = 0
        for idx, [inst_id, inst_info] in enumerate(nuc_data["nuc"].items()):
            overlay = np.zeros((self.patch_size, self.patch_size))
            if inst_info == None:
                continue
            else:
                inst_contour = np.array(inst_info["contour"])
                # check if nucleus is at border ( coords == 0 or coords = patch_size)
                if np.sum(inst_contour == 0) > 0 or np.sum(inst_contour == self.patch_size-1) > 0:
                    continue
                else:
                    cv2.fillPoly(overlay, pts =[inst_contour], color=(255,255,255))
                    label_img = label(overlay)
                    props = pd.DataFrame.from_dict(regionprops_table(label_img, properties=self.property_list))
                    props["file_name"] = self.file_name
                    props["nuc_type"] = inst_info["type"]
                    self.nucleus_frame = pd.concat([self.nucleus_frame, props], ignore_index=True)
                    nuc_count += 1
                    # plt.figure
                    # plt.imshow(label_img)
                    # plt.show()

        print("Nuclei kept: ", nuc_count)

    def save_data(self):

        file_path = "nuc_frame.json"
        self.nucleus_frame.to_json(file_path, orient="index")

    def plot_patch(self, patch_path):

        patch = cv2.imread(patch_path)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(patch)
        plt.show()


    def remove_edge_nucs(self, nuc_data):
        nuc_count = 0
        removed = 0
        list_to_remove = []
        for idx, [inst_id, inst_info] in enumerate(nuc_data["nuc"].items()):
            overlay = np.zeros((self.patch_size, self.patch_size))
            if inst_info == None:
                continue
            else:
                inst_contour = np.array(inst_info["contour"])
                # check if nucleus is at border ( coords == 0 or coords = patch_size)
                if np.sum(inst_contour == 0) > 0 or np.sum(inst_contour == self.patch_size-1) > 0:
                    list_to_remove.append(inst_id)
                    removed += 1
                    continue

        for x in list_to_remove:
            del nuc_data["nuc"][x]

        return nuc_data

    def checkPatch(self, nuc_file, percent, data_folder):

        with open(nuc_file) as json_file:
            nuc_data = json.load(json_file)
    
        # remove nulcei touching the patch border
        nuc_data_removed = self.remove_edge_nucs(nuc_data)

        neo_count = 0
        for idx, [inst_id, inst_info] in enumerate(nuc_data_removed["nuc"].items()):
            if nuc_data["nuc"][inst_id]["type"] == 1:
                neo_count += 1

        nuc_number = len(nuc_data_removed["nuc"])
        if neo_count == 0 or nuc_number == 0:
            ratio = 0
        else:
            ratio  = neo_count/nuc_number
        # print("Total: ", nuc_number)
        # print("Neoplastic: ", neo_count)
        # print("Ratio: ", ratio)

        keep = ratio > percent

        if not keep:
            file_name = nuc_file.split(".json")[0].split("/")[-1] + ".png"
            file_path = os.path.join(data_folder, file_name)
            # print(file_name)
            # print("Ratio: {0}",ratio)
            # self.plot_patch(file_path)

        return keep, neo_count

    def filterPatches(self, percent, data_folder):

        keep_list = []
        neo_list = []
        for nuc_file in tqdm(self.nuc_files):
            keep, neo_count = self.checkPatch(nuc_file, percent, data_folder)
            keep_list.append(keep)
            neo_list.append(neo_count)

        kept_files = np.array(self.nuc_files)[np.array(keep_list)]
        print("Kept files: ", sum(keep_list))
        print("Removed files: ", sum(~np.array(keep_list)))


if __name__ == "__main__":
    # parent_folder = "C:\\Users\\phili\\Desktop\\Uni\\Master_Thesis\\data\\nuc_data"
    parent_folder = "/media/user/easystore/patches_0_thresh_26wsi/5e107e6be51b4924be4cb6ed129d27cc"
    data_folder = "/media/user/easystore/patches_0_thresh_26wsi/5e107e6be51b4924be4cb6ed129d27cc/overlay"
    
    nucheck = nuCheck(parent_folder=parent_folder)
    nucheck.init_nuc_frame()
    # nucheck.work_patches()
    # nucheck.save_data()
    nucheck.filterPatches(percent=0.85, data_folder=data_folder)
