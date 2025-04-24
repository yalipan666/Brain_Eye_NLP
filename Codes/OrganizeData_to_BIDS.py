# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 18:45:15 2025

@author: panyz

organize data in the right folder with the right names for sharing data in the BIDS formate
"""

import numpy as np
import os
import shutil  # shell utilities

# set data path
path_data = r"X:\YPan_DataSharing\Data"

"""
Semantic violation datasets
"""
path_sou = r"Y:\Semantic\RawData"
subjects = ["20210702_b4bb","20210702_b5f3","20210708_b398","20210708_b5f6",
            "20210716_b5e6","20210719_b395","20210719_b4bc","20210826_b38f",
            "20210909_b4be","20210909_b5eb","20210910_b4bb","20210911_b3f9",
            "20210911_b588","20210914_b397","20210915_b38e","20210916_b4ba",
            "20210927_b3a2","20210927_b4b9","20211002_b3f4","20211004_b398",
            "20211005_b4be","20211007_b396","20211007_b4bc","20211011_b4c2",
            "20211014_b4bb","20211021_b395","20211022_b4bc","20211028_b3b9",
            "20211028_b4c2","20211104_b394","20211104_b4bc","20211108_b394",
            "20211111_b395","20211111_b4c0"]
task2_id = np.array([*range(5,14), *range(16,35)])
task_1 = "task-semantic"
task_2 = "task-lexical"

for i in range(len(subjects)):
    # create subject folder
    sid = "sub-" + f"{i+1}".zfill(3)  #e.g., sub-001
    sub_folder = rf"{path_data}\{sid}"  
    os.makedirs(sub_folder, exist_ok=True)
    
    # display
    print(f"Processing the participant of: {sid}.")
    
    if np.isin(i,task2_id-1):
        task_all = [task_1, task_2]
    else:
        task_all = [task_1]
        
    for task in task_all:
        # get prefix of files
        if task == task_1:
            prefix = ""
        else:
            prefix = "fa_"
        # create folder for task and copyfile files into it
        task_folder = rf"{sub_folder}\{task}"
        os.makedirs(task_folder,exist_ok=True)
        # copyfile PTB data
        source = rf"{path_sou}\PTB_data\{prefix}{subjects[i]}.mat"
        destination = rf"{task_folder}\{sid}_{task}_BEH.mat"
        if os.path.exists(source):
            shutil.copyfile(source, destination)
        else:
            print(f"Warning: {source} not found, skipping.")
        # copyfile ET data
        et_file = prefix + "".join([subjects[i][j] for j in [4,5,6,7,9,10,11,12]])
        source = rf"{path_sou}\EyeLink_data\{et_file}.asc"
        destination = rf"{task_folder}\{sid}_{task}_ET.asc"
        if os.path.exists(source):
            shutil.copyfile(source, destination)
        else:
            print(f"Warning: {source} not found, skipping.")
        # copyfile MEG data
        meg_subfile1 = subjects[i][slice(2,8)]
        meg_subfile2 = subjects[i][slice(9,13)]
        source = rf"{path_sou}\MEG_data\{subjects[i]}\{meg_subfile1}\{prefix}{meg_subfile2}"
        destination = rf"{task_folder}\{sid}_{task}_MEG"
        if os.path.exists(f"{source}.fif"):
            shutil.copyfile(f"{source}.fif", f"{destination}.fif")
        else:
            print(f"Warning: {source}.fif not found, skipping.")
        # when there're more than one .fif files
        for suffix in ["-1.fif","-2.fif","-3.fif"]:
            src_file = f"{source}{suffix}"
            dest_file = f"{destination}{suffix}"
            if os.path.exists(src_file):
                shutil.copyfile(src_file, dest_file)

 
    
print("All done!")
    
    
    
    
