# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:41:37 2025

@author: PanYZ

Epoch the MEG data (.mat) into trials with sentence onset and offset as the start and end.

"""


import numpy as np
import h5py
import scipy.io

# set data path
path_save = r"X:\Brain_Eye_NLP\Data"
triggers = {"sent_on": 4, "sent_off": 8} # trigger for the event of "sentence on/offset"



def read_matlab_string_array(h5_dataset, file):
    """Dereference and decode MATLAB string cell arrays from HDF5."""
    return [
        ''.join(chr(c[0]) for c in file[obj_ref][:]) if file[obj_ref][:].ndim > 1
        else ''.join(chr(c) for c in file[obj_ref][:])
        for obj_ref in h5_dataset[:].flatten()
    ]


def process_task_semantic():
    """
    Semantic violation datasets
    """
    path_data = r"Y:\Semantic\Analyse_data"
    
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
    
    for i in range(1,2): #range(len(subjects)):
        # create subject folder
        sid = "sub-" + f"{i+1}".zfill(3)  #e.g., sub-001
        
        # display
        print(f"Processing the participant of: {sid}.")
        
        if np.isin(i,task2_id-1):
            task_all = [task_1, task_2]
        else:
            task_all = [task_1]
            
        for task in task_all:
            # get prefix of files
            if task == task_1:
                prefix = "sv_"
            else:
                prefix = "fa_"
            
            # epoch the MEG data acoording to sentence onset and offset
            # get the MEG triggers of sentence onset and offset
            folder_dir =  rf"{path_data}\{prefix}{subjects[i]}"
            # load the Event.mat based on its type
            try:
                Event = scipy.io.loadmat(rf"{folder_dir}\Event.mat",struct_as_record=False, squeeze_me=True)
                trig = Event["Event"].Trigger_MEG
            except NotImplementedError:
                with h5py.File(rf"{folder_dir}\Event.mat", 'r') as f:
                    trig = f["Event"]["Trigger_MEG"][:]
            
            # get the onset and offset timepoint of each sentence
            sent_on_trig = trig[1,np.where(trig[0,:] == triggers["sent_on"])]
            sent_off_trig = trig[1,np.where(trig[0,:] == triggers["sent_off"])]
          
            # get all the valide labels and their indices
            
            with h5py.File(rf"{folder_dir}\epoch_BL_Cross.mat", 'r') as f:
                # Dereference both label arrays
                all_labels = read_matlab_string_array(f["epoch_BL_Cross"]["hdr"]["label"], f)
                valid_labels = read_matlab_string_array(f["epoch_BL_Cross"]["label"], f)

                # Convert to NumPy arrays
                all_labels = np.array(all_labels)
                valid_labels = np.array(valid_labels)

                # Get indices of all valid_labels in all_labels
                label_to_indices = {
                    label: np.where(all_labels == label)[0]
                    for label in valid_labels
                }
                # flatten all indices
                all_matched_indices = np.concatenate(list(label_to_indices.values()))
                
            # epoch the MEG data using triggers
            with h5py.File(rf"{folder_dir}\data_icaclean.mat", 'r') as f:
                data = f["data"]  # Get the dataset object without loading data
                channel_matrix = data[:,all_matched_indices]
                
                # Initialize dictionary to store epochs and metadata
                epoch_data = {
                    'epochs': [],  # Will store the actual epoch data
                    'channels': valid_labels,  # Labels for the channels
                    }
                
                # Extract each epoch
                for t in range(len(sent_on_trig[0])):                    
                    # Convert trigger indices to integers and ensure they're valid
                    start_idx = int(sent_on_trig[0][t]) - 1  # convert to 0-based indexing
                    end_idx = int(sent_off_trig[0][t]) - 1   # convert to 0-based indexing
                    # Verify indices are valid
                    if start_idx < 0 or end_idx >= data.shape[0]:
                        print(f"Warning: Invalid indices for epoch {t}: start={start_idx}, end={end_idx}")
                        continue
                    # epoch the data
                    epoch_data['epochs'].append(channel_matrix[start_idx:end_idx+1,:])
                
                # Save the epoch data in HDF5 format
                output_file = rf"{path_save}\{sid}_{task}_MEGdata.h5"
                with h5py.File(output_file, 'w') as hf:
                    # Create a group for epochs
                    epochs_group = hf.create_group('epochs')
                    # Save each epoch as a separate dataset
                    for t, epoch in enumerate(epoch_data['epochs']):
                        epochs_group.create_dataset(f'epoch_{t}', data=epoch)
                    # Save valid labels as strings
                    dt = h5py.special_dtype(vlen=str)
                    labels = hf.create_dataset('channels', 
                                                  (len(epoch_data['channels']),), 
                                                  dtype=dt)
                    labels[:] = epoch_data['channels']
                    
                # release memory
                del epoch_data
    
    print("All done!")
    

def main():
    process_task_semantic()
 
if __name__ == "__main__":
    main()


    
    
    
    

