# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:00:28 2025

@author: panyz

Processing eye tracking data for reading tasks (with rapid invisible frequency  
                                                tagging presentation).
    please note that the extraction of eye movement data is only from one eye, 
    i.e., monocular tracking 

Getting all  fixations when reading a given sentence, including the information
about: the sentence, fixation durations, fixation onset wrt the sent_on trigger, 
fixated_word_ID for each fixation (scan path). All in the dict format

"""

from pathlib import Path
import scipy.io
import numpy as np
import re
import json


# set path
path_data = Path(r"X:\YPan_DataSharing\Data") # get the list of subject folders
path_save = r"X:\Brain_Eye_NLP\Data"

# set all parameteres
triggers = {"sent_on": "Trigger_4", "sent_off": "Trigger_8"} # trigger for the event of "sentence on/offset"
y_error = 200 # the tolerant error in y coordinates for locating words, unit in pixels


# convert Numpy types into regular python types that can be written into Json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # for everything else, use the default method
        return super(NumpyEncoder, self).default(obj)
        


def locate_the_fixation(x,y,allwordloc):
    """
    locate each fixation to the words in a given sentence
    Parameters
    ----------
    x : list
        a list of floats numbers that represent the x coordinates of the fixation
    y : list
        a list of floats numbers that represent the y coordinates of the fixation
    word_coord : array
        ndarray that contains the xy coordinates of each word in the sentence
        dim: num_sentence*number_words*4, for the 3rd dim [x_start,y_start,x_end,y_end]

    Returns
    -------
    word_id : list
        a list of number of the word_id that the fixations were at, i.e., the scan path of this sentence

    """
    
    # remove empty cells
    allwordloc[~allwordloc.astype(bool)] = np.nan
    
    # calculate word boundaries
    xspace = allwordloc[1,0] - allwordloc[0,2]
    xposlim = np.column_stack((allwordloc[:,0] - xspace, allwordloc[:,2]))
    yposlim = np.column_stack((allwordloc[:,1] - y_error, allwordloc[:,3] + y_error))

    word_id = []
    for i in range(len(x)): # loop over words
        idx = np.where((x[i] > xposlim[:,0]) & (x[i] < xposlim[:,1]) &
                           (y[i] > yposlim[:,0]) & (y[i] < yposlim[:,1]))[0]
        word_id.append(int(idx[0]) if len(idx)>0 else np.nan)
       
    return word_id


def get_eye_data(et_file, word_coord, sent_mat):
    """
    Process eye movement data from an ASC file
    
    Parameters:
    -----------
    et_file: str
        Path to the ASC file
    word_coord: numpy.ndarray
        Matrix of word coordinates [trial*words*4], used to locate the eye fixations to specific words
    triggers: dict
        Dictionary containing trigger information of the eye movement data
  
    Returns:
    --------
    dict
        Dictionary containing eye movement data for each sentence
    """
    
    #initialization
    sentence_scr = 0 # there's a sentence on screen or not
    sentence_data = []
   
    with open(et_file, "r") as f:
        for line in f:
            # check for trial onset and get the sentence_id
            match = re.search(r"Sentence_ (\d+)", line)
            if match:
                sent_id = int(match.group(1))
                i = sent_id - 1 # python indexing from 0
                this_sent = {} # initialize dict for this sentence
                this_sent["sentence_id"] = sent_id
                sentence = sent_mat[i]
                sentence = [item for item in sentence if item != []]
                this_sent["sentence_material"] = sentence
                # initialize the lists for xy coordinates and durations
                start_time = []
                fix_dur = []
                x = []
                y = []
            
            # check for sentence onset
            if triggers["sent_on"] in line:
                sentence_scr = 1
                match = re.search(r'\d+', line)
                senton_time = int(match.group(0))
            elif triggers["sent_off"] in line:
                sentence_scr = 0
                # locate the fixation
                word_id = locate_the_fixation(x,y,word_coord[i])
                # save out sentence information
                valid_idx = ~np.isnan(word_id) # where fixations are on a word rather than somewhere else on the screen
                # reformatting to np.array as list doesn't support fance bool indexing
                word_id = np.array(word_id)
                fix_dur = np.array(fix_dur)
                start_time = np.array(start_time)
                this_sent["scan_path"] = word_id[valid_idx].astype(int)
                # re-align the first fixation to the onset of sentence, in case it starts before the sentence onset--i.e., staring at the starting box
                fix_dur[0] = fix_dur[0] + start_time[0]
                this_sent["fixation_durations"] = fix_dur[valid_idx]
                start_time[0] = 0
                this_sent["fixation_onset_wrt_sentence_onset"] = start_time[valid_idx]
                sentence_data.append(this_sent)
                
            # when there's a sentence on screen
            if sentence_scr:
                # get the fixation data 
                #[StartTime EndTime Duration AveragedXPosition AveragedYPosition AveragedPupilSize]]
                if "EFIX" in line:
                    vals = re.findall(r"\d+\.?\d*", line)
                    start_time.append(int(vals[0])-senton_time)
                    fix_dur.append(int(vals[2]))
                    x.append(float(vals[3]))
                    y.append(float(vals[4]))
               
            if "end of block" in line:
                break

    return sentence_data
    

def process_subject(subject_dir):
    """
    Process eye movement data for a single subject, looping over tasks
    
    Parameters: 
    -----------
    subject_dir: Path
        Path to the directory of subject folder
    task_dir: Path 
        Path to the directory of each task 

    Returns:
    --------
    task_data: dict
        dictionary with sentence materials and fixation data for all sentences
    """

    # loop over tasks 
    for task_dir in subject_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        # load behavioural data to get word coordinates and sentence materials
        beh_file = list(task_dir.glob("*_BEH.mat"))[0]
        beh_data = scipy.io.loadmat(beh_file, struct_as_record=False, squeeze_me=True)
        sent_mat = beh_data["Para"].SentMat
        
        """ frequency tagging mode display each sentences repeatedly in 4 
        quandrants, so the coordinates from psychotoolbox is halfed. When we 
        compare the coordinates from eye link to psychotoolbox, the coordinates 
        needs to be doubled"""
        word_coord = beh_data["Result"].WordLocation
        word_coord = word_coord*2
        
        # processing eye movement data for each sentence
        et_file = list(task_dir.glob("*_ET.asc"))[0]
        task_data = get_eye_data(str(et_file), word_coord,sent_mat) # a list of sentence_dict
        
        # save out data
        output_file = rf"{path_save}\{subject_dir.name}_{task_dir.name}_EYEdata.json"
        with open(output_file,"w") as f:
            json.dump(task_data, f, indent=4, cls=NumpyEncoder)
        print(f"Data saved to {output_file}")
  

# loop over participant folders
def main():
    """
    Main function to process all subjects' eye movment data

    Parameters: 
    -----------
    path_data: directory of the data 
    """     
    
    # loop over subject, processing each subject 
    for subject_dir in path_data.iterdir():
        if not subject_dir.is_dir():
            continue
        print(f"Processing subject: {subject_dir.name}")
        process_subject(subject_dir)

        
    # # only run one specific subject
    # dirs = [d for d in path_data.iterdir() if d.is_dir()]
    # subject_dir = dirs[2]
    # process_subject(subject_dir)

if __name__ == "__main__":
    main()
