# Elizaveta Zavialova
# Here, EEG props (file describing the EEG data) are converted into model-compatible data
# For example, if the EEG data was collected while main speaker was vocalizing story 1 chunk 2,
# we need to access the waveform, the POS tags, the word onsets, etc. from the data of chunk 2

import scipy.io as sci
import numpy as np
import os
import eelbrain_utilities as eel_utils
import winsound
import utilities
# -----------------------CONSTANTS-----------------------

def get_number_of_time_points():
    return 6000

def get_number_of_trials(subject):
    if subject in eel_utils.get_weird_subjects():
        return 60
    return 75

def extract_props(subject):
    path = f"{os.getcwd()}\\4Elizaveta\{subject}_ENV.mat"
    mat = {}
    sci.loadmat(path, mat, variable_names=["PROPS"])
    mat = mat["PROPS"]
    return mat

# In this function we load the specified feature from the sampled arrays of the corresponding story
# i.e. we can load sampling_pos_story1_bool.mat, sampling_phoneme_story1_bool.mat, etc.
def load_unsliced_feature_matlab_array(feature, story):
    path = f"{os.getcwd()}/sampled_data/story{story}/{feature}"
    loaded = {}
    sci.loadmat(f"{path}", loaded, appendmat=True)
    loaded = loaded[feature]
    return loaded

# This function splices the feature fragments for a relevant time course from the large time course array
def slice_features(story, chunk, feature):
    N_TIME = get_number_of_time_points()
    features = load_unsliced_feature_matlab_array(feature, story)
    features = features[:, chunk*N_TIME:(chunk+1)*N_TIME]
    return features

def get_story(subject, trial_nr):
    trial_nr = trial_nr - 1  # to convert from trial numbers that start from 1
    COL_ATTENTION = 2  # This specifies which story the subject was paying attention to
    props = extract_props(subject)
    trial_props = props[trial_nr]  # each row corresponds to a trial
    story = trial_props[COL_ATTENTION]
    return story

# inverts 2 to 1 and 1 to 2
def invert_attention(story):
    if story == 3: return 3
    if story == 1: return 2
    else: return 1

# stores the information about which column in the props array corresponds to:
# a) what story the subject was paying attention to
# b) what chunk of the story 1 the subject was paying attention to
# c) what chunk of the story 2 the subject was paying attention to
def get_props_constants():
    COL_ATTENTION = 2
    COL_STORY1 = 5
    COL_STORY2 = 6
    return COL_ATTENTION, COL_STORY1, COL_STORY2

# In this function, we input subject number and trial number and output the corresponding linguistic features
# Attention parameter specifies which g\feaatures to get
def get_features(subject, trial_nr, Attention=True):
    COL_ATTENTION, COL_STORY1, COL_STORY2 = get_props_constants()
    props = extract_props(subject)
    trial_props = props[trial_nr-1] # each row corresponds to a trial
    story = trial_props[COL_ATTENTION]

    if story == 3: return None, None
    if not Attention: story = invert_attention(story)

    if story == 1: chunk = trial_props[COL_STORY1]
    else: chunk = trial_props[COL_STORY2]

    features_list = ["Sampled_pos", "Sampled_phoneme", "Sampled_prob"]
    features = []
    for f in features_list:
        features.append(slice_features(story, chunk, f))
    return features, features_list

# we create folders for each subject where the spliced features will be stored
def make_folders(subjects):
    path = f"{os.getcwd()}/features"
    if not os.path.exists(path):
        for subject in subjects:
            N_TRIALS = get_number_of_trials(subject)
            for i in range(1, N_TRIALS+1):
                os.makedirs(f"{os.getcwd()}/features/{subject}/trial{i}", exist_ok=True)
    else: print("The folders already exist")

# save the features for a given subject and trial in a corresponding trial folder
# doesn't save anything if focus was on the music (i.e . story == 3)
def save_trial_features(subject, trial_nr, Attention = True):
    features, feature_names = get_features(subject, trial_nr, Attention)
    if features is not None:
        for i in range(len(features)):
            np.save(f"{os.getcwd()}/features/{subject}/trial{trial_nr}/{feature_names[i]}_Attention_{Attention}", features[i])

# remove empty folders and create the json file with all removed trials
def remove_empty_folders():
    story3 ={}
    root = f"{os.getcwd()}/features"
    dirs = eel_utils.make_participant_string_list(starting_participant=2, ending_participant=19)
    for d in dirs:
        story_trials = []
        dir = os.listdir(f"{root}/{d}")
        for trial in dir:
            if len(os.listdir(f"{root}/{d}/{trial}")) == 0:
                os.rmdir(f"{root}/{d}/{trial}")
                story_trials.append(int(trial.replace("trial", "")))
        story3[d] = story_trials
    utilities.saveAsJSON(story3, "removed_trials")

def add_envelope(participant, trial, story, Attention=True):
    map_story_to_envelope = {1: "V1S_env", 2: "V2S_env"}
    path = f"{os.getcwd()}/4Elizaveta/S{participant}_ENV.mat"
    print(f"envelope is {map_story_to_envelope[story]}")
    mat = {}
    sci.loadmat(path, mat)
    mat = mat[map_story_to_envelope[story]]
    envelope = mat[trial-1]
    envelope_derivatives = np.gradient(envelope)
    np.save(f"{os.getcwd()}/features/S{participant}/trial{trial}/envelope_Attention_{Attention}", envelope)
    np.save(f"{os.getcwd()}/features/S{participant}/trial{trial}/envelope_derivatives_Attention_{Attention}", envelope_derivatives)
    return envelope, envelope_derivatives

def check_for_empty_rows(arr):
    empty_rows = []
    for i in range(len(arr)):
        if np.all(arr[i] == arr[0]):
            print(f"row {i} is empty")
            empty_rows.append(i)
    arr = arr.delete(empty_rows, axis=0)
    return arr, empty_rows

def save_everything(s, t, Attention = True):
    save_trial_features(s, t, Attention)
    story = get_story(s, t)
    if story != 3:
        if not Attention:
            story = invert_attention(story)
        add_envelope(s, t, story, Attention)

def beep():
    winsound.Beep(500, 1000)


if __name__ == "__main__":
    participants = eel_utils.make_participant_string_list(starting_participant=2, ending_participant=19)
    make_folders(participants)
    for s in participants:
        if s not in eel_utils.get_weird_subjects():
           for t in range(1, get_number_of_trials(s)+1):
               save_everything(s, t, Attention = True)
               save_everything(s, t, Attention = False)

    remove_empty_folders()
    beep()
