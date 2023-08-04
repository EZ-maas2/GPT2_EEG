# Elizaveta Zavialova
# here we create framework for Temporal response function (TRF) model training with eelbrain package
import os

import eelbrain as eel
import numpy as np
import winsound

np.set_printoptions(threshold=np.inf)
import pandas as pd
import scipy.io as sci
import utilities
import pickle
#
# # ------------------------------
# # CONSTANTS
NUM_TIMEPOINTS = 6000
NUM_DATA_PER_FOLDER = 50
NUM_TRIALS = 75

# --------- DIMENSION OBJECTS ---------------
# create the sensor descriptor object
# for two specified channels

# time descriptor is initialized as UTS (uniform time series)
Time = eel.UTS(0, 0.01, NUM_TIMEPOINTS)  # start, step, num samples
POS = eel.Categorial("POS", ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB", "ADP", "AUX", "CCONJ", "DET", "NUM", "PART",
                             "PRON", "SCONJ",	"SYM",	"PUNCT",	"X"]) # 15, 17 for s1; 15 for s2
# I cant remove zero rows, because then the data will not be aligned
# I will replace zero with a very small number
PHONEME = eel.Categorial("PHONEME",
                         ['2:', '6', '9', '<p:>', '<usb>', '?', '@', 'C', 'E', 'E:', 'I', 'N', 'O', 'OY', 'S', 'U', 'Y',
                          'Z', 'a', 'a:', 'aI', 'aU', 'b', 'd', 'dZ', 'e', 'e:', 'f', 'g', 'h', 'i', 'i:', 'j', 'k',
                          'l', 'm', 'n', 'o', 'o:', 'p', 'r', 's', 't', 'tS', 'ts', 'u', 'u:', 'v', 'x', 'y', 'y:',
                          'z'])

# 1: Function to load all the MATL
# AB features data from the feature folder
# REASON TO CHANGE THIS FUNCTION WOULD BE IF THERE ARE MORE FILES IN THE FEATURE FOLDER

def load_features(path, Attention=True):
    pos = np.load(f"{path}/Sampled_pos_Attention_{Attention}.npy")
    prob = np.load(f"{path}/Sampled_prob_Attention_{Attention}.npy")
    phoneme = np.load(f"{path}/Sampled_phoneme_Attention_{Attention}.npy")
    envelope = np.load(f"{path}/envelope_Attention_{Attention}.npy")
    envelope_der = np.load(f"{path}/envelope_derivatives_Attention_{Attention}.npy")

    return pos, prob, phoneme, envelope, envelope_der



def modelExists(parameters, subject):
    return os.path.exists(f"{os.getcwd()}/model_data/pickled/model_for_S{subject}_{make_parameters_string(parameters)}.pkl")


# Envelope doe snot require cropping because on the feature to folder assignment level we already cropped it
def load_all_trials_features(subject, Attention=True):
    pos_arr, prob_arr, phoneme_arr, envelope_arr, envelope_der_arr = make_empty_arrays()
    pos_onset_arr = pos_arr.copy()
    trial_counter = 0
    for t in range(1, NUM_TRIALS+1):
        path = f"{os.getcwd()}/features/{subject}/trial{t}"
        if os.path.exists(path):
            pos, prob, phoneme, envelope, envelope_der = load_features(path, Attention)
            pos_arr[trial_counter, :, :] = pos
            prob_arr[trial_counter, :] = prob
            phoneme_arr[trial_counter, :, :] = phoneme
            envelope_arr[trial_counter, :] = envelope
            envelope_der_arr[trial_counter, :] = envelope_der
            pos_onset_arr[trial_counter, :] = calculate_onset(pos)
            trial_counter += 1
    return pos_arr, prob_arr, phoneme_arr, envelope_arr, envelope_der_arr, pos_onset_arr


# 1B Load EEG from a different location
# SHAPE OF EEG IS (TRIAL, CHANNEL, TIME)
def load_eeg(subject):
    if type(subject) == int:
        subject = f"S{subject}"
    path = f"{os.getcwd()}/4Elizaveta/{subject}_EEG.mat"
    mat = {}
    sci.loadmat(path, mat, variable_names=["EEG0"])
    mat = np.array(mat["EEG0"])
    return mat

# in this function we load the eeg data from the .mat file and select the relevant channels
def prepare_eeg(subject, channels = False):
    new_eeg = load_eeg(subject)
    if channels:
        labels = get_Sensor_labels()
        channels_indeces = [i for i, name in enumerate(labels) if name in channels]
        # here we have to get the indices of the channels from the strings

        new_eeg = eeg_relevant_channels(channels_indeces, new_eeg)
    new_eeg = remove_irrelevant_trials(new_eeg)
    return new_eeg

# takes in dictionary with model results and appends  it to a json file
def update_model_log(information):
    print("Updating model log...")
    path = f"{os.getcwd()}/model_data"
    utilities.make_folder(path)
    if os.path.isfile(f"{path}/model_log.json"):
        model_log = utilities.openJSON(f"{path}/model_log.json")
        model_log[len(model_log) + 1] = information
    else:
        model_log = {1: information}
    utilities.saveAsJSON(model_log, f"{path}/model_log")
    print("Model log updated!")

# 2: Function to convert the MATLAB data to NDVAR
# takes in the features and EEG data and outputs the NDVAR object

# pos has  3 dimensions: 1) TRIAL, 2) POS (Categorical), 3) time
# phoneme has  3 dimensions: 1) TRIAL, 2) PHONEME (Categorical), 3) time
# prob has 2 dimensions: 1) TRIAL, 2) time

# outputted EEG should have 3 different dimensions: 1) trial, 2)channel, 3)time

# S2 and S5 are weird because they have 60 trials total
def get_weird_subjects():
    return ["S2", "S5"]

def get_r(res):
    return res.r

def get_r2(res):
    return res.proportion_explained

def get_trf(res, scaled = False):
    if scaled:
        return res.h_scaled
    return res.h

def tanh_for_averaging(value):
    tanh = np.tanh(value)
    return tanh


def select_Sensors(names = False):
    Sensors = make_Sensors()
    if names:
        idx = {name:i for i, name in enumerate(Sensors.names) if name in names}
        names = idx.keys()
        idx = idx.values()
        locs = [Sensors.locs[i] for i in idx]
        Sensors = eel.Sensor(locs, names)
    return Sensors


def make_training_data(participant, channels = False, Attention = True):
    Sensors = select_Sensors(channels)
    Cases = eel.Case(NUM_DATA_PER_FOLDER)
    eeg = prepare_eeg(participant, channels)
    pos, prob, phoneme, envelope, envelope_der, pos_onset = load_all_trials_features(participant, Attention)

    pos = replace_flat(pos)
    pos_onset = replace_flat(pos_onset)

    pos_ndvar = eel.NDVar(pos, dims = (Cases, POS, Time), name = f"POS_Attention_{Attention}")
    phoneme_ndvar = eel.NDVar(phoneme, dims = (Cases, PHONEME, Time), name = f"PHONEME_Attention_{Attention}")
    prob_ndvar = eel.NDVar(prob, dims = (Cases, Time), name = f"PROB_Attention_{Attention}")
    envelope_ndvar = eel.NDVar(envelope, dims = (Cases, Time), name = f"ENVELOPE_Attention_{Attention}")
    eeg_ndvar = eel.NDVar(eeg, dims = (Cases, Sensors, Time), name = f"EEG_Attention_{Attention}")
    envelope_der_ndvar = eel.NDVar(envelope_der, dims = (Cases, Time), name = f"ENVELOPE_DER_Attention_{Attention}")
    pos_onset_ndvar = eel.NDVar(pos_onset, dims = (Cases, POS, Time), name = f"POS_ONSET_Attention_{Attention}")
    return [pos_ndvar, phoneme_ndvar, prob_ndvar, envelope_ndvar, envelope_der_ndvar, pos_onset_ndvar, eeg_ndvar]

# 3: Function to create an array of all requested NDVAR features


# --------------------UTILITY FUNCTIONS------------------------------------------------
def make_Sensors():
    locs = get_Sensor_locs()
    labels = get_Sensor_labels()
    return eel.Sensor(locs = locs, names=list(labels))

def load_Sensors():
    return pd.read_excel(f"{os.getcwd()}/4Elizaveta/channels.xlsx", header=None).to_numpy()
def get_Sensor_labels():
    mat = load_Sensors()
    for ind, label in enumerate(mat[:, 0]):
        mat[ind, 0] = label.strip("\'")
    labels = mat[:, 0]
    return labels

def get_Sensor_locs():
    mat = load_Sensors()
    locs = [(mat[index, 1], mat[index, 2], mat[index, 3]) for index in range(len(mat[:, 0]))]
    return locs


def calculate_onset(timeseries):
    new_timeseries = np.zeros((timeseries.shape))
    new_timeseries[new_timeseries == 0] = np.nan
    for rix, row in enumerate(timeseries):
        shifted_row = row.copy()
        shifted_row = np.roll(shifted_row, 1)
        shifted_row[0] = 0.0
        new_row = np.subtract(row, shifted_row)

        new_row[new_row == -1] = 0
        # for some reason, 0 -1 gives 255 instead of -1
        new_row[new_row == 255] = 0

        new_timeseries[rix, :] = new_row
    return timeseries

def show_data_partitions(partitions, participant):
    plot = eel.plot.preview_partitions(NUM_DATA_PER_FOLDER, partitions, test=True)
    plot.save(f"{os.getcwd()}/model_data/data_partitions_participant{participant}.png")


# in this function we reduce the EEG data to the relevant channels
# takes an array of channel indeces and full EEG array as input, and outputs the reduced EEG array
def eeg_relevant_channels(channels_list, eeg):
    new_eeg = np.zeros([NUM_TRIALS, len(channels_list), NUM_TIMEPOINTS])
    for ind, channel in enumerate(channels_list):
        new_eeg[:, ind, :] = eeg[:, channel, :]
    return new_eeg

def remove_irrelevant_trials(arr):
    removed_trials = [11, 12, 13, 14,15, 26, 27,28,29, 30, 41, 42, 43, 44, 45, 56, 57, 58, 59, 60, 71, 72, 73, 74, 75] # REMOVED TRIALS ARE ALL THE SAME APART FROM S2 AND S5
    removed_trials = [i - 1 for i in removed_trials]
    arr = np.delete(arr, removed_trials, axis=0)  # delete trials that are not relevant
    return arr


def make_empty_arrays():
    pos_array = np.empty((NUM_DATA_PER_FOLDER, len(POS), NUM_TIMEPOINTS))
    prob_array = np.empty((NUM_DATA_PER_FOLDER, NUM_TIMEPOINTS))
    phoneme_array = np.empty((NUM_DATA_PER_FOLDER, 52, NUM_TIMEPOINTS))
    envelope_array = np.empty((NUM_DATA_PER_FOLDER, NUM_TIMEPOINTS))
    envelope_der_array = np.empty((NUM_DATA_PER_FOLDER, NUM_TIMEPOINTS))
    return pos_array, prob_array, phoneme_array, envelope_array, envelope_der_array


def get_base_plus_sequence_of_features(sequence):
    base = ['Envelope', 'Envelope_Derivative']
    standard = ['POS','Phoneme','Probability','Envelope', 'Envelope_Derivative']
    leftover = []
    output = []
    output.append(str(base))
    for s in sequence:
        leftover = [x for x in standard if x in base or x in s or x in leftover]
        output.append(str(leftover))
    return output


def get_base_plus_individual_features(sequence):
    base = ['Envelope', 'Envelope_Derivative']
    standard = ['POS', 'Phoneme', 'Probability', 'Envelope', 'Envelope_Derivative']
    leftover = []
    output = []
    output.append(str(base))
    for s in sequence:
        leftover = [x for x in standard if x in base or x in s]
        output.append(str(leftover))
    return output

def get_individual_and_sequential_models(sequence):
    params = get_base_plus_individual_features(sequence)
    params2 = get_base_plus_sequence_of_features(sequence)
    return params+params2

def get_all_channels():
    all_channels = ['FP1', 'FP2', 'AF7', 'AFz', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7',
                    'FC3', 'FCz', 'FC4', 'FT8', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3',
                    'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'POz', 'PO8',
                    'O1', 'O2', 'FPz', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                    'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2',
                    'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'Oz', 'TP10',
                    'EOGhorizontal', 'EOGvertical', 'TP9']
    return all_channels

# -----------------------------------------------
def replace_flat(arr):
    for t in range(len(arr)):
        for i in range(len(arr[t])):
            if arr[t][i].all() == 0:
                arr[t][i] = 0.00000000000000001
    return arr


def make_participant_string_list(starting_participant = 3, ending_participant= 19, exclude = [2, 5]):
   return [f"S{i}" for i in range(starting_participant, ending_participant) if i not in exclude]


def make_parameters_string(parameters):
    if not type(parameters)==str:
        parameters = [p for p in parameters if parameters[p]]
    return parameters


def unpickle_boosting_result(participant, parameters):
    if type(parameters) == dict:
        parameters = make_parameters_string(parameters)
    with open(f"{os.getcwd()}/model_data/pickled/model_for_{participant}_{parameters}.pkl", "rb") as f:
        boostingResult = pickle.load(f)
    return boostingResult

def beep():
    winsound.Beep(500, 1000)