# Elizaveta Zavialova
# here we create framework for boosting model training with eelbrain package
# takes in an array of participants that will be used for training data

import os

import eelbrain as eel
from eelbrain_utilities import *
from timeit import default_timer as timer
import datetime
import pickle
import json
import pandas as pd


# where parameters is a dictionary of parameters of the iormat "POS": True
# and params is an array of variables corresponding to the parameteres
def remove_indeces(params, parameters):
    indeces_to_remove = []
    for index, p in enumerate(parameters):
        if not parameters[p]:
            indeces_to_remove.append(index)
    return [p for i, p in enumerate(params) if i not in indeces_to_remove]


def get_only_relevant_NDVARS(participant, parameters, channels):
    params = make_training_data(participant, channels)
    params = remove_indeces(params, parameters)
    expected_eeg = params.pop()  # returns last and eeg is last in the list
    return params, expected_eeg

def make_model(participant, parameters_first_speaker, parameters_second_speaker = False, channels = False, show_partitions = False):
    params, expected_eeg = get_only_relevant_NDVARS(participant, parameters_first_speaker, channels)
    if parameters_second_speaker:
        print("Adding parameters from a secondary speaker")
        params_2 = get_only_relevant_NDVARS(participant, parameters_second_speaker, channels)[0]
        params = params + params_2

    print("Fetched data. Making the model...")
    expected_eeg = eel.filter_data(expected_eeg, low = 0.5, high = 8)
    res = eel.boosting(expected_eeg, params, basis = 0.1, tstart = -0.05, tstop = 0.60, partitions= 5,
                       scale_data = True, test = True, error = 'l2')
    return res


# This function takes in a boosting result and model training information and saves it to a log
def create_model_log_entry(result, participant, parameters, parameters_second_speaker = False):
    print("Making an entry for the model log...")
    time = result.t_run
    r2 = result.proportion_explained
    r = result.r

    params = make_parameters_string(parameters)

    if parameters_second_speaker:
        params_2 = make_parameters_string(parameters_second_speaker)
        params = params + params_2

    information = {
                      "runtime, seconds:": time,  "participant": participant,
                      "params": params}

    for s in r2.sensor:
        information[f"r2_{s}"] = r2.sub(sensor = s)
        information[f"r_{s}"] = r.sub(sensor = s)

    update_model_log(information)


def save_model(result, participant, parameters):
    params = make_parameters_string(parameters)
    with open(f"{os.getcwd()}/model_data/pickled/model_for_{participant}_{params}.pkl", "wb") as f:
        pickle.dump(result, f)


def make_model_dataframe():
    with open(f"{os.getcwd()}/model_data/model_log.json", "r") as f:
        model_log = json.load(f)
    df = pd.DataFrame(model_log)
    df.to_csv(f"{os.getcwd()}/model_data/model_log.csv")

# can specify the EEG channels as a list of strings (i.e ["FCz", "Cz"])
def run_for_all(parameters, parameters_second_speaker = False, channels = False):
    participants = [f"S{i}" for i in range(2, 19) if i not in [2, 5]]
    for participant in participants:
        run_for_one(participant, parameters, parameters_second_speaker, channels)
        print(f"{participant}| {make_parameters_string(parameters)}| Done running at {datetime.datetime.now().strftime('%H:%M:%S')}")


def run_for_one(participant, parameters, parameters_second_speaker = False, channels = False):
    boosting_result = make_model(participant, parameters, parameters_second_speaker, channels)
    create_model_log_entry(boosting_result, participant, parameters, parameters_second_speaker)
    save_model(boosting_result, participant, parameters)
    make_model_dataframe()





if __name__ == "__main__":
    # SELECT WHAT CHANNELS TO USE HERE
    all_channels = get_all_channels()
    cluster_channels = ["Fz", "FCz", "Cz", "FC1", "FC2", "FC3","FC4", "F1", "F2", "C1", "C2", "CPz", "Pz", "POz", "P1", "P2", "FT7",
                "FT8", "FT9", "FT10", "TP7", "TP8", "T7", "T8"]

    CHANNELS = all_channels
    # --------------------------------
    # SELECT WHAT PARAMETERS TO USE HERE
    parameters = {"POS": False,
                  "Phoneme": False,
                  "Probability": False,
                  "Envelope": True,
                  "Envelope_Derivative": True,
                  "POS_Onset": False}

    parameters_second_speaker = {"POS": False,
                  "Phoneme": False,
                  "Probability": False,
                  "Envelope": False,
                  "Envelope_Derivative": False,
                  "POS_Onset": False}

    # --------------------------------
    print(f"Started at {datetime.datetime.now().strftime('%H:%M:%S')}")
    t_start = timer()

    # RUN THE MODEL HERE
    # YOUR OPTIONS INCLUDE RUN_FOR_ALL AND RUN_FOR_ONE
    # I.E run_for_one("S2", parameters, parameters_second_speaker, channels = all_channels)
    # OR run_for_all(parameters, parameters_second_speaker, channels = all_channels)
    # CHANNELS DEFAULT TO ALL_CHANNELS, BUT YOU CAN SPECIFY THEM AS A LIST OF STRINGS
    # PARAMETERS_SECOND_SPEAKER CAN BE OMITTED IF YOU DON'T WANT TO USE IT

    parameters["Probability"] = True
    run_for_all(parameters, channels = all_channels)


    beep()
    print(f"Done! Ran for {timer() - t_start} seconds")

