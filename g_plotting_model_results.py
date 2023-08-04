import os

import eelbrain as eel
import numpy as np
import scipy.io

from eelbrain_utilities import *
import matplotlib.pyplot as plt
import pandas as pd


def make_topomap(participant, parameters, whatIsPlotted_f, whatIsPlotted):
    paramstring = make_parameters_string(parameters)
    res = unpickle_boosting_result(participant, parameters)
    sensornames = res.r.sensor.names
    bad_channels = ['EOGhorizontal','EOGvertical']
    r = whatIsPlotted_f(res)
    if bad_channels[0] or bad_channels[1] in sensornames:
        mask  = [s in bad_channels for s in sensornames]
        r = r.mask(mask)
        sensornames = r.sensor.names

    print(f"{len(sensornames)}|{sensornames}")

    fig = eel.plot.Topomap(r, show = True, clip = 'circle', head_pos=-0,sensorlabels='name')
    fig.save(f"TopoMaps/Topomap_{participant}_{paramstring}_{whatIsPlotted}.png")



# the idea is that we are making a dataframe containing the parameters from a list of models
# ONLY CAN BE DONE FOR ONE PARTICIPANT NOW
# MODELS ARE REPRESENTED AS LIST OF PARAMETER STRINGS
def track_values_same_participant_different_models(participant, models, what_RESULT_to_track_func,
                                                   channels = get_all_channels(), model_names = None):
    # 2D array where each row is a channel and each column is a model
    results = np.empty((len(channels), len(models)))
    for col, model in enumerate(models):
        res = unpickle_boosting_result(participant, model)
        result = what_RESULT_to_track_func(res)
        for row, channel in enumerate(channels):
            results[row][col] = result.sub(sensor = channel)

    df = pd.DataFrame(results, index = channels)
    if model_names:
        df.columns = model_names
    return df

def track_values_same_model_different_participants(participants, parameters, what_RESULT_to_track_func,
                                                   channels = get_all_channels(), participant_names = None):
    # 2D array where each row is a channel and each column is a participant
    results = np.empty((len(channels), len(participants)))
    for col, participant in enumerate(participants):
        res = unpickle_boosting_result(participant, parameters)
        result = what_RESULT_to_track_func(res)
        for row, channel in enumerate(channels):
            results[row][col] = result.sub(sensor = channel)

    df = pd.DataFrame(results, index = channels)
    if participant_names:
        df.columns = participant_names
    return df


def average_across_participants_one_channel(participants, parameters, channel, what_to_plot_func, saveData = True):
    values = []
    for participant in participants:
        res = unpickle_boosting_result(participant, parameters)
        value = what_to_plot_func(res)
        values.append(value.sub(sensor = channel))
    averages = tanh_for_averaging(values)
    if saveData:
        averages_dict = {"values": values, "averages_tanh": averages, "participants": participants, "parameters": parameters}
        scipy.io.savemat(f"Boxplots/Matlab_data/{str(participants)}_{parameters}_{channel}_{what_to_plot_func.__name__}_averages.mat", averages_dict)
    return averages

# for a given model type, for a given channel, extract r-values or r-squared values,
# tanh them, and plot them as a box plot
def boxplot_average(participants, parameters, channels, what_to_plot_func, saveData = True):

    function_name = what_to_plot_func.__name__.strip('_get')
    parameterstring = make_parameters_string(parameters)
    image_name = f"Boxplot_Average_{parameterstring}_{channels}_{function_name}"
    boxplots_directory = f"{os.getcwd()}/Boxplots/"

    fig = plt.figure(figsize =(20, 10))

    for ix, channel in enumerate(channels):
        averages = average_across_participants_one_channel(participants, parameters, channel, what_to_plot_func, saveData)
        plt.boxplot(averages, autorange=True, positions=[ix])
        for i,a in enumerate(averages):
            plt.plot(ix, a, marker = 'o')

    plt.title(f"Average for model {parameterstring.strip('[').strip(']')}")
    if len(channels)>1:
        plt.xticks(range(0,len(channels)), channels) # replace numbers with channel names
    else: plt.xticks([1], channels)

    plt.ylabel(function_name)
    if not os.path.exists(boxplots_directory):
        os.mkdir(boxplots_directory)
    plt.savefig(f"{boxplots_directory}/{image_name}")




# take in a dataframe where each row is a channel and each column is a model
def make_plot(df, subject, name, xlabel, models_or_participants, sensors = None):
    if sensors:
        for s in sensors:
            sen = df.loc[s].array
            fig = plt.figure(figsize=(10, 5))
            plt.plot(models_or_participants, sen, label = s)
            plt.plot(models_or_participants, sen, 'oy')
            plt.title(f"{name} values, {subject}, {s}")
            plt.xlabel(xlabel)
            plt.ylabel(f"{name}")
            plt.xticks(rotation=-5)
            plt.tight_layout()

            if not os.path.exists(f"{name}_values/"):
                os.mkdir(f"{name}_values/")
                print("made directory")
            plt.savefig(f"{name}_values/{subject}_{s}.png")


if __name__ == "__main__":
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

    channels = ["Fz", "FCz", "Cz", "FC1", "FC2", "FC3","FC4", "F1", "F2", "C1", "C2", "CPz", "Pz", "POz", "P1", "P2", "FT7",
                "FT8", "TP7", "TP8", "T7", "T8"]

    participants = make_participant_string_list(starting_participant=3, ending_participant=18)
    params = get_individual_and_sequential_models(["Phoneme", "POS", "Probability"])

    for p in params:
        boxplot_average(participants, p, get_all_channels(), get_r2)
        boxplot_average(participants,p, get_all_channels(), get_r)