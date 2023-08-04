import json
import os
from utilities import BASfile, saveAsJSON
import numpy as np
from scipy.io import savemat



def main(story, chunks):
    rootDir = os.getcwd()
    dir = rootDir + f"/data/text partition TextGrid/text partition TextGrid Story{story}/"
    # GET THE ONSET FOR EVERY PHONEME IN CHUNK 1 STORY 1
    phonemes_bas = []
    start_times = []

    # STEP 1: EXTRACT ALL INFORMATION FROM THE "POS AND TIME AND PROBS" JSON FILE (probs, pos)

    # EXTRACT PHONEMES:
    for chunk in range(0, chunks):
        full_dir = dir + f"Part{chunk+1}_mono_norm.TextGrid"
        new_start_times = []
        BASfile(full_dir, "phonemes", phonemes_bas, new_start_times, [])
        new_start_times = [value+60*chunk for value in new_start_times]
        start_times = start_times+new_start_times
    # APPEND PHONEMES WITH TIMES
    phonemes = []
    for i in range(len(phonemes_bas)):
        if (i+1) < len(phonemes_bas): inner_phoneme = [phonemes_bas[i], [start_times[i], start_times[i+1]]]
        else: inner_phoneme = [phonemes_bas[i], [start_times[i], 60 * chunks]]
        phonemes.append(inner_phoneme)



    # EXTRACT POS AND PROBABILITIES
    pos_tags = []
    probs = []
    with open(f"{rootDir}/json_matrices/final_results_data/data_story{story}.json") as file:
        f = json.load(file)
        for entry in f:
            inner_pos_tag = {}
            inner_probs = {}
            if int(f[entry]['chunk']) < chunks + 1:
                inner_pos_tag = [f[entry]["pos"], [f[entry]["starting_time"], f[entry]["ending_time"]]]
                inner_probs = [f[entry]['log_probabilities'], [f[entry]["starting_time"], f[entry]["ending_time"]]]
                pos_tags.append(inner_pos_tag)
                probs.append(inner_probs)
    #-----------------------------------------
    # THIS FUNCTION TAKES A LIST OF THE FORMAT [[<SAMPLE1>, [START_TIME, END_TIME]], [<SAMPLE2>, [START_TIME, END_TIME]]]
    # IT THEN CHECKS FOR TIME INCONTIGUITY AND INSERTS ['X']

    def buffer_with_pauses(bufferable):
        indeces = []
        for index in range(0, len(bufferable)-1):
            end_time_this = bufferable[index][1][1]
            start_time_next = bufferable[index+1][1][0]
            if end_time_this != start_time_next: # A PAUSE
                indeces.append([index, end_time_this, start_time_next]) # NOTE DOWN WHERE INSERTIONS MUST BE MAADE
        for scaling, i in enumerate(indeces): bufferable.insert(i[0]+1+ scaling, ['X', [i[1], i[2]]])
        if bufferable[len(bufferable)-1][1][1] < chunks* 60:
            bufferable.append(['X', [bufferable[len(bufferable)-1][1][1], chunks*60]])

    buffer_with_pauses(pos_tags)
    buffer_with_pauses(probs)
    #------------------------------------------

    # current sample has a format ['<sample>',[t1, t2]]
    def check_range(t, sample_index, list_samplable, tag):
        sample = list_samplable[sample_index][0]
        sample_time = list_samplable[sample_index][1]
        # TIME IS WITHIN THE SAMPLE'S RANGE, TAKE THAT SAMPLE
        if sample_time[0] <= t < sample_time[1]:
            return sample, sample_index
        else:
            sample_index = sample_index + 1
            sample, sample_index = check_range(t, sample_index, list_samplable, tag)
            return sample, sample_index


    #---------------------------------------------
    # here OPTIONS IS THE MAPPING FROM POS_TAGS ETC TO 1s AND 0s
    # list_samplable is has a format [[<p:>, [0, 1], [<p:>, [1, 2.3333], ..]]

    def make_sampled_matrix(OPTIONS, tag, list_samplable):
        if tag == 'pos': n = 17
        elif tag == 'phoneme': n = 52
        else : n = 1
        sampling_list = OPTIONS.copy()
        sampling_list_big = np.transpose(np.array([np.zeros(n, int)])) # the first zero column
        sample_index = 0
        sampling_list_big_probs = []

        # FOR EACH TIME STEP WE CHECK WHETHER THE TIME IS WITHIN THE TIME RANGE
        for i, t in enumerate(time):
            print(f"time {t} out of {chunks*60}")
            sample, sample_index = check_range(t, sample_index, list_samplable, tag)
            print(f"sample is {sample}")
            # ONE-HOT ENCODING IS ONLY PERFORMED FOR PHONEMES AND POS
            if not tag == 'prob':
                print(OPTIONS)
                for index, value in enumerate(OPTIONS):
                    if value == sample: sampling_list[index] = 1
                    else: sampling_list[index] = 0
                print(sampling_list)
                sampling_list = np.transpose(np.array(sampling_list))
                print(sampling_list_big)
                print(sampling_list)
                sampling_list_big = np.column_stack((sampling_list_big, sampling_list))
            else:
                sampling_list_big_probs.append(sample)
                sampling_list_big = sampling_list_big_probs
                sampling_list_big = [value if value != "X" else 0 for value in sampling_list_big]
        print(sampling_list_big)
        samp = {"Sampled": sampling_list_big}
        savemat(f"sampling_{tag}_story{story}.mat", samp)

    #--------------
    # MAP PHONEMES TO INTEGERS
    PHONEMES = ['2:', '6', '9', '<p:>', '<usb>', '?', '@', 'C', 'E', 'E:', 'I', 'N', 'O', 'OY', 'S', 'U',
         'Y', 'Z', 'a', 'a:', 'aI', 'aU', 'b', 'd', 'dZ', 'e', 'e:', 'f', 'g', 'h', 'i', 'i:', 'j',
         'k', 'l', 'm', 'n', 'o', 'o:', 'p', 'r', 's', 't', 'tS', 'ts', 'u', 'u:', 'v', 'x', 'y',
         'y:', 'z']

    POS_LIST = ["ADJ", "ADV","INTJ", "NOUN","PROPN", "VERB",
                "ADP", "AUX", "CCONJ","DET", "NUM", "PART", "PRON", "SCONJ",
                "SYM", "PUNCT", "X"]

    #--------------
    # IF WE ARE ON THE EDGE OF TWO PHONEMES - TAKE BOTH
    # SAME FOR GPT2 (EXCLUSIVE FOR FLOAT VECTOR)
    # POS SAMPLING

    # In this part, we make a True/False encoded matrix for POS and phonemes and float encoded vector for probs


    sampling_fr = 100 #100Hz!!
    time = np.arange(0.5, chunks*60, 1/sampling_fr, dtype = float) # sampling frequency = 1 second

    make_sampled_matrix([], "prob", probs)
    make_sampled_matrix(PHONEMES.copy(), "phoneme", phonemes)
    make_sampled_matrix(POS_LIST.copy(), "pos", pos_tags)

main(story=1, chunks=76)
main(story = 2, chunks=74)