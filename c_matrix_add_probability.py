# This script is meant to align the existing probability data (produced by a_gpt2Surprisal.py) and POS and time matrix, produced by
# BAS system
import json
import utilities
import os
rootDir = os.getcwd()

story = 2
start_chunk = 1
stop_chunk = 74

json_time_pos = rootDir + f"/json_matrices/pos_tag_and_time_story{story}_mc.json"
json_probability = rootDir + f"/data/probabilities/Story{story}/chunks{start_chunk}-{stop_chunk}.json"
bad_tokens =  ["-", ".", ",", "!", "!!!", "?", "\"", "\".", "\"!", "\"?", '…', '(', ')', '–', ':', '"',  '),', '(,', '�', "›", "–,", "�", "’", ";"]

f = open(json_time_pos, encoding="UTF-8")
json_time_pos = json.load(f)

f2 = open(json_probability, encoding="UTF-8")
json_probability = json.load(f2)


# - Save all probabilities in a separate file
allProbabilities = {}
global_index = 0

# extract every sentence
for index in range(len(json_probability)):
    # extracts probability for each sentence
    probabilities = json_probability[str(index)]['log_probabilities_context_0']
    prob5 = json_probability[str(index)]['log_probabilities_context_5']
    # every word within sentence
    for index5, word in enumerate(probabilities): # word here represents a token index as well
        inner_prob = {}
        inner_prob['word'] = word[0]
        inner_prob['prob'] = word[1]
        if True:
            inner_prob['prob5'] = prob5[index5]
        allProbabilities[global_index] = inner_prob
        global_index += 1


# This file is important for multProb function
utilities.saveAsJSON(allProbabilities, rootDir + f'/json_matrices/allProbabilities_b4multProb_story{story}')

# Modified ALLPROBABILITIES MATRIX SUCH THAT IT CONTAINS THE RECALCULATED PROBABILITIES
allProbabilities = utilities.modifyAllProbabilities(story)
utilities.saveAsJSON(allProbabilities, rootDir + f'/json_matrices/allProbabilities_story{story}')
# ALL PROBABILITIES NOW HAVE PROB5

allowed_index = [key for key, value in allProbabilities.items()]
# allowed indexes are created elsewhere but modified here


def attemptProb(token):
    probability = "not_found"
    prob5 = None
    for index in allowed_index:
        item = allProbabilities[index]
        if item['word'] == token:
            allowed_index.remove(index)
            probability = item['prob']
            prob5 = item['prob5']
            return probability, prob5

    return probability, prob5



allowed_index = [key for key, value in allProbabilities.items()]
utilities.saveAsJSON(allowed_index, rootDir + f'/json_matrices/TEST_allowedIndex_story{story}')
for index in range(len(json_time_pos)):
    token = json_time_pos[str(index)]['word']
    # ---------- words with - are jammed together
    if "-" in token:
        tokenList = token.split("-")
        token = ""
        for i in tokenList:
            token += i
    # ----------
    attemptedProbability, prob5 = attemptProb(token)
    if attemptedProbability == 'not_found':
        print(f"no match for token {token}")
    json_time_pos[str(index)]['log_probabilities'] = attemptedProbability
    json_time_pos[str(index)]['log_probabilities_context_5'] = prob5

# HERE MAIN RESULT IS SAVED !!!
utilities.saveAsJSON(json_time_pos, rootDir + f'/json_matrices/pos_and_time_and_probs_story{story}')




leftovers = {}
for index in allowed_index:
    if allProbabilities[index]['word'] not in bad_tokens:
        leftovers[index] = allProbabilities[index]

if len(leftovers) != 0:
    print(f"just {len(leftovers)} more to go C:")
    utilities.saveAsJSON(leftovers, rootDir + f'/json_matrices/allProbabilities_leftover_story{story}')

