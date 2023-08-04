#
# This is where POS tags are introduced!!
#
# In this python file I aim to create a JSON file logged on words in Story 1 and 2
# the associating entries are:
# 1) word_start
# 2) word_stop
# 3) surprisal value (from /data/probabilities/Story1/chunks1-76)
# 4) POS and lemma from the language model

import spacy
from spacy.lang.de import German
import utilities
import os

# ---------------------------------------------
#                    SETUP
# ---------------------------------------------
startChunk = 1
chunks = 74
story = 2

# setup spacy to later extract POS tags
nlp = German()
nlp_pos = spacy.load('de_core_news_sm')
tokenizer = nlp.tokenizer
# ---------------------------------------------

allTokens = []
badTokens = ['', '\n', '\xa0']
finalJSON = {}

# for every word in story 1, we record it's timecode from MAU file
# its stored like :
# intervals [2]:
#             xmin = 0.089705
#             xmax = 0.336825
#             text = "Alle"

BASTokensList = []
wordStartList = []
wordEndList = []
posList = {}


# Step 1 - For each chunk, for each sentence extract POS tags and TAG tags, and record them into POS list
for chunk in range(startChunk, chunks + 1):

    # convert docx to txt with UTF-8 encoding
    fileName = f"Part{chunk}_mono_norm"
    rootDir = os.getcwd()
    pathText = rootDir + f"/data/text partition txt/text partition txt Story{story}/"
    utilities.docxToText(pathText + f"story{story}_chunks_docx/" + fileName, pathText + fileName)

    pathTextFull = pathText + fileName + ".txt"

    with open(pathTextFull, "r", encoding="UTF-8") as file:
        text = file.readlines()

    for sentence in text:
        doc = nlp_pos(sentence)
        keys = len(posList.keys()) - 1
        for i in range(0, len(doc)):
            posList_in = {}
            posList_in["word"] = str(doc[i]).lower()
            posList_in["pos"] = doc[i].pos_
            posList_in["tag"] = doc[i].tag_
            posList_in["chunk"] = chunk
            posList[keys + i] = posList_in


    # ---------------------------------------------------------------------
    BASTokensList = utilities.BASfile(rootDir + f"/data/text partition TextGrid/text partition TextGrid Story{story}/Part{chunk}_mono_norm.TextGrid", "words", BASTokensList,
                      wordStartList, wordEndList)

#  -----------------------------

# eure seems to be a very specific case, because pos tagging identifies it as punctuation
new_posList = {k: v for k, v in posList.items() if (v["pos"] != "PUNCT" or v["word"] == 'eure')}
new_posList = {k: v for k, v in new_posList.items() if v["pos"] != "SPACE"}

# extract indexes of all tokens that have 's
for key in new_posList:
    if new_posList[key]["word"] == "’s":
        new_posList[key-1]["word"] = new_posList[key-1]["word"] + "’s" # try immediately adding 's to the previous token

# remove independent 's
new_posList = {k: v for k, v in new_posList.items() if v["word"] != "’s"}

keys = list(new_posList.keys())
final_posList = {}
for i in range(0, len(new_posList.keys())):
    index = keys[i]
    final_posList[i] = new_posList[index]


utilities.saveAsJSON(final_posList, rootDir + rf"\json_matrices\final_posList_story{story}")
utilities.saveAsJSON(BASTokensList, rootDir + f"\json_matrices\BAStokensList_story{story}")


# --------------------------------------------------------------
# Objective: to take POS, TAG and CHUNK from the final_posList
# Iterate through all values in the final posList
# If you found this word somewhere close to where it is supposed to be in BAS file,
# then remove it from the final_posList

def getPOSfromToken(searched_token, token_index):
    for key, value in final_posList.items():
        word = value['word'].lower()
        if (word == searched_token.lower()) and (token_index in range(key-10, key + 10)):
            pos = value['pos']
            tag = value['tag']
            chunk = value['chunk']
            final_posList.pop(key) # remove found entry from finalPosList
            return pos, tag, chunk

    return "not_found_pos", "not_found_tag", "not_found_chunk"

# --------------------------------------------------------------


# get over every word that was identified by the audio transcription  (MAUS system)
# look up in the final_posList json file
for index, token in enumerate(BASTokensList):
    finalJSON_in = {}
    finalJSON_in['word'] = token.lower()
    pos, tag, chunk = getPOSfromToken(token, index)
    if index > len(BASTokensList)-5:
        print(f"pos for token {token} was identified as {pos}")

    finalJSON_in['pos'] = pos
    finalJSON_in['tag'] = tag
    finalJSON_in['chunk'] = chunk
    if type(chunk) != str:
        finalJSON_in['starting_time'] = 60*(chunk - 1) + float(wordStartList[index]) # 60*(chunk - 1) time corrected for continuity between chunks (60 because every recording is exactly  minute)
        finalJSON_in['ending_time'] = 60*(chunk - 1) + float(wordEndList[index])
    finalJSON[index] = finalJSON_in

print(f"There are {len(new_posList.keys())} words in POS")
print(f"There are {len(finalJSON)} words in BAS")


utilities.saveAsJSON(finalJSON, rootDir + f"\json_matrices\pos_tag_and_time_story{story}_mc")

