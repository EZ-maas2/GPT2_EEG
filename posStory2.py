import json

import spacy
from spacy.lang.de import German
import utilities

# setup spacy to later extract POS tags

nlp_pos = spacy.load('de_core_news_sm')
posList = {}
posArray = []
# There might be an issue with how POS tags are extracted for Story 2
pathTextFull = "../EEG_GPT3/data/Story2.txt"




def compareEveryWord():
    with open("D:\lars project\EEG_GPT3\json_matrices\BAStokensList_story2.json", "r", encoding="UTF-8") as file:
        BAS = file.readlines()
    with open(r"D:\lars project\EEG_GPT3\json_matrices\final_posList_story2.json",  "r", encoding="UTF-8") as f:
        posList = json.loads(f.read())
    for token in posList:
        word = posList[token]['word'].lower()
        word = word.lower()
        posArray.append(word)


    new_BAS = []
    for value in BAS:
        value = value.replace('"', '')
        value = value.strip()
        value = value.strip(",\n")
        value = value.lower()
        if value not in ['[', ']']:
            new_BAS.append(value)
    new_posArray = []
    for entry in posArray:
        value = value.strip()
        value = value.strip(",\n")
        if entry not in ['[', ']', '.', '!', '-', '–', ',', '?', '\n', ':', '(', ")", '"',"'", "...",'…']:
            new_posArray.append(entry)

    print(f"length pos is {len(new_posArray)}; bas is {len(new_BAS)}")

    for index in range(0, min(len(new_BAS), len(new_posArray))):
        if new_posArray[index] != new_BAS[index]:
            print(f"pos word is _{new_posArray[index]}_, index is {index}; bas word is _{new_BAS[index]}_")

compareEveryWord()





# ALL THAT SHOULD BE DONE IN MATRIX CREATION------------------------------------------------------------
# with open(pathTextFull, "r", encoding="UTF-8") as file:
#     text = file.readlines()
#
# doc_special = nlp_pos("ging")
#
# for sentence in text:
#     doc = nlp_pos(sentence)
#     keys = len(posList.keys())
#     for i in range(0, len(doc)):
#         posList_in = {}
#         posList_in["word"] = str(doc[i]).lower()
#         posList_in["pos"] = doc[i].pos_
#         posList_in["tag"] = doc[i].tag_
#
#         posList[keys + i] = posList_in
#         posArray.append(str(doc[i]).lower())
# new_posList = {k: v for k, v in posList.items() if v["pos"] != "PUNCT" and v["pos"] != "SPACE"}
# utilities.saveAsJSON(new_posList,"..\posStory2_separateFromMatrixCreation")

# ---------------------------------------------------------------------