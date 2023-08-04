"""
Objective: to extract semantic predictors from the gpt2 model on audiobook input


Minicons probability extraction code: https://github.com/huggingface/transformers/issues/2648
German GPT2: https://huggingface.co/dbmdz/german-gpt2
Sentence separation: https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences

Paper on minicones: https://kanishka.xyz/post/minicons-running-large-scale-behavioral-analyses-on-transformer-lms/

@article{misra2022minicons,
    title={minicons: Enabling Flexible Behavioral and Representational Analyses of Transformer Language Models},
    author={Kanishka Misra},
    journal={arXiv preprint arXiv:2203.13112},
    year={2022}
}

Large German GPT2 Model:
@misc{Minixhofer_GerPT2_German_large_2020,
author = {Minixhofer, Benjamin},
doi = {10.5281/zenodo.5509984},
month = {12},
title = {{GerPT2: German large and small versions of GPT2}},
url = {https://github.com/bminixhofer/gerpt2},
year = {2020}
}
"""


import os.path
import spacy
from minicons import scorer
import utilities
from transformers import pipeline
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------
'''
Parse Story to get surprisal for each sentence
'''
# Step 1:
# Set up the input and the model

# Specify which story is to be analyzed and how many subsections does it have (Story 1 - 76, Story 2 - 74)
story = 2
chunks = 74
startChunk = 1
rootDir = os.getcwd()
pathText = rootDir + f"/data/text partition txt/text partition txt Story{story}/"
probabilitiesPath = rootDir + f"/data/probabilities/Story{story}"

# -------------------------------------------
# CHANGE MODEL NAME HERE
# -------------------------------------------
boolSpacy = False  # set to True if the model uses spacy for tokenization
modelName_spacy = "de_dep_news_trf"
modelName_not_spacy = "dbmdz/german-gpt2" # "bert-base-german-cased" # "dbmdz/german-gpt2"
tokenizerName = modelName_not_spacy  # can be changed for a non-spacy model


# ------------------------------------------------------
def intializeModel(boolSpacy):
    if boolSpacy:
        spacy.prefer_gpu()
        model = spacy.load(modelName_spacy)
        model = scorer.IncrementalLMScorer("spacy/de_dep_news_trf", "cpu")
    else:
        model = scorer.IncrementalLMScorer(modelName_not_spacy, "cpu")

    return model

# ------------------------------------------------------
'''
Specs for getSurprisal function:
@param: requires the text input split into sentences. 
numSentences is needed to create the context 
'''

# @param:
# * numSent is number of sentences that is necssary for the context
# returns: The string with N sentences appended to the beginning of the selected sentence
def context(numSent, sentence, sentIndex, inputText):
    context_plus_sentence = sentence
    next_index = sentIndex - 1
    while (sentIndex-numSent <= next_index < sentIndex) and (next_index >= 0) :
        context_plus_sentence = inputText[next_index] + context_plus_sentence
        next_index = next_index - 1
    return context_plus_sentence

def plot_context(sentIndex, model, inputText, makeFigure, showPlot):
    contextValues = [0, 5, 10]
    context_5 = []
    sentence_visual = inputText[sentIndex]
    words_main_sentence = model.token_score(sentence_visual, rank=False, prob=False)[0]
    words = []
    for word in words_main_sentence:
        words.append(word[0])
    length = len(words)
    words.pop(0)
    y = []  # array of all context states
    if makeFigure: plt.figure()
    for c in contextValues:
        contSent = context(c, sentence_visual, sentIndex, inputText)
        arrayLogProbs = model.token_score(contSent, rank=False, prob=False)[0]
        arrayLogProbs.pop(0)
        # from the array that contains probabilities of all these things, we only take last
        arrayLogProbs_final = arrayLogProbs[-(length - 1):]
        y_inner = []
        # extract only log_probabilities, without text
        for entry in arrayLogProbs_final:
            y_inner.append(entry[1])
        y.append(y_inner)
        if c == 5:
            context_5 = y_inner
        if makeFigure:
            plt.plot(words, y_inner, label=f"context is {c}")
    if makeFigure:
        plt.ylabel("log probabilities")
        plt.legend()
        plt.savefig("context_sentence_" + str(sentIndex) + f"_story{story}")
        if showPlot: plt.show()
    return context_5


def getSurprisal(inputText):
    surprisalDictOuter = {}
    model = intializeModel(boolSpacy)
    inputText = ['<|endoftext|> ' + sentence for sentence in inputText]

    for index, sentence in enumerate(inputText):
        context_5 = plot_context(index, model, inputText, False, False)
        ''' 
        CHANGE INTRODUCED HERE: EACH SENTENCE IS PADDED WITH START TOKEN
        '''
        arrayOfSuggestedWords = []
        print(f"[{index+1} out of {len(inputText)}], sentence is {sentence.replace('<|endoftext|> ', '')}")
        arrayOfProbabilities = model.token_score(sentence, rank=False, prob=True)[0]
        arrayOfProbabilities.pop(0)
        arrayOfLogProbabilities = model.token_score(sentence, rank=False, prob=False)[0]
        arrayOfLogProbabilities.pop(0)
        arrayOfRanks = model.token_score(sentence, rank=True, prob=False)[0]
        arrayOfRanks.pop(0)



        # what is this part for???
        for i, entry in enumerate(arrayOfRanks):
            entry = list(entry)
            entry.pop(1)
            arrayOfRanks[i] = entry

        #arrayOfSuggestedWords = getPredictedWord(sentence, model)

        # ---------------------------------------- Add everything in the dictionary ------------------------------------
        surprisalDictInner = {}
        surprisalDictInner["sentence"] = sentence.replace('<|endoftext|> ', '') # token taken into account in calculation but not shown in final version
        surprisalDictInner["probabilities"] = arrayOfProbabilities
        surprisalDictInner["log_probabilities_context_0"] = arrayOfLogProbabilities
        surprisalDictInner["ranks"] = arrayOfRanks
        surprisalDictInner["log_probabilities_context_5"] = context_5
        surprisalDictInner["predicted_words"] = arrayOfSuggestedWords
        surprisalDictOuter[index] = surprisalDictInner
    return surprisalDictOuter


# ----------------------------------------------------------------------------------------------------------------------


# does not generate token 1 necessary, instead uses the gpt2 predictive algorithm (which one exactly??) to predict one
def getPredictedWord(sentence, model):
    model = pipeline('text-generation', model)
    generatedWordsArray = ["First Word, nothing generated"]
    modelInput = ""
    arrayOfwords = sentence.split()
    for word in arrayOfwords:
        print(f"word is : {word}")
        modelInput += (" " + word)  # do not change the position of the space
        # make sure to use greedy algorithm or top-k with k=1
        gptSuggestion = model(modelInput, max_length=1)[0]["generated_text"]
        gptSuggestionSplit = gptSuggestion.split()
        gptSuggestion = gptSuggestionSplit[len(gptSuggestionSplit)-1] # algorithm generates context + new word, only take the new word
        gptSuggestion = utilities.checkPunctuation(gptSuggestion)
        print(f"suggestion is {gptSuggestion}")
        generatedWordsArray.append(gptSuggestion)
    return generatedWordsArray


# Main Segment---------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Step 2: Compute surprisal for each word in a sentence.
    inputText_array = utilities.convertChunkIntoSentences(startChunk, chunks, pathText, story)
    print(f"There are {len(inputText_array)} sentences being processed..")
    surprisalDict = getSurprisal(inputText_array)

    # Step 3: save each sentence into a dictionary
    saveProbabilitiesTo = probabilitiesPath + f"/chunks{startChunk}-{chunks}"

    if not os.path.exists(probabilitiesPath):
        os.mkdir(probabilitiesPath)
    utilities.saveAsJSON(surprisalDict, saveProbabilitiesTo)



