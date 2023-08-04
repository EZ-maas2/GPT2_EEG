"""
Objective: to extract semantic (and syntatic?) predictors from the gpt2 model on audiobook input


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
"""
import os.path

from minicons import scorer
import utilities
from transformers import pipeline
#import pandas as pd





# ------------------------------------------------------

'''
@param: requires the text input split into sentences. 

'''
def getSurprisal(inputText):
    # Warning: This will download a 550mb model file if you do not already have it!
    # pretrained german gpt2 can be retrieved from "dbmdz/german-gpt2"
    surprisalDictOuter = {}

    model = scorer.IncrementalLMScorer("dbmdz/german-gpt2", 'cpu')
    pipe = pipeline('text-generation', model="dbmdz/german-gpt2",
                    tokenizer="dbmdz/german-gpt2")

    for sentence in inputText:
        surprisalDictInner = {}
        preparedText = model.prepare_text(sentence)

        arrayOfProbabilities = model.compute_stats(preparedText, rank=False, prob=True)

        #print(f"The conditional probabilities for each word in sentence '{sentence}' is {arrayOfProbabilities}")
        arrayOfLogProbabilities, arrayOfRanks  = model.compute_stats(model.prepare_text(sentence), rank=True, prob=False)
        arrayOfSuggestedWords = getPredictedWord(sentence, pipe)

        surprisalDictInner["probabilities"] = arrayOfProbabilities[0]
        surprisalDictInner["log_probabilities"] = arrayOfLogProbabilities[0]
        surprisalDictInner["ranks"] = arrayOfRanks[0]
        surprisalDictInner["predicted_words"] = arrayOfSuggestedWords
        surprisalDictOuter[sentence] = surprisalDictInner
    return surprisalDictOuter

# does not generate token 1 necessary, instead uses the gpt2 predictive algorithm (which one exactly??) to predict one
def getPredictedWord(sentence, model):
    sentence.replace(".", '')
    generatedWordsArray = []
    modelInput = ""
    arrayOfwords = sentence.split()
    arrayOfwords.pop()

    for word in arrayOfwords:
        print(f"word is : {word}")
        modelInput += " " + word

        # make sure to use greedy algorithm or top-k with k=1
        gptSuggestion = model(modelInput, max_length=1)[0]["generated_text"]
        gptSuggestionSplit = gptSuggestion.split()
        gptSuggestion = gptSuggestionSplit[len(gptSuggestionSplit)-1]
        print(gptSuggestion)
        generatedWordsArray.append(gptSuggestion)
    return generatedWordsArray

# ------------------------------------------------------
'''
Parse Story to get surprisal for each sentence
'''
# Step 1: Separate text chunck into sentences
story = 1
chunks = 2
startChunk = 1
inputText =[]
pathText = f"D:/lars project/EEG_GPT3/data/text partition txt/text partition txt Story{story}/"
probabilitiesPath = f"D:/lars project/EEG_GPT3/data/probabilities/Story{story}"

# surprisalDictOuter will have structure {"sentence" = "..": surprisalDictInner}
# surprisalDictInner will have structure {"ranks" = [], "probabilities" = [], "recommended words" = []}


for chunk in range(startChunk, chunks+1):
    fileName = f"Part{chunk}_mono_norm"
    pathTextFull = pathText+fileName+".txt"

    if not os.path.exists(pathTextFull):
        print(f"attempting .docx to .txt with {pathText+fileName}")
        utilities.docxToText(pathText+fileName)

    with open(pathTextFull, "r") as file:
        text = file.readlines()
        for sentence in text:
            inputChunkText = utilities.splitIntoSentences(sentence)
            for i in inputChunkText:
                if i != "":
                    inputText.append(i)

# Step 2: Compute surprisal for each word in a sentence.
print(f"Computing surprisal for {pathTextFull}")
surprisalDict = getSurprisal(inputText)


# Step 3: save each sentence into a dictionary with a structure
# {“sentence = ”: “array of probabilities = ”:,
# 		“array of ranks =”:,
# 		“array of preferred words = ”:}



saveProbabilitiesTo = probabilitiesPath+f"/chunks{startChunk}-{chunks}"
if not os.path.exists(probabilitiesPath):
    os.mkdir(probabilitiesPath)

utilities.saveAsJSON(surprisalDict, saveProbabilitiesTo)
utilities.saveAsCSV(surprisalDict, saveProbabilitiesTo)
utilities.prettyPrintSave(surprisalDict, saveProbabilitiesTo)

with open(saveProbabilitiesTo+".txt", "w") as file:
    file.write(str(surprisalDict))

#pd.DataFrame(surprisalDict)

