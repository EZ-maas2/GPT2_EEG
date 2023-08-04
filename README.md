# GPT2

Guide through data files:

 * BAStokenList relies on audio, pos relies on the text file
 * BAStokenList, final_posList and pos_and_time_story are created in the matrixCreation.py file
 * BAStokenList is created in matrixCreation and used in posStory2 for alignment (specifically, it's filled out by utilities.BAS)

Workflow:
---------

# Step 0(A) - Time and Phonemes - :
  Sound file -> BAS file;
  Contains phonemes with starting and ending times, (spoken) words with starting and ending times
  !! THIS IS WHERE ALL PHONETIC SPELLINGS ARE INTRODUCED!!
  Saved to:   ../text partition TextGrid Story{story}/Part{part}_mono_norm


# Step 0(B) - Compare all words - :
  IN: compareAllWords.py
  Make sure that all words in BAS file are the same as all words in text files (/text partition txt/ Part{part}_mono_norm)
  and in tokenized text
  If not REPLACE BAS VERSIONS MANUALLY as there are more sources with text and text is the original source anyways

# Step 0(C) - PAD THE SENTENCES WITH [Start] and [End] Tokens:
  IN: a_gpt2Surpprisal.py


# Step 1 - Surprisal - :
  IN: a_gpt2Surprisal.py
  (1) takes text as an input; (2) calculates probabilities; (3) Saves to ../probabilities/chunks1-7{6|4}

# Step 2 - POS tagging - :
  IN: b_matrixCreation.py
  (1) Based on the text version of chunks, POS and TAG tags (and chunk numbers) are calculated and saved to ../json_matrices/final_posList_story{story}
  (2) BASTokenList are also created there, and they contain every word in BAS file - Can compare words in audio and text version
  (3) Match all words in BASTokenList to POS, TAG and CHUNKS, save result to \json_matrices\pos_tag_and_time_story{story}_mc"


# Step 3 - Add Probability to the POS and TIME (and TAG and CHUNK) matrix -:
  IN: c_matrix_add_probability.py
  (1) take POS, TIME, TAG and CHUNKS matrix and add probability to it
  ISSUE: words in POS, TIME, TAG and CHUNKS matrix are from BAS file
  (2) Sum log probabilities if tokenized
  (3) Save final result to f'../json_matrices/pos_and_time_and_probs_story{story}'
  
  ADD context probabilities to this step:
  (4) whenever the normal probability is added, the context probability with the same index must be added as well
  -> in the final file all probabilities with all contexts are present

# Step 4 - Prepare Training data -:
  IN: e_props_to_features
  Here the information about what story the participant was focusing on is converted to
  a collection of model-usable .npy arrays, which are saved to /features folder

# Step 5 - Make the model -:
  IN: f_eelbrain.py, with helper functions stored in eelbrain_utilities.py
  (1) In the bottom of the .py file user can select parameters of the model that has to run
  Choices concern:
      a) Which electrodes should be included (a few premade options or a custom list of strings)
      b) Whether to make model for one or all participants (choice between run_for_one and run_for_all functions)
      c) What parameters to include (including second speaker parameters)
  (2) The model is run and pickled to ../model_data/pickled folder
  (3) The information about the model is also saved in the big ../model_data/model_log.json and ../model_data/model_log.csv

# Step 6 - Plot the results of the model -:
  IN: g_plotting_model_results.py
