# Find out which words in BAS files do not match with words in the tokenized scripts
import json
import utilities
import os

rootDir = os.getcwd()

with open(rootDir+ f"\data\Story1_tokenized.json", encoding="UTF-8") as file:
    story1_written  = json.load(file)
    file.close()

with open(rootDir + f"\data\Story2_tokenized.json", encoding="UTF-8") as file2:
    story2_written = json.load(file2)
    file2.close()

story1_BAS = []
story2_BAS = []

for chunk in range(1, 76+1):
    utilities.BASfile(rootDir+f"/data/text partition TextGrid/text partition TextGrid Story1/Part{chunk}_mono_norm.TextGrid", "words",
                      story1_BAS, [], [])

for c in range(1, 74+1):
    utilities.BASfile(rootDir+f"/data/text partition TextGrid/text partition TextGrid Story2/Part{c}_mono_norm.TextGrid", "words",
                      story2_BAS, [], [])




print(f"Story 2 -- length text {len(story2_written)}, length BAS {len(story2_BAS)} ")

for i in range(0, min(len(story2_BAS), len(story2_written))):
    if story2_BAS[i].lower() != story2_written[i].lower():
        print(f"[{i}]: BAS[{story2_BAS[i].lower()}] not equal to text[{story2_written[i].lower()}]; Index {i}, chunk~ {int(i/150)}")


print(f"Story 1 --text-- [{len(story1_written)}], len BAS [{len(story1_BAS)}]")
for i in range(0, min(len(story1_BAS), len(story1_written))):
    if story1_BAS[i].lower() != story1_written[i].lower():
        print(f"[{i}]: BAS[{story1_BAS[i].lower()}] not equal to text[{story1_written[i].lower()}]; Index {i}, chunk~ {int(i/150)}")