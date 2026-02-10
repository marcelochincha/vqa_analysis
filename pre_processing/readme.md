# Pre processing instructions

Join both the jsons of vlms answers and human responses in a single csv file. The csv file should have the following columns:

AGENT,VIDEO,QUESTION_NUM,QUESTION,ANSWER

Where:
- AGENT: The name of the agent (e.g., "human_region_x", "vlm1", "vlm2", etc.)
- VIDEO: The name of the video (e.g., "Robusto2_XXX")
- QUESTION_NUM: The number of the question (e.g., 1, 2, 3, etc.)
- QUESTION: The text of the question (e.g., "What is the color of the car?")
- ANSWER: The text of the answer provided by the agent (e.g., "The car is red.")

Make sure to handle any missing or incomplete data appropriately, and ensure that the final CSV file is well-formatted and ready for analysis.

Repetition may occur in the csv file, due to the fact that the same question may be asked to multiple agents for the same video. This is expected and should be retained in the final dataset for comprehensive analysis.

Finally pass to vlm processing script the csv file with the human answers, so that it can be merged with the vlm answers in a single csv file.