import pandas as pd
import numpy as np
import os

df = pd.read_csv(os.path.join("data", "r2.csv"))

#then check each block sample of the data 
def divide_into_blocks(df, block_size=5):
    df = df.sort_values(by=["VIDEO", "QUESTION_NUM"])
    df["BLOCK"] = (df["QUESTION_NUM"] - 1) // block_size + 1
    return df

df = divide_into_blocks(df)
df_b2 = df[df["BLOCK"] == 2]

def clean_answer(answer):
    try:
        # Attempt to directly convert to a number
        return float(answer)
    except ValueError:
        # Apply heuristics if direct conversion fails
        words = str(answer).split()
        if words:
            try:
                # Check if the first word is a number
                return float(words[0])
            except ValueError:
                pass
            try:
                # Check if the last word is a number
                return float(words[-1])
            except ValueError:
                pass
        # Mark for manual review if no number is found at the start or end
        return "REVISAR"

# Apply cleaning to block 2 answers
df_b2["CLEAN_ANSWER"] = df_b2["ANSWER"].apply(clean_answer)

# Separate answers needing manual review
to_review = df_b2[df_b2["CLEAN_ANSWER"] == "REVISAR"]

# Prompt user to replace values marked as 'REVISAR'
for index, row in to_review.iterrows():
    print(f"Answer needing review: ({row['ANSWER']}) FROM  {row['AGENT']} in {row['VIDEO']}|Q{row['QUESTION_NUM']}")
    replacement = input("REPLACE TO? (Enter a number or leave blank to skip): ")
    if replacement.strip():
        try:
            df_b2.at[index, "CLEAN_ANSWER"] = float(replacement)
        except ValueError:
            print("Invalid input. Skipping this answer.")

# Save results
df_b2.to_csv(os.path.join("data", "cleaned_block_2.csv"), index=False)
to_review = df_b2[df_b2["CLEAN_ANSWER"] == "REVISAR"]
to_review.to_csv(os.path.join("data", "review_block_2.csv"), index=False)

print("Limpieza completada. Respuestas revisadas guardadas en 'cleaned_block_2.csv'.")
print("Respuestas que necesitan revisión manual guardadas en 'review_block_2.csv'.")
