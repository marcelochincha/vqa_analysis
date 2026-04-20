import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import os
import pandas as pd
from tqdm import tqdm
import pickle

class Embedder():
    def __init__(self):
        self.func = lambda x: np.random.rand(768)  # Dummy embedding function, replace with actual model encoding
    def __call__(self, s):
        return self.func(s)  # Replace with actual batch encoding

def save_cache(cache, filename):
    with open(filename, 'wb') as f:
        pickle.dump(cache, f)

def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return None
    
def load_pairwise_comparations(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return None

def preprocess(df_data):
    df_data = df_data.copy()
    df_data["REPETITION"] = df_data["REPETITION"].astype(np.uint8)
    df_data["QUESTION_NUM"] = df_data["QUESTION_NUM"].astype(np.uint8)
    df_data["VIDEO"] = df_data["VIDEO"].apply(lambda x: int(x.split("_")[1])).astype(np.uint8)  # Remove file extension
    df_data["AGENT"] = df_data["AGENT"].astype("category")
    df_data["BLOCK"] = df_data["BLOCK"].astype(np.uint8)

    #set the index to be the combination of VIDEO, QUESTION_NUM, AGENT and REPETITION
    df_data.set_index(["VIDEO", "QUESTION_NUM", "AGENT", "REPETITION"], inplace=True, drop=False)
    return df_data

#this is a generic function that should generate a dictionary for convert:
# str -> embed
# str -> amr graph
# str -> any other pre_processed object that we want to use in the comparations
#THE CACHE IS A DICTIONARY THAT MAPS STRINGS TO THEIR PRE_PROCESSED VERSION, THIS WAY WE AVOID REPEATEDLY PRE_PROCESSING THE SAME STRING
def generate_cache(df_data, obj_cache, answers_column, objecter, checkpoint=100,cache_file = "cache.pkl"):
    loaded_strings = set(df_data[answers_column].unique())
    current_strings = set(obj_cache.keys())
    string_to_processes = list(loaded_strings - current_strings)
    
    if len(string_to_processes) == 0:
        print("✓ Cache is already up to date")
        return obj_cache   
    print(f"Processing {len(string_to_processes)} new strings for cache...")
    for start in tqdm(range(0, len(string_to_processes), checkpoint)):
            end = min(start + checkpoint, len(string_to_processes))
            batch_answers = string_to_processes[start:end]
            batch_embeddings = objecter(batch_answers)
            # Store in cache
            for j, emb in enumerate(batch_embeddings):
                obj_cache[batch_answers[j]] = emb
            save_cache(obj_cache, cache_file)
    return obj_cache

def generate_comparations_df(df_data):
    KEYS = ["VIDEO", "BLOCK", "QUESTION_NUM", "AGENT", "REPETITION"]
    df_data = df_data.copy().reset_index(drop=True)
    df_comp = pd.merge(df_data[KEYS], df_data[KEYS], on=["VIDEO", "QUESTION_NUM", "BLOCK"], suffixes=('_I', '_J')) 
    print(f"Generated {len(df_comp)} pairwise comparations")
    print(df_comp.head())
    return df_comp
            
def cosine_pairwise(row,df_str_answers, cache):
    #get the string from the df answers
    str_i = df_str_answers.loc[(row["VIDEO"], row["QUESTION_NUM"], row["AGENT_I"], row["REPETITION_I"]), "ANSWER"]
    str_j = df_str_answers.loc[(row["VIDEO"], row["QUESTION_NUM"], row["AGENT_J"], row["REPETITION_J"]), "ANSWER"]

    #now get the embeddings from the cache
    emb_i = cache[str_i]
    emb_j = cache[str_j]

    #normalize the embeddings and compute the cosine similarity
    emb_i = emb_i / np.linalg.norm(emb_i)
    emb_j = emb_j / np.linalg.norm(emb_j)
    cosine_sim = np.dot(emb_i, emb_j)
    return cosine_sim

def get_video_sector(video_id):
    return "Lima" if video_id <= 100 else "NYC"

def aggregate_comparations(df_comp):
    REQUIRED_COLUMNS = {"VIDEO", "QUESTION_NUM", "AGENT_I", "AGENT_J", "BLOCK", "VIDEO_SECTOR", "RESULT"}
    if not REQUIRED_COLUMNS.issubset(df_comp.columns):
        raise ValueError(f"DataFrame must contain columns: {REQUIRED_COLUMNS}")
    
    agg_df = (
        df_comp
        .groupby(
            ["VIDEO", "QUESTION_NUM", "AGENT_I", "AGENT_J", "BLOCK", "VIDEO_SECTOR"],
            as_index=False
        )["RESULT"]
        .mean()
    )
    agg_df2 = (
        agg_df
        .groupby(
            ["AGENT_I", "AGENT_J", "BLOCK", "VIDEO_SECTOR"],
            as_index=False
        )["RESULT"]
        .mean()
    )  
    return agg_df2 
    
def main():
    print("==== VQA DATA PROCESSOR ====")
    print("Loading data...")
    df = pd.read_csv("./data/r2.csv",keep_default_na=False)
    print("Preprocessing data...")
    df_str_answers = preprocess(df)
    print("✓ Data ready")

    #LOAD CACHE
    print("Loading cache...")
    cache = load_cache(PREPROCESS_CACHE_FILE)
    should_embed = cache is None or len(cache) < len(df_str_answers["ANSWER"].unique())
    if should_embed:
        print("Cache is missing or incomplete. Generating new cache...")
        embeder = Embedder()
        cache = generate_cache(df_str_answers, cache, "ANSWER", embeder)
    else:
        print("✓ Cache loaded with all required entries")
    print(f"✓ Cache updated with {len(cache)} entries")


    #get only the repetition 1 for the embedding matrix, since the other repetitions are the same question and answer
    df_str_answers = df_str_answers[df_str_answers["REPETITION"] == 1]
    embed_matrix = np.stack(df_str_answers["ANSWER"].map(cache))
    print(f"Embed matrix shape: {embed_matrix.shape}")
    
    #reduce_and_plot_pca_by_block(embed_matrix, df_str_answers)

    #now compute the cosine pairwise comparations and save them in a dataframe for later use, we can use the cache to avoid recomputing the embeddings
    print("\nGenerating pairwise comparations...")
    df_str_answers["VIDEO_SECTOR"] = df_str_answers["VIDEO"].apply(get_video_sector)
    df_comp = generate_comparations_df(df_str_answers)
    df_comp_cached = load_pairwise_comparations(PAIRWISE_COMPARATIONS_FILE)


    #comput here
    if df_comp_cached is not None and len(df_comp_cached) == len(df_comp):
        print("✓ Pairwise comparations already computed and loaded from file")
        df_comp = df_comp_cached
    else:
        #get the diference from df_comp and df_comp_cached to only compute the missing comparations
        if df_comp_cached is not None:
            df_comp_check = df_comp.merge(df_comp_cached, on=["VIDEO", "QUESTION_NUM", "AGENT_I", "AGENT_J"], how="left", suffixes=('', '_cached'))
            missing_mask = df_comp_check["RESULT"].isna()
            print(f"Computing {missing_mask.sum()} missing pairwise comparations...")
        else:
            print("Computing all pairwise comparations...")
            missing_mask = np.ones(len(df_comp), dtype=bool)  # All comparations are missing if no cache is loaded        
            print(df_comp.head())
        df_comp.loc[missing_mask, "RESULT"] = df_comp[missing_mask].progress_apply(lambda row: cosine_pairwise(row, df_str_answers, cache), axis=1)
        #save the comparations in a csv file for later use
        df_comp.to_csv(PAIRWISE_COMPARATIONS_FILE , index=False)

    print("BLOCK check")
    print(df_comp.head())
    df_comp["VIDEO_SECTOR"] = df_comp["VIDEO"].apply(get_video_sector)

#LOAD AND SETUP DATA
if __name__ == "__main__":
    main()