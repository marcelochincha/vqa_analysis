import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def get_score(comp_obj, row, df_str_answers, cache):
    #get the string from the df answers
    str_i = df_str_answers.loc[(row["VIDEO"], row["QUESTION_NUM"], row["AGENT_I"], row["REPETITION_I"]), "ANSWER"]
    str_j = df_str_answers.loc[(row["VIDEO"], row["QUESTION_NUM"], row["AGENT_J"], row["REPETITION_J"]), "ANSWER"]
    #now get the embeddings from the cache
    obj_i = cache[str_i]
    obj_j = cache[str_j]
    #normalize the embeddings and compute the cosine similarity
    return comp_obj(obj_i, obj_j)

def compute_scores(df_comp, missing_mask, df_str_answers, cache, comparator):
    df_comp.loc[missing_mask, "RESULT"] = df_comp[missing_mask].progress_apply(lambda row: get_score(comparator,row, df_str_answers, cache), axis=1)

# pipeline function that takes the input csv, preprocesses it, generates the cache and the pairwise comparations, and returns the comparations dataframe, it also takes an optional objecter function to generate the embeddings, if not provided it will use a default one
# if the objecter is not provided, it will be assumed that the comparation does NOT require the use of an embedding model, and the cache will be generated with the raw strings, and the comparation function will be a simple string comparison (1 if equal, 0 if not)
def process_pipeline(
        input_csv="./data/r2.csv",
        preprocess_cache_file="./cache/preprocess_cache.pkl",
        pairwise_comparations_file="./cache/pairwise_comparations.csv",
        preprocessor=None, # Object with a __call__ method that takes a list of strings and returns a list of embeddings, if None, the raw strings will be used as embeddings
        comparator=None # Object with a __call__ method that takes two objects and returns a similarity score
    ):
    print("==== VQA DATA PROCESSOR ====")
    print("Loading data...")
    df = pd.read_csv(input_csv,keep_default_na=False)
    print("Preprocessing data...")
    df_str_answers = preprocess(df)
    print("✓ Data ready")

    #LOAD CACHE
    print("Loading cache...")
    cache = load_cache(preprocess_cache_file)
    should_embed = cache is None or len(cache) < len(df_str_answers["ANSWER"].unique())
    if should_embed:
        print("Pre process cache is missing or incomplete. Generating new cache...")
        cache = generate_cache(df_str_answers, cache, "ANSWER", preprocessor)
    else:
        print("✓ Cache loaded with all required entries")
    print(f"✓ Cache updated with {len(cache)} entries")


    #get only the repetition 1 for the embedding matrix, since the other repetitions are the same question and answer
    df_str_answers = df_str_answers[df_str_answers["REPETITION"] == 1]
    embed_matrix = np.stack(df_str_answers["ANSWER"].map(cache))
    print(f"Embed matrix shape: {embed_matrix.shape}")
    
    print("\nGenerating pairwise comparations...")
    df_str_answers["VIDEO_SECTOR"] = df_str_answers["VIDEO"].apply(get_video_sector)
    df_comp = generate_comparations_df(df_str_answers)
    df_comp_cached = load_pairwise_comparations(pairwise_comparations_file)

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

        compute_scores(df_comp,missing_mask, df_str_answers, cache, comparator)
        #save the comparations in a csv file for later use
        df_comp.to_csv(pairwise_comparations_file , index=False)

    print("BLOCK check")
    print(df_comp.head())
    df_comp["VIDEO_SECTOR"] = df_comp["VIDEO"].apply(get_video_sector)
    return df_comp