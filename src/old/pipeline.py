import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import os
import pandas as pd
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
from src.old import config
from sklearn.decomposition import PCA
from src.heatmap_common import plot_similarity_heatmap
tqdm.pandas()

EMBED_MODEL_SMALL = "all-MiniLM-L6-v2"  # For testing or smaller datasets
BATCH_SIZE = 64  # Adjust based on your GPU/CPU capabilities
CHECKPOINT = 512
CACHE_FILE = "embed_cache.pkl"

class Embedder():
    def __init__(self):
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(EMBED_MODEL_SMALL)
        print("✓ Model loaded")

    def __call__(self, s):
        if isinstance(s, list):
            return self.model.encode(s, batch_size=BATCH_SIZE, show_progress_bar=False)
        else:
            return self.model.encode(s)

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
def generate_cache(df_data, obj_cache, answers_column, objecter):
    loaded_strings = set(df_data[answers_column].unique())
    current_strings = set(obj_cache.keys())
    string_to_processes = list(loaded_strings - current_strings)
    
    if len(string_to_processes) == 0:
        print("✓ Cache is already up to date")
        return obj_cache   
    print(f"Processing {len(string_to_processes)} new strings for cache...")
    for start in tqdm(range(0, len(string_to_processes), CHECKPOINT)):
            end = min(start + CHECKPOINT, len(string_to_processes))
            batch_answers = string_to_processes[start:end]
            batch_embeddings = objecter(batch_answers)
            # Store in cache
            for j, emb in enumerate(batch_embeddings):
                obj_cache[batch_answers[j]] = emb
            save_cache(obj_cache, CACHE_FILE)
    return obj_cache

def generate_comparations_df(df_data):
    KEYS = ["VIDEO", "BLOCK", "QUESTION_NUM", "AGENT", "REPETITION"]
    df_data = df_data.copy().reset_index(drop=True)
    df_comp = pd.merge(df_data[KEYS], df_data[KEYS], on=["VIDEO", "QUESTION_NUM", "BLOCK"], suffixes=('_I', '_J')) 
    print(f"Generated {len(df_comp)} pairwise comparations")
    print(df_comp.head())
    return df_comp

def plot_by_block(reduced_embeddings : np.array, df: pd.DataFrame , reductor : str) -> None:
    """Create PCA plots separated by block and video sector."""
    
    REQUIRED_COLUMNS = {"BLOCK", "VIDEO_SECTOR", "AGENT"}
    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {REQUIRED_COLUMNS}")
    
    
    print("\n=== PCA by Block and Sector ===")
    for block in sorted(df["BLOCK"].unique()):
        # Fit PCA once per block with ALL videos (Lima + NYC)
        block_mask = df["BLOCK"] == block
        embeddings_block = reduced_embeddings[block_mask]
        df_block = df[block_mask]
        min_x, max_x = embeddings_block[:, 0].min(), embeddings_block[:, 0].max()
        min_y, max_y = embeddings_block[:, 1].min(), embeddings_block[:, 1].max()    

        # Now plot each sector separately in the same embedding space
        for sector in ['Lima', 'NYC']: 
            sector_mask = df_block["VIDEO_SECTOR"] == sector
            embeddings_2d = embeddings_block[sector_mask]
            df_subset = df_block[sector_mask]
            print(f"  {sector}: {len(df_subset)} samples")
            if len(df_subset) < 1:
                print(f"Skipping (no data)")
                continue
            
            # Plot with individual agent colors and markers
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Collect handles for grouped legend
            handles_vlm = []
            handles_lima = []
            handles_nyc = []
            
            # Plot VLMs (orange tones)
            for agent in config.VLM_AGENTS:
                if agent in df_subset["AGENT"].values:
                    agent_mask = df_subset["AGENT"] == agent
                    mask_array = agent_mask.values
                    
                    scatter = ax.scatter(
                        embeddings_2d[mask_array, 0],
                        embeddings_2d[mask_array, 1],
                        color=config.PLOT_COLORS["VLM"],
                        marker=config.AGENT_MARKERS_MAP[agent],
                        label=agent,
                        alpha=0.7,
                        s=60,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    handles_vlm.append(scatter)
            
            # Plot Lima humans (light blue tones)
            for agent in config.LIMA_AGENTS:
                if agent in df_subset["AGENT"].values:
                    agent_mask = df_subset["AGENT"] == agent
                    mask_array = agent_mask.values
                    scatter = ax.scatter(
                        embeddings_2d[mask_array, 0],
                        embeddings_2d[mask_array, 1],
                        color=config.PLOT_COLORS["HUMAN_LIMA"],
                        marker=config.AGENT_MARKERS_MAP[agent],
                        label=agent,
                        alpha=0.7,
                        s=60,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    handles_lima.append(scatter)
            
            # Plot NYC humans (dark blue tones)
            for agent in config.NYC_AGENTS:
                if agent in df_subset["AGENT"].values:
                    agent_mask = df_subset["AGENT"] == agent
                    mask_array = agent_mask.values
                    scatter = ax.scatter(
                        embeddings_2d[mask_array, 0],
                        embeddings_2d[mask_array, 1],
                        color=config.PLOT_COLORS["HUMAN_NYC"],
                        marker=config.AGENT_MARKERS_MAP[agent],
                        label=agent,
                        alpha=0.7,
                        s=60,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    handles_nyc.append(scatter)
            
            # Create grouped legend: VLMs | GRUPO LIMA | GRUPO NYC
            all_handles = []
            all_labels = []

            if handles_vlm:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ VLMs ━━━',
                                             markerfacecolor=config.PLOT_COLORS["VLM"], markersize=0, linestyle='None'))
                all_labels.append('━━━ VLMs ━━━')
                for h in handles_vlm:
                    all_handles.append(h)
                    all_labels.append(h.get_label())

            if handles_lima:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ LIMA HUMANS ━━━',
                                             markerfacecolor=config.PLOT_COLORS["HUMAN_LIMA"], markersize=0, linestyle='None'))
                all_labels.append('━━━ LIMA HUMANS ━━━')
                for h in handles_lima:
                    all_handles.append(h)
                    all_labels.append(h.get_label())

            if handles_nyc:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ NYC HUMANS ━━━',
                                             markerfacecolor=config.PLOT_COLORS["HUMAN_NYC"], markersize=0, linestyle='None'))
                all_labels.append('━━━ NYC HUMANS ━━━')
                for h in handles_nyc:
                    all_handles.append(h)
                    all_labels.append(h.get_label())

            ax.set_title(f"PCA - Block {block} - Videos {sector} (Q{(block-1)*5+1}-Q{block*5})", 
                        fontsize=config.PLOT_CONFIG["title_fontsize"], fontweight="bold")
            ax.set_xlabel(f"DIM 1", fontsize=11)
            ax.set_ylabel(f"DIM 2", fontsize=11)
            ax.legend(all_handles, all_labels, loc="center left", bbox_to_anchor=(1, 0.5), 
                     frameon=True, fontsize=config.PLOT_CONFIG["legend_fontsize"], ncol=1)
            ax.grid(True, alpha=0.3)
            
            # add a little padding to the limits for better visualization
            x_padding = (max_x - min_x) * 0.05
            y_padding = (max_y - min_y) * 0.05
            ax.set_xlim(min_x - x_padding, max_x + x_padding)
            ax.set_ylim(min_y - y_padding, max_y + y_padding)            
            
            plt.tight_layout()
            
            sector_label = sector.lower()
            output_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, 
                                      f"{reductor}_{block}_{sector_label}.png")
            plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
            plt.close()
            
            print(f"  Saved: {os.path.basename(output_path)}")
            
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

def plot_cosine_heatmap(df_agg,title,column, color_label, value_range, fmt,output_dir):
    REQUIRED_COLUMNS = {"AGENT_I", "AGENT_J", "BLOCK", "VIDEO_SECTOR", column}
    if not REQUIRED_COLUMNS.issubset(df_agg.columns):
        raise ValueError(f"DataFrame must contain columns: {REQUIRED_COLUMNS}")
    
    regions = df_agg["VIDEO_SECTOR"].unique()
    blocks = df_agg["BLOCK"].unique()
    
    v_min = value_range[0]
    v_max = value_range[1]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cmap", ["#ffffff", "#963fc9"])
    
    for region in regions:
        for block in blocks:
            print(f"Generating heatmap for {region} Block {block}...")
            subset = df_agg[(df_agg["VIDEO_SECTOR"] == region) & (df_agg["BLOCK"] == block)]
            if subset.empty:
                print(f"Skipping heatmap for {region} Block {block} (no data)")
                continue
            
            pivot_table = subset.pivot_table(
                index="AGENT_I", 
                columns="AGENT_J", 
                values=column
            )
            print(pivot_table)
            
            plot_similarity_heatmap(
                pivot_table,
                title= title +  f" - Block {block}|{region}",
                color_label= color_label,
                output_path= os.path.join(output_dir, f"heatmap_{region.lower()}_block{block}_{column}.png"),
                cmap=cmap,
                fmt = fmt,
                vmin=v_min,
                vmax=v_max
            )
            
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


def reduce_pca_by_block(embeddings, df):
    reduced_embeddings = np.zeros((embeddings.shape[0], 2))
    pca = PCA(n_components=2, random_state=42)
    for block in sorted(df["BLOCK"].unique()):
        block_mask = df["BLOCK"] == block
        embeddings_block = embeddings[block_mask]
        pca.fit(embeddings_block)
        print(f"Block {block}: Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
        reduced_embeddings[block_mask] = pca.transform(embeddings_block)
        
    return reduced_embeddings


from umap import UMAP
def reduce_umap_by_block(embeddings, df):
    reduced_embeddings = np.zeros((embeddings.shape[0], 2))
    for block in sorted(df["BLOCK"].unique()):
        block_mask = df["BLOCK"] == block
        embeddings_block = embeddings[block_mask]
        umap = UMAP(n_components=2, random_state=42)
        reduced_embeddings[block_mask] = umap.fit_transform(embeddings_block)
    return reduced_embeddings


def compute_thresholds(df_comp):
    #check the value of result to convert it to 0 or 1 a boolean and then compute the porcentage of aggrement for each agent pair, block and video sector
    df_comp["AGREEMENT"] = df_comp["RESULT"] > 0.5
    df_agg = (
        df_comp
        .groupby(["AGENT_I", "AGENT_J", "BLOCK", "VIDEO_SECTOR"], as_index=False)["AGREEMENT"]
        .sum()
    )
    #now divide it by the origianl number of pairs between those agents, block and video sector to get the percentage of agreement
    df_agg["TOTAL_PAIRS"] = df_comp.groupby(["AGENT_I", "AGENT_J", "BLOCK", "VIDEO_SECTOR"])["AGREEMENT"].transform("count")
    df_agg["AGREEMENT_RATE"] = df_agg["AGREEMENT"] * 100 / df_agg["TOTAL_PAIRS"]
    
    print("\n=== Agreement Rates ===")
    print(df_agg[["AGENT_I", "AGENT_J", "BLOCK", "VIDEO_SECTOR", "AGREEMENT_RATE"]].head(20))
    return df_agg
    

    
def main():
    print("==== VQA DATA PROCESSOR ====")
    print("Loading data...")
    df = pd.read_csv("./data/r2.csv",keep_default_na=False)
    print("Preprocessing data...")
    df_str_answers = preprocess(df)
    print("✓ Data ready")

    #LOAD CACHE
    print("Loading cache...")
    cache = load_cache(CACHE_FILE)
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
    df_comp_cached = load_pairwise_comparations(config.PAIRWISE_COMPARATIONS_FILE)


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
        df_comp.to_csv(config.PAIRWISE_COMPARATIONS_FILE , index=False)

    print("BLOCK check")
    print(df_comp.head())


    #add region
    df_comp["VIDEO_SECTOR"] = df_comp["VIDEO"].apply(get_video_sector)
    
    df_agg_threshold = compute_thresholds(df_comp)
    
    
    #get thresholding
    
    #agg_df = aggregate_comparations(df_comp)
    plot_cosine_heatmap(df_agg_threshold,"Cosine similarity Agreement", "AGREEMENT_RATE", "Agreement (%)", (0,100), "%d", config.OUTPUT_EMBEDDINGS_DIR)
    #
    #pca_embeds = reduce_pca_by_block(embed_matrix, df_str_answers)
    #plot_by_block(pca_embeds, df_str_answers, "pca")
    #
    #umap_embeds = reduce_umap_by_block(embed_matrix, df_str_answers)
    #plot_by_block(umap_embeds, df_str_answers, "umap")


#LOAD AND SETUP DATA
if __name__ == "__main__":
    main()