"""Run all VQA analyses: heatmaps, embeddings, and bias analysis."""
import sys
import os
import argparse
import glob
from datetime import datetime


def purge_all_caches():
    """Delete all cached scores and intermediate results to force full recomputation."""
    from src import config
    
    patterns = [
        os.path.join(config.OUTPUT_STSB_DIR, "*.csv"),
        os.path.join(config.OUTPUT_SMATCH_DIR, "*.csv"),
        os.path.join(config.OUTPUT_BERT_DIR, "*.csv"),
        os.path.join(config.OUTPUT_EMBEDDINGS_DIR, "pairwise_scores_cache.csv"),
        os.path.join(config.OUTPUT_EMBEDDINGS_DIR, "embeddings_cache_keyed.pkl"),
        config.BIAS_CACHE["scores"],
    ]
    
    deleted = 0
    for pattern in patterns:
        for f in glob.glob(pattern):
            os.remove(f)
            deleted += 1
            print(f"  Deleted: {os.path.basename(f)}")
    
    print(f"  Purged {deleted} cached file(s)")


def run_analysis(module_name: str, description: str):
    """Run a single analysis module."""
    print("\n" + "=" * 70)
    print(f"Starting: {description}")
    print("=" * 70)
    
    try:
        if module_name == "heatmap_bert":
            from src import heatmap_bert
            heatmap_bert.main()
        elif module_name == "heatmap_smatch":
            from src import heatmap_smatch
            heatmap_smatch.main()
        elif module_name == "heatmap_stsb":
            from src import heatmap_stsb
            heatmap_stsb.main()
        elif module_name == "embed_analysis":
            from src import embed_analysis
            embed_analysis.main()
        elif module_name == "bias_analysis":
            from src import bias_analysis
            bias_analysis.main()
        elif module_name == "heatmap_embed":
            from src import heatmap_embed
            heatmap_embed.main()
        
        print(f"\n✓ {description} completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ {description} failed with error:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all analyses in sequence."""
    parser = argparse.ArgumentParser(description="VQA Analysis Suite")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Purge all caches and recompute everything from scratch")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "VQA ANALYSIS SUITE" + " " * 30 + "║")
    print("║" + " " * 15 + "Running All Analyses" + " " * 33 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Auto-discover agents from CSV
    from src import config, utils
    agents = utils.get_agent_groups()
    print(f"\n  Agents discovered: {len(agents['lima'])} Lima, {len(agents['nyc'])} NYC, {len(agents['vlm'])} VLM")
    
    if args.force_recompute:
        print("\n⚠ --force-recompute: purging all caches...")
        purge_all_caches()
    
    analyses = [
        ("embed_analysis", "Embedding Analysis (UMAP & PCA)"),
        ("bias_analysis", "Bias Analysis (VLMs vs Humans)"),
        ("heatmap_bert", "BERT Score Heatmaps"),
        ("heatmap_smatch", "SMATCH Score Heatmaps (AMR-based)"),
        ("heatmap_stsb", "STSB-RoBERTa Score Heatmaps"),
    ]
    
    results = {}
    
    for module, description in analyses:
        success = run_analysis(module, description)
        results[description] = success
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 25 + "SUMMARY" + " " * 37 + "║")
    print("╠" + "=" * 68 + "╣")
    
    for description, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        padding = " " * (50 - len(description))
        print(f"║  {description}{padding}{status}  ║")
    
    print("╠" + "=" * 68 + "╣")
    print(f"║  Total time: {str(duration).split('.')[0]}" + " " * (68 - 15 - len(str(duration).split('.')[0])) + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    # Exit code
    all_success = all(results.values())
    if all_success:
        print("✓ All analyses completed successfully!\n")
        sys.exit(0)
    else:
        print("⚠ Some analyses failed. Check logs above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
