"""Run all VQA analyses: heatmaps, embeddings, and bias analysis."""
import sys
from datetime import datetime


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
    start_time = datetime.now()
    
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "VQA ANALYSIS SUITE" + " " * 30 + "║")
    print("║" + " " * 15 + "Running All Analyses" + " " * 33 + "║")
    print("╚" + "=" * 68 + "╝")
    
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
