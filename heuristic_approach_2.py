# %%
import pandas as pd
import re
import unicodedata
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple
import sys
import json


# %%
def analyze_text_normalization(csv_path: str) -> None:
    """
    Analyze a text normalization dataset to identify patterns in character preservation
    across different scripts and normalization patterns.

    Args:
        csv_path: Path to the CSV file containing the text normalization dataset
    """
    print(f"Loading dataset from {csv_path}...")
    # Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    df = df.rename(columns={"raw_comp_writers_text": "raw", "CLEAN_TEXT": "clean"})

    # Handle NaN values
    df["clean"] = df["clean"].fillna("")

    # 1. Analyze script preservation
    script_stats = analyze_script_preservation(df)
    with open("script_stats.json", "w") as f:
        json.dump(script_stats, f, indent=4)

    # 2. Analyze normalization patterns
    pattern_stats = analyze_normalization_patterns(df)
    with open("pattern_stats.json", "w") as f:
        json.dump(pattern_stats, f, indent=4)

    # 3. Visualize results
    visualize_results(script_stats, pattern_stats)

    # 4. Print summary
    print_summary(script_stats, pattern_stats)


def get_script_name(char: str) -> str:
    """
    Identify the script of a character.

    Args:
        char: A single character

    Returns:
        The name of the script
    """
    code = ord(char)

    # Latin-based diacritics - we'll categorize basic Latin + Latin-1 Supplement with diacritics
    if code < 0x0100:
        # Check if it has diacritics
        if unicodedata.combining(char) or unicodedata.decomposition(char):
            return "Latin-diacritics"
        return "Basic-Latin"

    # Other scripts
    if 0x0400 <= code <= 0x04FF:
        return "Cyrillic"
    if 0x0600 <= code <= 0x06FF:
        return "Arabic"
    if 0x0900 <= code <= 0x097F:
        return "Devanagari"
    if 0x0E00 <= code <= 0x0E7F:
        return "Thai"
    if 0x1100 <= code <= 0x11FF or 0xAC00 <= code <= 0xD7AF:
        return "Korean"
    if 0x3040 <= code <= 0x309F:
        return "Hiragana"
    if 0x30A0 <= code <= 0x30FF:
        return "Katakana"
    if 0x4E00 <= code <= 0x9FFF:
        return "CJK"

    # For all other scripts, check unicode database
    try:
        script = unicodedata.name(char).split()[0]
        return script
    except:
        return "Other"


def get_scripts_in_text(text: str) -> Set[str]:
    """
    Identify all scripts present in a text.

    Args:
        text: A string of text

    Returns:
        A set of script names
    """
    scripts = set()
    for char in text:
        if not char.isspace() and not char.isascii():
            script = get_script_name(char)
            scripts.add(script)
    return scripts


def analyze_script_preservation(df: pd.DataFrame, sample_size: int = 10000) -> Dict:
    """
    Analyze how different scripts are preserved in the normalization process.

    Args:
        df: DataFrame containing 'raw' and 'clean' text columns
        sample_size: Number of rows to analyze (for speed)

    Returns:
        Dictionary with script preservation statistics
    """
    print("\nAnalyzing script preservation...")

    # Sample the dataframe if it's large
    if len(df) > sample_size:
        analysis_df = df.sample(sample_size, random_state=42)
    else:
        analysis_df = df

    script_stats = defaultdict(lambda: {"total": 0, "preserved": 0, "percentance": 0})
    non_latin_rows = 0
    preserved_rows = 0

    for i, row in analysis_df.iterrows():
        raw = row["raw"]
        clean = row["clean"]

        if isinstance(raw, float):
            continue
        # Skip if raw is empty
        if not raw or raw.isspace():
            continue

        # Get scripts in raw and clean text
        raw_scripts = get_scripts_in_text(raw)
        clean_scripts = get_scripts_in_text(clean)

        # Track rows with non-Latin characters
        if raw_scripts:
            non_latin_rows += 1
            if clean_scripts:
                preserved_rows += 1

        # For each script in raw, check if it's preserved in clean
        for script in raw_scripts:
            script_stats[script]["total"] += 1
            if script in clean_scripts:
                script_stats[script]["preserved"] += 1
            script_stats[script]["percentance"] = script_stats[script]["preserved"] / script_stats[script]["total"]

    # Calculate preservation rate
    if non_latin_rows > 0:
        overall_preservation_rate = (preserved_rows / non_latin_rows) * 100
    else:
        overall_preservation_rate = 0

    # Add overall rate to the stats
    script_stats["overall"] = {"total": non_latin_rows, "preserved": preserved_rows, "rate": overall_preservation_rate}

    return script_stats


def analyze_normalization_patterns(df: pd.DataFrame, sample_size: int = 10000) -> Dict:
    """
    Analyze common normalization patterns in the dataset.

    Args:
        df: DataFrame containing 'raw' and 'clean' text columns
        sample_size: Number of rows to analyze (for speed)

    Returns:
        Dictionary with normalization pattern statistics
    """
    print("\nAnalyzing normalization patterns...")

    # Sample the dataframe if it's large
    if len(df) > sample_size:
        analysis_df = df.sample(sample_size, random_state=42)
    else:
        analysis_df = df

    # Define pattern recognition regexes
    publishing_terms_regex = re.compile(r"PUBLISHING|COPYRIGHT|RIGHTS|ADMIN|STUDIO|MUSIC|ENTERTAINMENT", re.IGNORECASE)
    business_entities_regex = re.compile(r"LIMITED|LTD|LLC|INC|CORP|GMBH|PTY|S\.A\.|N\.V\.|CO\.|ASSOCIATES", re.IGNORECASE)

    # Initialize counters
    patterns = {"publishing_terms_removed": 0, "business_entities_removed": 0, "non_latin_rows_empty": 0, "name_structure_changed": 0, "total_analyzed": len(analysis_df)}

    for i, row in analysis_df.iterrows():
        raw = row["raw"]
        clean = row["clean"]

        if isinstance(raw, float):
            continue
        # Skip if raw is empty
        if not raw or raw.isspace():
            continue

        # Check if publishing terms are removed
        if publishing_terms_regex.search(raw) and not publishing_terms_regex.search(clean):
            patterns["publishing_terms_removed"] += 1

        # Check if business entities are removed
        if business_entities_regex.search(raw) and not business_entities_regex.search(clean):
            patterns["business_entities_removed"] += 1

        # Check if non-Latin rows are normalized to empty strings
        if get_scripts_in_text(raw) and not clean:
            patterns["non_latin_rows_empty"] += 1

        # Check if name structure changed (e.g., "Last, First" to "First Last")
        if "," in raw and "," not in clean:
            # Simple heuristic to detect name inversions
            raw_parts = [part.strip() for part in raw.split(",")]
            if len(raw_parts) == 2 and f"{raw_parts[1]} {raw_parts[0]}" == clean:
                patterns["name_structure_changed"] += 1

    return patterns


def print_summary(script_stats: Dict, pattern_stats: Dict) -> None:
    """
    Print a summary of the analysis results.

    Args:
        script_stats: Dictionary with script preservation statistics
        pattern_stats: Dictionary with normalization pattern statistics
    """
    print("\n" + "=" * 70)
    print("                   TEXT NORMALIZATION ANALYSIS SUMMARY")
    print("=" * 70)

    # Overall preservation rate
    print(f"\nNon-Latin Character Preservation:")
    print(f"  Overall preservation rate: {script_stats['overall']['rate']:.2f}%")
    print(f"  Rows with non-Latin characters: {script_stats['overall']['total']}")
    print(f"  Rows with preserved non-Latin characters: {script_stats['overall']['preserved']}")

    # Script-specific preservation rates
    print("\nPreservation rates by script:")
    for script, stats in sorted(script_stats.items(), key=lambda x: (-(x[1]["preserved"] / x[1]["total"] if x[1]["total"] > 0 else 0), x[0])):
        if script != "overall" and stats["total"] > 0:
            rate = (stats["preserved"] / stats["total"]) * 100
            print(f"  {script}: {stats['preserved']}/{stats['total']} ({rate:.2f}%)")

    # Normalization patterns
    print("\nNormalization Patterns:")
    total = pattern_stats["total_analyzed"]
    print(f"  Publishing terms removed: {pattern_stats['publishing_terms_removed']} ({pattern_stats['publishing_terms_removed']/total*100:.2f}%)")
    print(f"  Business entities removed: {pattern_stats['business_entities_removed']} ({pattern_stats['business_entities_removed']/total*100:.2f}%)")
    print(f"  Non-Latin rows normalized to empty: {pattern_stats['non_latin_rows_empty']} ({pattern_stats['non_latin_rows_empty']/total*100:.2f}%)")
    print(f"  Name structure changed: {pattern_stats['name_structure_changed']} ({pattern_stats['name_structure_changed']/total*100:.2f}%)")

    # Summary of findings
    print("\nKey Findings:")
    print("  * The dataset shows selective character preservation across different scripts")
    print("  * Business terms and publishing-related information are commonly removed")
    if pattern_stats["non_latin_rows_empty"] > 0:
        print(f"  * {pattern_stats['non_latin_rows_empty']} rows with non-Latin characters were normalized to empty strings")


def visualize_results(script_stats: Dict, pattern_stats: Dict) -> None:
    """
    Visualize the analysis results using matplotlib.

    Args:
        script_stats: Dictionary with script preservation statistics
        pattern_stats: Dictionary with normalization pattern statistics
    """
    # Set the style
    plt.style.use("ggplot")

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. Script preservation rates
    script_data = []
    for script, stats in script_stats.items():
        if script != "overall" and stats["total"] >= 3:  # Only include scripts with enough data
            rate = (stats["preserved"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            script_data.append((script, rate, stats["total"]))

    # Sort by preservation rate
    script_data.sort(key=lambda x: x[1], reverse=True)

    # Extract data for plotting
    scripts = [item[0] for item in script_data]
    rates = [item[1] for item in script_data]
    counts = [item[2] for item in script_data]

    # Create the bar chart
    bars = ax1.barh(scripts, rates, color="skyblue")

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, f"n={count}", va="center", fontsize=8)

    # Customize the plot
    ax1.set_xlabel("Preservation Rate (%)")
    ax1.set_title("Script Preservation Rates")
    ax1.set_xlim(0, 110)  # Leave room for the count labels

    # 2. Normalization patterns
    pattern_labels = ["Publishing terms\nremoved", "Business entities\nremoved", "Non-Latin rows\nto empty", "Name structure\nchanged"]

    pattern_values = [
        pattern_stats["publishing_terms_removed"] / pattern_stats["total_analyzed"] * 100,
        pattern_stats["business_entities_removed"] / pattern_stats["total_analyzed"] * 100,
        pattern_stats["non_latin_rows_empty"] / pattern_stats["total_analyzed"] * 100,
        pattern_stats["name_structure_changed"] / pattern_stats["total_analyzed"] * 100,
    ]

    # Create the bar chart
    pattern_bars = ax2.bar(pattern_labels, pattern_values, color="lightcoral")

    # Add count labels
    for i, bar in enumerate(pattern_bars):
        count = [pattern_stats["publishing_terms_removed"], pattern_stats["business_entities_removed"], pattern_stats["non_latin_rows_empty"], pattern_stats["name_structure_changed"]][i]
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"n={count}", ha="center", fontsize=8)

    # Customize the plot
    ax2.set_ylabel("Percentage of Analyzed Rows (%)")
    ax2.set_title("Normalization Patterns")
    ax2.set_ylim(0, max(pattern_values) * 1.2)  # Leave room for the count labels

    # Add an overall title
    plt.suptitle("Text Normalization Analysis", fontsize=16)

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("normalization_analysis.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved as 'normalization_analysis.png'")
    plt.close()


# %%
if __name__ == "__main__":
    analyze_text_normalization(r"normalization_assesment_dataset_10k.csv")
