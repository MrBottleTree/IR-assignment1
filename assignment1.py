import numpy as np
import matplotlib.pyplot as plt
from loader.data_loader import PlainDataLoader
from itertools import groupby
from collections import Counter
from scipy.stats import linregress

def extract_chinese_segments(text, is_chinese_func):
    """Extracts contiguous blocks of Chinese characters."""
    segments = []
    for k, g in groupby(text, key=is_chinese_func):
        if k:
            segments.append("".join(g))
    return segments

def is_chinese_char(char):
    if len(char) != 1: return False
    return 0x4E00 <= ord(char) <= 0x9FFF

def calculate_zipf_metrics(ranks, freqs, label):
    """
    Calculates the slope (alpha) and R^2 of the log-log data.
    Returns a formatted string for the legend and the slope.
    """
    # Work in log-log space
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)
    
    # Perform linear regression to find the slope
    slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)
    
    r_squared = r_value**2
    
    print(f"--- Metrics for {label} ---")
    print(f"Slope (Alpha): {slope:.4f} (Ideal is -1.0)")
    print(f"R-squared:     {r_squared:.4f}")
    print("-" * 30)
    
    # Using raw string r"..." to handle backslashes correctly for LaTeX
    legend_label = fr"{label} ($\alpha$={slope:.2f}, $R^2$={r_squared:.2f})"
    
    return slope, r_squared, legend_label

# --- Main Execution ---
obj = PlainDataLoader()
dataset_name = 'tangsong' 

# Safe loading
if dataset_name in obj.datasets.keys():
    data = obj.body_extractor(dataset_name)
else:
    data = obj.body_extractor(list(obj.datasets.keys())[3])

print(f"Processing {len(data)} lines...")

# Counting
segment_counts = Counter()
char_counts = Counter()

for line in data:
    segments = extract_chinese_segments(line, is_chinese_char)
    segment_counts.update(segments)
    for seg in segments:
        char_counts.update(seg)

# Sorting
def get_arrays(counter_obj):
    sorted_counts = sorted(counter_obj.values(), reverse=True)
    return np.arange(1, len(sorted_counts) + 1), np.array(sorted_counts)

rank_seg, freq_seg = get_arrays(segment_counts)
rank_char, freq_char = get_arrays(char_counts)

# --- Plotting with Metrics ---
plt.figure(figsize=(12, 8))

k = 10000 

slope_seg, r2_seg, label_seg = calculate_zipf_metrics(rank_seg[:k], freq_seg[:k], "Phrases")
slope_char, r2_char, label_char = calculate_zipf_metrics(rank_char[:k], freq_char[:k], "Characters")

# Calculate Ideal Zipf for BOTH datasets
# Ideal = First Frequency / rank
ideal_zipf_char = freq_char[0] / rank_char
ideal_zipf_seg = freq_seg[0] / rank_seg

# Plotting
# 1. Phrases (Blue) and its Ideal (Cyan Dashed)
plt.loglog(rank_seg[:k], freq_seg[:k], 'b-', linewidth=2, label=label_seg)
plt.loglog(rank_seg[:k], ideal_zipf_seg[:k], 'c--', linewidth=1.5, label="Ideal Phrase Slope (-1.0)")

# 2. Characters (Green) and its Ideal (Red Dashed)
plt.loglog(rank_char[:k], freq_char[:k], 'g-', linewidth=2, alpha=0.8, label=label_char)
plt.loglog(rank_char[:k], ideal_zipf_char[:k], 'r--', linewidth=2, label="Ideal Char Slope (-1.0)")

plt.grid(True, which="both", ls="-", alpha=0.4)
plt.xlabel('Rank (log scale)', fontsize=12)
plt.ylabel('Frequency (log scale)', fontsize=12)
plt.title(f"Zipf's Law Analysis: Phrases vs Characters (Dataset: {dataset_name})", fontsize=14)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()