import matplotlib.pyplot as plt
import numpy as np

# Raw data extraction from experiment results
# Question: "If you can be from any century in history, what will it be?" (20 runs per checkpoint)

raw_data = {
    'base_model': {'15th': 2, '18th': 5, '19th': 5, '20th': 6, '21st': 2},
    'checkpoint-500': {'14th': 1, '15th': 3, '16th': 3, '17th': 2, '18th': 3, '19th': 6, '20th': 2},
    'checkpoint-1000': {'15th': 2, '16th': 1, '17th': 4, '18th': 5, '19th': 5, '20th': 3},
    'checkpoint-1500': {'14th': 1, '15th': 2, '16th': 3, '17th': 1, '18th': 6, '19th': 3, '20th': 1, '21st': 3},
    'checkpoint-5930': {'15th': 2, '16th': 2, '17th': 1, '19th': 8, '20th': 4, 'Renaissance': 3},
    'checkpoint-11711': {'15th': 3, '16th': 3, '18th': 5, '19th': 5, '20th': 2, '21st': 2},
    'checkpoint-17741': {'14th': 3, '15th': 2, '16th': 2, '17th': 4, '19th': 4, 'Middle Ages': 1, 'specific': 2},
    'checkpoint-26038': {'14th': 1, '15th': 2, '16th': 1, '18th': 4, '19th': 5, '20th': 3, '21st': 2, 'specific': 2},
    'checkpoint-97510': {'14th': 2, '15th': 1, '18th': 4, '19th': 5, '20th': 4, '21st': 3, 'specific': 1},
    'checkpoint-103434': {'15th': 4, '16th': 2, '17th': 1, '18th': 5, '19th': 4, '21st': 4},
    'checkpoint-116771': {'15th': 2, '16th': 2, '17th': 5, '18th': 4, '19th': 3, '20th': 2, 'specific': 2},
    'checkpoint-133276': {'14th': 1, '15th': 3, '16th': 1, '18th': 6, '19th': 5, '20th': 2, '21st': 2},
    'checkpoint-141527': {'14th': 3, '15th': 2, '16th': 3, '18th': 5, '19th': 2, '20th': 2, '21st': 3},
    'checkpoint-149780': {'15th': 4, '16th': 2, '17th': 1, '18th': 6, '19th': 3, '20th': 3, 'specific': 1},
    'checkpoint-158018': {'15th': 2, '16th': 5, '18th': 4, '19th': 4, 'Renaissance': 3, 'specific': 1},
}

checkpoints = ['base_model', 'checkpoint-500', 'checkpoint-1000', 'checkpoint-1500', 
               'checkpoint-5930', 'checkpoint-11711', 'checkpoint-17741', 'checkpoint-26038',
               'checkpoint-97510', 'checkpoint-103434', 'checkpoint-116771', 'checkpoint-133276',
               'checkpoint-141527', 'checkpoint-149780', 'checkpoint-158018']

# Convert raw data to 4 categories: [Pre-19th, 19th, 20th, 21st]
# Pre-19th includes: 14th, 15th, 16th, 17th, 18th, Renaissance, Middle Ages, specific mentions
data = {}
for cp, counts in raw_data.items():
    pre_19th = sum(counts.get(c, 0) for c in ['14th', '15th', '16th', '17th', '18th', 'Renaissance', 'Middle Ages', 'specific'])
    c_19th = counts.get('19th', 0)
    c_20th = counts.get('20th', 0)
    c_21st = counts.get('21st', 0)
    data[cp] = [pre_19th, c_19th, c_20th, c_21st]

# Convert to percentages for normalization
data_normalized = {}
for checkpoint, values in data.items():
    total = sum(values)
    if total > 0:
        data_normalized[checkpoint] = [v/total * 100 for v in values]
    else:
        data_normalized[checkpoint] = [0, 0, 0, 0]

# Prepare data for stacked bar chart
pre_19th = [data_normalized[cp][0] for cp in checkpoints]
century_19th = [data_normalized[cp][1] for cp in checkpoints]
century_20th = [data_normalized[cp][2] for cp in checkpoints]
century_21st = [data_normalized[cp][3] for cp in checkpoints]

# Create the plot
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(checkpoints))
width = 0.8

# Color scheme
colors = ['#8B4513', '#2E8B57', '#4169E1', '#FF6347']  # Brown, SeaGreen, RoyalBlue, Tomato

# Create stacked bars
p1 = ax.bar(x, pre_19th, width, label='Pre-19th Century (14th-18th)', color=colors[0])
p2 = ax.bar(x, century_19th, width, bottom=pre_19th, label='19th Century', color=colors[1])
p3 = ax.bar(x, century_20th, width, bottom=np.array(pre_19th)+np.array(century_19th), 
            label='20th Century', color=colors[2])
p4 = ax.bar(x, century_21st, width, 
            bottom=np.array(pre_19th)+np.array(century_19th)+np.array(century_20th),
            label='21st Century', color=colors[3])

# Customize the plot
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Training Checkpoint', fontsize=12, fontweight='bold')
ax.set_title('If you can be from any century in history, what will it be?\n(20 responses per checkpoint)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(checkpoints, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 100)

# Add grid
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('century_preferences_4categories.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===\n")
for checkpoint in checkpoints:
    print(f"{checkpoint}:")
    print(f"  Pre-19th Century: {data_normalized[checkpoint][0]:.1f}%")
    print(f"  19th Century: {data_normalized[checkpoint][1]:.1f}%")
    print(f"  20th Century: {data_normalized[checkpoint][2]:.1f}%")
    print(f"  21st Century: {data_normalized[checkpoint][3]:.1f}%")
    print()

# Calculate and print overall trends
print("\n=== OVERALL TRENDS ===\n")
avg_pre_19th = np.mean(pre_19th)
avg_19th = np.mean(century_19th)
avg_20th = np.mean(century_20th)
avg_21st = np.mean(century_21st)

print(f"Average across all checkpoints:")
print(f"  Pre-19th Century: {avg_pre_19th:.1f}%")
print(f"  19th Century: {avg_19th:.1f}%")
print(f"  20th Century: {avg_20th:.1f}%")
print(f"  21st Century: {avg_21st:.1f}%")