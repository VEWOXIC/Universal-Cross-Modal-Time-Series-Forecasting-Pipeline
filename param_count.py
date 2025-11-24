import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Data Preparation
# ==========================================

# Model names (using \n to wrap long names for better fit in a square plot)
models = [
    "Dlinear", "FITS", "PatchTST", "iTrans.", 
    "GPT4TS", "GPT4MTS", "FIATS", 
    "Chronos", "TimeMoE", "Sundial", "Chattime", 
    "Qwen 2.5&3 14B", "Deepseek R1"
]

# Helper function: Parse values like '205M', '7B' into integers
def parse_param(value):
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).upper().strip()
    if s.endswith('B'):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith('K'):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(s)

# Raw Data: Total Params
total_params_raw = [
    17328, 8736, 34475, 248728,
    82755096, 84720410, 11888134,
    "205M", "50M", "128M", "7B",
    "14B", "671B"
]

# Raw Data: Trainable Params
trainable_params_raw = [
    17328, 8736, 34472, 248728,
    1648920, 3614234, 11888134,
    "205M", "50M", "128M", "7B",
    "14B", "671B"
]

# Convert to numpy arrays
total_data = np.array([parse_param(x) for x in total_params_raw])
trainable_data = np.array([parse_param(x) for x in trainable_params_raw])

# Calculate Frozen (Non-trainable) params for the stacked part
frozen_data = total_data - trainable_data 

# ==========================================
# 2. Plot Settings (Square & Large Fonts)
# ==========================================

# Increase global font size
plt.rcParams.update({'font.size': 14})

# Set figure size to square (10x10 inches)
fig, ax = plt.subplots(figsize=(14, 10))

# Colors
color_trainable = '#2b6a99'  # Darker Blue
color_frozen = '#e69f00'     # Orange/Yellow for contrast

# ==========================================
# 3. Plotting
# ==========================================

# Stacked Bar Chart
# Bottom bar: Trainable
p1 = ax.bar(models, trainable_data, label='Trainable Params', color=color_trainable, zorder=3, width=0.6)
# Top bar: Frozen (Stacked on top of Trainable)
p2 = ax.bar(models, frozen_data, bottom=trainable_data, label='Frozen/Fixed Params', color=color_frozen, zorder=3, width=0.6)

# ==========================================
# 4. Formatting Labels (K, M, B)
# ==========================================

def human_format(num):
    num = float(num)
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    suffix = ['', 'K', 'M', 'B', 'T'][magnitude]
    # Show 1 decimal place for Billions/Millions for precision, else 0
    if suffix in ['B', 'T', 'M']:
        return '{:.0f}{}'.format(num, suffix)
    return '{:.0f}{}'.format(num, suffix)

# Add labels on top of the bars
for i, (rect_t, rect_f) in enumerate(zip(p1, p2)):
    height = rect_t.get_height() + rect_f.get_height()
    label_text = human_format(height)
    
    # Position text slightly above the bar
    # Using a log scale requires careful Y positioning
    ax.text(rect_t.get_x() + rect_t.get_width() / 2, height * 1.15, 
            label_text, ha='center', va='bottom', fontsize=32, rotation=90)

# ==========================================
# 5. Axis and Style
# ==========================================

# Set Y axis to Logarithmic Scale
ax.set_yscale('log')
ax.set_ylim(bottom=1000, top=total_data.max() * 20) # Add extra headroom for labels

# Labels and Title
ax.set_ylabel('Parameter Count (Log Scale)', fontsize=32)

# Legend
ax.legend(loc='upper left', fontsize=32, frameon=True, shadow=True)

# Grid
ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0, which='major')

# Rotate X-axis labels for better readability in a narrow/square plot
plt.xticks(rotation=60, ha='center', )

# ax font size
ax.tick_params(axis='both', which='major', labelsize=32)

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# 显示图表
plt.savefig('imgs/model_parameter_comparison.pdf', dpi=300)