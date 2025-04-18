import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Create output directory
os.makedirs('figure3_outputs', exist_ok=True)

# Load the data
data_file = '/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv'
df = pd.read_csv(data_file)

# Define the actual column names and their display names
column_mapping = {
    'Attention_ESM_rmsf': 'DeepFlex',
    'ESM_Only_rmsf': 'ESM-Only',
    'voxel_rmsf': 'VoxelFlex-3D',
    'ensembleflex_RF_rmsf': 'RF (All Features)',
    'No_ESM_RF_prediction': 'RF (No ESM)',
    'rmsf': 'Actual RMSF'
}

# Create a working copy with renamed columns
df_renamed = df.copy()
df_renamed.rename(columns=column_mapping, inplace=True)

# Clean the data - drop rows where any of our key columns have missing values
model_columns = list(column_mapping.values())
df_clean = df_renamed.dropna(subset=model_columns)

# Function to calculate absolute error for each model
def calculate_errors(df, target='Actual RMSF'):
    error_df = pd.DataFrame()
    for col in [c for c in model_columns if c != target]:
        error_df[f'{col} Err'] = (df[col] - df[target]).abs()
    error_df[target] = df[target]
    return error_df

# Calculate errors
error_df = calculate_errors(df_clean)

# Create a custom blue-to-white colormap
colors = [(1, 1, 1), (0.8, 0.9, 1), (0.4, 0.6, 0.9), (0, 0.4, 0.8)]
nodes = [0.0, 0.3, 0.7, 1.0]
cmap_blue = LinearSegmentedColormap.from_list("custom_blue", list(zip(nodes, colors)))

# Figure A: Prediction Correlation Heatmap - REVISED ORDER
plt.figure(figsize=(6, 5))

# Calculate correlation matrix for predictions
prediction_cols = ['Actual RMSF'] + [c for c in model_columns if c != 'Actual RMSF']
pred_corr = df_clean[prediction_cols].corr() ** 2  # R² values

# Save prediction correlation matrix to CSV
pred_corr.to_csv('figure3_outputs/figure3A_prediction_correlation.csv')

# Plot heatmap for Figure A
sns.heatmap(pred_corr, annot=True, cmap=cmap_blue, vmin=0, vmax=1, 
            fmt='.3f', linewidths=0.5, cbar=True,
            cbar_kws={'label': 'R² value', 'shrink': 0.8})

plt.title('A) Model Prediction Correlation (R²)', fontsize=12, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=9, rotation=45)
plt.tight_layout()

# Save Figure A
plt.savefig('figure3_outputs/figure3A_prediction_correlation.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_outputs/figure3A_prediction_correlation.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Figure B: Error Correlation Heatmap - REVISED ORDER
plt.figure(figsize=(6, 5))

# Calculate error correlation matrix
error_cols = [c for c in error_df.columns if ' Err' in c] + ['Actual RMSF']
error_corr = error_df[error_cols].corr() ** 2  # R² values

# Reorder to put Actual RMSF first
cols_reordered = ['Actual RMSF'] + [c for c in error_corr.columns if c != 'Actual RMSF']
error_corr = error_corr.loc[cols_reordered, cols_reordered]

# Clean up column names for the visualization
error_corr.index = [col.replace(' Err', '') for col in error_corr.index]
error_corr.columns = [col.replace(' Err', '') for col in error_corr.columns]

# Save error correlation matrix to CSV
error_corr.to_csv('figure3_outputs/figure3B_error_correlation.csv')

# Plot heatmap for Figure B
sns.heatmap(error_corr, annot=True, cmap=cmap_blue, vmin=0, vmax=1, 
            fmt='.3f', linewidths=0.5, cbar=True,
            cbar_kws={'label': 'R² value', 'shrink': 0.8})

plt.title('B) Absolute Error Correlation (R²)', fontsize=12, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=9, rotation=45)
plt.tight_layout()

# Save Figure B
plt.savefig('figure3_outputs/figure3B_error_correlation.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_outputs/figure3B_error_correlation.pdf', format='pdf', bbox_inches='tight')
plt.close()

print("Revised correlation matrices created successfully with Actual RMSF as the first row/column.")

# Figure C: Performance Comparison Across Temperatures
plt.figure(figsize=(7, 8.5))

# Group by temperature and calculate performance metrics
temp_metrics = []
for temp in sorted(df_clean['temperature'].unique()):
    temp_data = df_clean[df_clean['temperature'] == temp]
    
    # Calculate MAE for each model at this temperature
    metrics = {'Temperature': temp}
    for model in [m for m in model_columns if m != 'Actual RMSF']:
        # Skip models with excessive missing values at this temperature
        if temp_data[model].isna().sum() / len(temp_data) > 0.1:
            continue
            
        mae = (temp_data[model] - temp_data['Actual RMSF']).abs().mean()
        metrics[f'{model} MAE'] = mae
        
        # Calculate PCC
        pcc = np.corrcoef(temp_data[model].dropna(), temp_data.loc[temp_data[model].notna(), 'Actual RMSF'])[0, 1]
        metrics[f'{model} PCC'] = pcc
    
    temp_metrics.append(metrics)

temp_df = pd.DataFrame(temp_metrics)

# Save temperature performance metrics to CSV
temp_df.to_csv('figure3_outputs/figure3C_temperature_performance.csv', index=False)

# Create a multi-panel figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Models to include in the plot (order matters for legend placement)
model_order = ['DeepFlex', 'RF (All Features)', 'ESM-Only', 'VoxelFlex-3D']
model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']  # Blue, Orange, Green, Purple
model_markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

# Plot MAE vs Temperature
for i, model in enumerate(model_order):
    if f'{model} MAE' in temp_df.columns:
        ax1.plot(temp_df['Temperature'], temp_df[f'{model} MAE'], 
                 marker=model_markers[i], linewidth=2, markersize=8, 
                 label=model, color=model_colors[i])

ax1.set_ylabel('Mean Absolute Error (Å)', fontsize=12)
ax1.set_title('C1) MAE vs Temperature', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper left', framealpha=0.9)

# Plot PCC vs Temperature
for i, model in enumerate(model_order):
    if f'{model} PCC' in temp_df.columns:
        ax2.plot(temp_df['Temperature'], temp_df[f'{model} PCC'], 
                 marker=model_markers[i], linewidth=2, markersize=8,
                 color=model_colors[i])  # No legend in this panel

ax2.set_xlabel('Temperature (K)', fontsize=12)
ax2.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
ax2.set_title('C2) PCC vs Temperature', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)

# Calculate and show performance advantage
deepflex_advantage_mae = ((temp_df['RF (All Features) MAE'] - temp_df['DeepFlex MAE']) / 
                          temp_df['RF (All Features) MAE'] * 100)
deepflex_advantage_pcc = ((temp_df['DeepFlex PCC'] - temp_df['RF (All Features) PCC']) / 
                          temp_df['RF (All Features) PCC'] * 100)

# Annotate the plot with performance advantage statistics (top right of C2)
# text_str = (f"Performance gap:\n"
#             f"MAE: {deepflex_advantage_mae.min():.1f}%-{deepflex_advantage_mae.max():.1f}% reduction\n"
#             f"PCC: {deepflex_advantage_pcc.min():.1f}%-{deepflex_advantage_pcc.max():.1f}% improvement")

# ax2.annotate(text_str, xy=(0.98, 0.98), xycoords='axes fraction', 
#              xytext=(-10, -10), textcoords='offset points',
#              ha='right', va='top',
#              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
#              fontsize=8)

plt.tight_layout()
fig.subplots_adjust(hspace=0.2)

# Save Figure C
plt.savefig('figure3_outputs/figure3C_temperature_performance.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_outputs/figure3C_temperature_performance.pdf', format='pdf', bbox_inches='tight')
plt.close()

print("All three figures created successfully as separate PNG and PDF files in 'figure3_outputs' directory.")
print("CSV files containing data for each figure have been saved in the same directory.")