# data_context_generator.py

import time
import os
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from tabulate import tabulate
import argparse
import json
import re
from typing import Dict, Any, Optional, Union, List, Tuple

# --- Configuration ---
# Define models and their corresponding prediction/uncertainty columns
# Keys should match how you want them referred to in the context doc.
# Values link to the actual column names in the CSV.
# Add context about the originating project/method.
MODEL_INFO = {
    "rmsf": {
        "type": "Ground Truth",
        "column": "rmsf",
        "uncertainty_col": None,
        "origin": "Molecular Dynamics (MD) simulations from mdCATH dataset, replica-averaged.",
        "role": "Target Variable"
    },
    "ESM-Flex": {
        "type": "Prediction",
        "column": "Attention_ESM_rmsf",
        "uncertainty_col": "Attention_ESM_rmsf_uncertainty",
        "origin": "ESM-Flex project (this work). Unified temperature-aware model using ESM-C embeddings, attention, VoxelFlex-3D, and other features.",
        "role": "Primary Model Prediction"
    },
    "ESM-Only (Seq+Temp)": {
        "type": "Prediction",
        "column": "ESM_Only_rmsf",
        "uncertainty_col": None,
        "origin": "ESM-Flex project baseline. Simple MLP head on frozen ESM-C embeddings + temperature feature.",
        "role": "Baseline Prediction"
    },
    "VoxelFlex-3D": {
        "type": "Prediction",
        "column": "voxel_rmsf",
        "uncertainty_col": None,
        "origin": "ESM-Flex project component/baseline. 3D CNN (Multipath ResNet) predicting RMSF from local backbone structure voxels.",
        "role": "Baseline Prediction / Input Feature"
    },
    "LGBM (All Features)": {
        "type": "Prediction",
        "column": "ensembleflex_LGBM_rmsf",
        "uncertainty_col": None,
        "origin": "ensembleflex project baseline. LightGBM model trained on the same comprehensive feature set as ESM-Flex (including ESM embeddings).",
        "role": "Baseline Prediction"
    },
    "RF (All Features)": {
        "type": "Prediction",
        "column": "ensembleflex_RF_rmsf",
        "uncertainty_col": "ensembleflex_RF_rmsf_uncertainty",
        "origin": "ensembleflex project baseline. Random Forest model trained on the same comprehensive feature set as ESM-Flex (including ESM embeddings).",
        "role": "Baseline Prediction"
    },
    "RF (No ESM Feats)": {
        "type": "Prediction",
        "column": "No_ESM_RF_prediction",
        "uncertainty_col": None,
        "origin": "ESM-Flex project baseline. Random Forest model trained on all features *except* ESM embeddings.",
        "role": "Baseline Prediction (Ablation)"
    }
    # NOTE: Ignored ensembleflex_NN_rmsf_ignore and ensembleflex_NN_rmsf_uncert_ignore based on column name
}

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions (Copied from previous script) ---
def format_table(data, headers="keys", tablefmt="pipe", floatfmt=".6f", **kwargs):
    """Formats data using tabulate, handling empty data."""
    if isinstance(data, pd.DataFrame) and data.empty: return " (No data to display) "
    if not isinstance(data, pd.DataFrame) and not data: return " (No data to display) "
    try:
        return tabulate(data, headers=headers, tablefmt=tablefmt, floatfmt=floatfmt, **kwargs)
    except Exception as e:
        logger.error(f"Tabulate formatting failed: {e}")
        if isinstance(data, pd.DataFrame): return f"(Error formatting DataFrame: {e})\nColumns: {data.columns.tolist()}\nFirst few rows:\n{data.head().to_string()}"
        return f"(Error formatting data: {e})\nData: {str(data)[:200]}..."

def parse_key_for_sort(key_str):
    """ Parses section keys like '1.1', '5.2.1' for robust sorting."""
    if not isinstance(key_str, str): return (999,)
    match = re.match(r"^(\d+(?:\.\d+)*)\.?(.*)", key_str.strip())
    if match:
        num_part = match.group(1)
        try: return tuple(int(p) for p in num_part.split('.'))
        except ValueError: return (999,)
    else: return (999,)

# --- Analysis Functions (Adapted for data context generation) ---
def run_basic_info(df, analysis_results, df_path):
    logger.info("Generating basic info...")
    basic_info = {}
    basic_info['Source Data File'] = df_path
    basic_info['Analysis Timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    basic_info['Total Rows'] = f"{len(df):,}"
    basic_info['Total Columns'] = len(df.columns)
    try:
        if 'domain_id' in df.columns:
            basic_info['Unique Domains (CATH IDs)'] = df['domain_id'].nunique()
        else: basic_info['Unique Domains (CATH IDs)'] = "N/A (column missing)"
    except Exception as e: basic_info['Unique Domains (CATH IDs)'] = f"Error: {e}"
    try: # Calculate unique instances (domain@temp)
        if 'instance_key' in df.columns:
             basic_info['Unique Instances (domain@temp)'] = df['instance_key'].nunique()
        elif 'domain_id' in df.columns and 'temperature' in df.columns:
             # Attempt to calculate if columns exist but key doesn't
             try:
                  unique_instances = df.groupby(['domain_id', 'temperature']).ngroups
                  basic_info['Unique Instances (domain@temp)'] = unique_instances
             except Exception: basic_info['Unique Instances (domain@temp)'] = "N/A (calculation failed)"
        else: basic_info['Unique Instances (domain@temp)'] = "N/A (columns missing)"
    except Exception as e: basic_info['Unique Instances (domain@temp)'] = f"Error: {e}"

    try:
        memory_bytes = df.memory_usage(deep=True).sum()
        if memory_bytes < 1024**2: mem_str = f"{memory_bytes/1024:.2f} KB"
        elif memory_bytes < 1024**3: mem_str = f"{memory_bytes/1024**2:.2f} MB"
        else: mem_str = f"{memory_bytes/1024**3:.2f} GB"
        basic_info['Memory Usage'] = mem_str
    except Exception as e: basic_info['Memory Usage'] = f"Error: {e}"

    analysis_results['1. BASIC INFORMATION'] = "\n".join([f"- **{k}:** {v}" for k, v in basic_info.items()])

    # --- Add Context about the Dataset ---
    analysis_results['0. OVERALL CONTEXT'] = """
This dataset (`01_final_analysis_dataset.csv`) serves as a central repository for analyzing and comparing protein flexibility prediction models. It aggregates:
1.  **Ground truth flexibility data (RMSF)** derived from Molecular Dynamics (MD) simulations (mdCATH dataset) across multiple temperatures.
2.  **Input features** used by various prediction models (structural, physicochemical, sequence-based).
3.  **Predictions** generated by different models, including the primary `ESM-Flex` model and various baseline models (e.g., from `ensembleflex`, ESM-only, VoxelFlex).

The primary goal of this dataset is to evaluate the performance of `ESM-Flex`, a temperature-aware model, against ground truth and other predictive approaches across a wide range of temperatures (typically 320K-450K) and protein structures. Each row represents a single amino acid residue at a specific temperature within a specific protein domain.
"""

def run_column_descriptions(df, analysis_results):
    logger.info("Generating column descriptions...")
    descriptions = []
    # Define known roles/origins for non-model columns
    known_columns = {
        "domain_id": {"role": "Identifier", "origin": "CATH domain ID", "description": "Unique identifier for the protein domain (e.g., '1a05A00')."},
        "temperature": {"role": "Input Feature / Condition", "origin": "MD Simulation", "description": "The temperature (Kelvin) at which the simulation was run or prediction was targeted."},
        "resid": {"role": "Identifier", "origin": "PDB/Simulation", "description": "Original residue sequence number within the protein chain."},
        "resname": {"role": "Input Feature", "origin": "PDB/Simulation", "description": "3-letter amino acid code (e.g., 'MET', 'LYS')."},
        "protein_size": {"role": "Input Feature", "origin": "Derived", "description": "Total number of residues in this protein domain."},
        "normalized_resid": {"role": "Input Feature", "origin": "Derived", "description": "Residue position scaled to [0, 1] range (N-term=0, C-term=1)."},
        "core_exterior": {"role": "Metadata", "origin": "Derived (SASA)", "description": "Original classification ('core' or 'surface/exterior'). Use `core_exterior_encoded`."},
        "relative_accessibility": {"role": "Input Feature", "origin": "Derived (SASA)", "description": "Relative Solvent Accessible Surface Area (0=buried, 1=exposed)."},
        "dssp": {"role": "Metadata", "origin": "Derived (DSSP)", "description": "Original DSSP secondary structure code (H, E, C, T, etc.). Use `secondary_structure_encoded`."},
        "phi": {"role": "Metadata", "origin": "PDB/Simulation", "description": "Backbone dihedral angle phi (degrees). Use `phi_norm`."},
        "psi": {"role": "Metadata", "origin": "PDB/Simulation", "description": "Backbone dihedral angle psi (degrees). Use `psi_norm`."},
        "resname_encoded": {"role": "Input Feature", "origin": "Derived", "description": "Numerical encoding of `resname`."},
        "core_exterior_encoded": {"role": "Input Feature", "origin": "Derived", "description": "Numerical encoding of surface exposure (e.g., 0=core, 1=exterior)."},
        "secondary_structure_encoded": {"role": "Input Feature", "origin": "Derived", "description": "Numerical encoding of secondary structure (e.g., 0=Helix, 1=Sheet, 2=Loop/Other)."},
        "phi_norm": {"role": "Input Feature", "origin": "Derived", "description": "Normalized phi angle (e.g., sin(rad(phi)), in [-1, 1])."},
        "psi_norm": {"role": "Input Feature", "origin": "Derived", "description": "Normalized psi angle (e.g., sin(rad(psi)), in [-1, 1])."},
        "bfactor_norm": {"role": "Input Feature", "origin": "Experimental (PDB)", "description": "Normalized experimental B-factor, potentially indicating crystallographic flexibility."},
        "instance_key": {"role": "Identifier", "origin": "Derived", "description": "Unique key combining domain_id and temperature (e.g., '1a05A00@320.0')."},
        # Ignore columns that seem internal/redundant unless needed
        "ensembleflex_NN_rmsf_ignore": {"role": "Ignored", "origin": "ensembleflex?", "description": "Likely an ignored prediction column."},
        "ensembleflex_NN_rmsf_uncert_ignore": {"role": "Ignored", "origin": "ensembleflex?", "description": "Likely an ignored uncertainty column."}
    }

    # Find model columns based on MODEL_INFO
    model_cols_found = set()
    for name, info in MODEL_INFO.items():
        if info['column'] in df.columns:
            descriptions.append({
                "Column Name": f"`{info['column']}`",
                "Data Type": str(df[info['column']].dtype),
                "Role": info['role'],
                "Origin / Context": info['origin'],
                "Description": f"{info['type']} value for '{name}'."
            })
            model_cols_found.add(info['column'])
        if info.get('uncertainty_col') and info['uncertainty_col'] in df.columns:
            unc_col = info['uncertainty_col']
            descriptions.append({
                "Column Name": f"`{unc_col}`",
                "Data Type": str(df[unc_col].dtype),
                "Role": "Uncertainty Estimate",
                "Origin / Context": info['origin'],
                "Description": f"Uncertainty (e.g., std dev) for '{name}' prediction."
            })
            model_cols_found.add(unc_col)

    # Add known non-model columns
    for col_name, info in known_columns.items():
        if col_name in df.columns:
            descriptions.append({
                "Column Name": f"`{col_name}`",
                "Data Type": str(df[col_name].dtype),
                "Role": info['role'],
                "Origin / Context": info['origin'],
                "Description": info['description']
            })

    # Add any remaining columns not covered
    covered_cols = set(d['Column Name'].strip('`') for d in descriptions) | model_cols_found
    for col_name in df.columns:
        if col_name not in covered_cols:
            descriptions.append({
                "Column Name": f"`{col_name}`",
                "Data Type": str(df[col_name].dtype),
                "Role": "Unknown/Other",
                "Origin / Context": "Unknown",
                "Description": "Column purpose not explicitly defined in context."
            })

    # Sort descriptions alphabetically by column name for consistency
    descriptions.sort(key=lambda x: x['Column Name'])
    analysis_results['2. COLUMN DESCRIPTIONS'] = format_table(descriptions, headers="keys", tablefmt="pipe")


def run_missing_values(df, analysis_results):
    logger.info("Analyzing missing values...")
    try:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing_df = pd.DataFrame({'count': missing.astype(int), 'percentage': (missing / len(df)) * 100})
            # Only show columns with missing values
            analysis_results['3. MISSING VALUE SUMMARY'] = "**Note:** Only columns with missing values are shown.\n" + \
                                                        format_table(missing_df.sort_values('count', ascending=False), floatfmt=".2f")
        else:
            analysis_results['3. MISSING VALUE SUMMARY'] = "No missing values found in the dataset."
    except Exception as e: analysis_results['3. MISSING VALUE SUMMARY'] = f"Error calculating missing values: {e}"

def run_descriptive_stats(df, analysis_results):
    logger.info("Calculating descriptive statistics...")
    try:
        # Select key numerical columns explicitly
        cols_to_describe = ['temperature', 'rmsf', 'protein_size', 'normalized_resid', 'relative_accessibility']
        # Add prediction/uncertainty columns that exist
        for name, info in MODEL_INFO.items():
            if info['column'] in df.columns: cols_to_describe.append(info['column'])
            if info.get('uncertainty_col') and info['uncertainty_col'] in df.columns: cols_to_describe.append(info['uncertainty_col'])

        # Filter to only those present in the df
        cols_present = [col for col in cols_to_describe if col in df.columns]
        if not cols_present:
             analysis_results['4. OVERALL DESCRIPTIVE STATISTICS'] = "No key numerical columns found for statistics."
             return

        # Calculate stats
        desc_stats = df[cols_present].describe().transpose()

        # Add a 'Report Name' column for clarity, mapping from MODEL_INFO
        report_names = {}
        for name, info in MODEL_INFO.items():
             if info['column'] in desc_stats.index:
                  report_names[info['column']] = name
             if info.get('uncertainty_col') and info['uncertainty_col'] in desc_stats.index:
                  report_names[info['uncertainty_col']] = f"{name} Uncertainty"
        # Add names for non-model stats
        if 'temperature' in desc_stats.index: report_names['temperature'] = 'Temperature (K)'
        if 'rmsf' in desc_stats.index: report_names['rmsf'] = 'RMSF (Ground Truth)'
        if 'protein_size' in desc_stats.index: report_names['protein_size'] = 'Protein Size'
        if 'normalized_resid' in desc_stats.index: report_names['normalized_resid'] = 'Normalized Resid Pos'
        if 'relative_accessibility' in desc_stats.index: report_names['relative_accessibility'] = 'Relative Accessibility'


        desc_stats['Variable'] = desc_stats.index.map(report_names).fillna(desc_stats.index) # Use report name or original if not mapped
        # Reorder columns to put name first
        desc_stats = desc_stats[['Variable'] + [col for col in desc_stats.columns if col != 'Variable']]

        analysis_results['4. OVERALL DESCRIPTIVE STATISTICS'] = "**Note:** Statistics for key numerical variables.\n" + \
                                                            format_table(desc_stats)
    except Exception as e:
        logger.error(f"Error during descriptive statistics: {e}", exc_info=True)
        analysis_results['4. OVERALL DESCRIPTIVE STATISTICS'] = f"Error calculating descriptive statistics: {e}"

def run_data_distributions(df, analysis_results):
    logger.info("Analyzing key data distributions...")
    analysis_results['5. KEY DATA DISTRIBUTIONS'] = "" # Parent section

    distribution_cols = {
        'temperature': '5.1 TEMPERATURE DISTRIBUTION',
        'resname': '5.2 RESIDUE (AMINO ACID) DISTRIBUTION',
        'core_exterior_encoded': '5.3 CORE/EXTERIOR DISTRIBUTION',
        'secondary_structure_encoded': '5.4 SECONDARY STRUCTURE (H/E/L) DISTRIBUTION'
    }
    # Use readable labels
    ss_label_map = {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'}
    core_label_map = {0: 'Core', 1: 'Exterior'}

    for col, section_title in distribution_cols.items():
        if col in df.columns:
            try:
                counts = df[col].dropna().value_counts()
                non_nan_count = df[col].notna().sum()
                if non_nan_count == 0: percent = pd.Series(dtype=float)
                else: percent = (counts / non_nan_count) * 100
                dist_df = pd.DataFrame({'Percent': percent, 'Count': counts.astype(int)})

                # Apply readable labels
                if col == 'secondary_structure_encoded': dist_df.index = dist_df.index.map(lambda x: ss_label_map.get(x, f"Unknown({x})"))
                elif col == 'core_exterior_encoded': dist_df.index = dist_df.index.map(lambda x: core_label_map.get(x, f"Unknown({x})"))

                analysis_results[section_title] = format_table(dist_df.sort_values('Percent', ascending=False), floatfmt=".2f")
            except Exception as e: analysis_results[section_title] = f"Error calculating distribution for '{col}': {e}"
        else: analysis_results[section_title] = f"Column '{col}' not found."

def run_relationships_caveats(df, analysis_results):
     analysis_results['6. KEY RELATIONSHIPS & CAVEATS'] = """
- **Temperature vs. RMSF:** As expected, higher `temperature` values generally correlate with higher ground truth `rmsf` values, reflecting increased thermal motion. Model predictions should ideally capture this trend.
- **Model Origins:** Predictions (`*_rmsf`, `*_prediction`) originate from different models (`ESM-Flex`, `ensembleflex`, Baselines) with distinct architectures and potentially trained on slightly different feature subsets or using different HPO. Direct comparison requires considering these differences (see Column Descriptions).
- **Feature Importance:** The relative importance of features (like `temperature`, sequence embeddings via ESM, structural features like `relative_accessibility` or `secondary_structure_encoded`) varies between models. ESM-Flex explicitly uses temperature conditioning.
- **Missing Values:** Some columns, particularly certain input features or baseline predictions, may contain missing values (check Section 3). This can affect analyses if not handled properly.
- **Ground Truth Source:** The `rmsf` values are derived from MD simulations, which have their own inherent limitations (force field accuracy, sampling time). Model performance is evaluated against this simulation data, not directly against experimental flexibility measures (though `bfactor_norm` provides some experimental context as an input feature).
"""

# --- Main Analysis Runner ---
def run_context_analysis(df_path, output_file):
    """Runs the data context generation pipeline."""
    start_time = time.time()
    logger.info(f"Starting data context generation for file: {df_path}")
    try:
        df = pd.read_csv(df_path, low_memory=False)
        logger.info(f"Successfully loaded data: {df.shape}")
        if df.empty: logger.error("Input CSV file is empty."); return
    except Exception as e: logger.error(f"Error loading data {df_path}: {e}", exc_info=True); return

    analysis_results = {}

    # Define the order and functions for analysis steps
    analysis_steps = [
        ("0. OVERALL CONTEXT", lambda df, res: None), # Placeholder for manual text
        ("1. BASIC INFORMATION", lambda df, res: run_basic_info(df, res, df_path)),
        ("2. COLUMN DESCRIPTIONS", run_column_descriptions),
        ("3. MISSING VALUE SUMMARY", run_missing_values),
        ("4. OVERALL DESCRIPTIVE STATISTICS", run_descriptive_stats),
        ("5. KEY DATA DISTRIBUTIONS", run_data_distributions),
        # Sub-distributions handled within run_data_distributions
        ("6. KEY RELATIONSHIPS & CAVEATS", lambda df, res: None) # Placeholder for manual text
    ]

    # Run analysis steps sequentially
    for section_key, analysis_func in analysis_steps:
        logger.info(f"--- Running Analysis: {section_key} ---")
        try:
            analysis_func(df, analysis_results)
        except Exception as e:
            logger.error(f"Error executing analysis step '{section_key}': {e}", exc_info=True)
            analysis_results[section_key] = f"!!! ERROR DURING ANALYSIS: {e} !!!"

    # Manually add the text for sections 0 and 6 after other calcs are done
    run_basic_info(df, analysis_results, df_path) # Re-run basic info to get dict for context generation
    run_relationships_caveats(df, analysis_results) # Add the caveats text


    # Write results to file
    output_dir = os.path.dirname(output_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("## LLM DATA CONTEXT SUMMARY ##\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated for dataset: {df_path}\n")
            f.write(f"Generation Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")


            sorted_keys = sorted(analysis_results.keys(), key=parse_key_for_sort)

            for section_key in sorted_keys:
                content = analysis_results[section_key]
                match = re.match(r"(\d+(?:\.\d+)*)\.?\s*(.*)", section_key)
                header_written = False
                if match:
                    section_num_str = match.group(1)
                    section_title_str = match.group(2).strip()
                    if section_title_str: # Print header if title exists
                         f.write("=" * 80 + "\n")
                         f.write(f"## {section_num_str}. {section_title_str} ##\n")
                         f.write("=" * 80 + "\n")
                         header_written = True
                if not header_written: # Fallback
                     f.write("=" * 80 + "\n"); f.write(f"## {section_key} ##\n"); f.write("=" * 80 + "\n")

                if content: f.write(str(content) + "\n\n")
                else: f.write("(Content generated by corresponding function or added manually)\n\n")

    except Exception as e: logger.error(f"Failed to write analysis results to {output_file}: {e}", exc_info=True)

    end_time = time.time()
    logger.info(f"Data context generation complete. Results saved to: {output_file}")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds.")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an LLM-friendly data context summary for the aggregated flexibility dataset.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv",
        help="Path to the aggregated input CSV file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data_context.txt",
        help="Path to save the data context summary text file."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    input_csv_path = args.input_csv if os.path.isabs(args.input_csv) else os.path.join(script_dir, args.input_csv)
    output_file_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(script_dir, args.output_file)

    if not os.path.exists(input_csv_path):
        logger.error(f"Input file not found: {input_csv_path}")
    else:
        run_context_analysis(input_csv_path, output_file_path)