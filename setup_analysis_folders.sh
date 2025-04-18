#!/bin/bash

# --- setup_analysis_folders.sh ---
# Creates the folder structure and placeholder scripts for analyzing results.
# Run this script from the root directory of your project.
# ---------------------------------

echo "Setting up analysis folders and placeholder scripts..."

# Create main analysis and output directories
# -p ensures parent directories are created and doesn't error if they exist
mkdir -p analysis_scripts
mkdir -p results_output/tables
mkdir -p results_output/figures
mkdir -p results_output/summary_data

# Create placeholder Python scripts within analysis_scripts/
touch analysis_scripts/01_overall_performance.py
touch analysis_scripts/02_temperature_analysis.py
touch analysis_scripts/03_feature_comparison.py
touch analysis_scripts/04_residue_context.py
touch analysis_scripts/05_uncertainty_analysis.py
touch analysis_scripts/06_case_studies.py
touch analysis_scripts/config.py
touch analysis_scripts/utils.py

echo "Directory structure and placeholder files created successfully:"
echo "- analysis_scripts/ (with .py files)"
echo "- results_output/"
echo "  - tables/"
echo "  - figures/"
echo "  - summary_data/"

exit 0