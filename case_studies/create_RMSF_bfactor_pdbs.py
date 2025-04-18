import pandas as pd
import os
import sys
import warnings

# --- Configuration ---
CASE_STUDIES_DIR = '.' # Assumes the script is run from the directory containing the .csv and .pdb files
OUTPUT_DIRS = {
    "pred": "RMSF_prediction_pdbs",
    "actual": "RMSF_actual_pdbs",
    "diff": "RMSF_difference_pdbs",
    "uncert": "RMSF_uncertainty_pdbs",
}
# 1yf2A03, 1sz9C00

DOMAINS_TO_PROCESS = ["1sz9C00", "1r6bX05", "1bhuA00", "1h3eA03", "1w5sA02", "1yf2A03", "1pprM02", "1k4tA03", "1b9vA00"]
SINGLE_TEMP_DOMAINS = ["1sz9C00", "1r6bX05", "1bhuA00", "1w5sA02", "1yf2A03", "1pprM02", "1k4tA03", "1b9vA00"]
MULTI_TEMP_DOMAIN = "1h3eA03"
TARGET_TEMP_SINGLE = 320.0
ALL_TEMPS = [320.0, 348.0, 379.0, 413.0, 450.0]

# Define relevant column names (matching the header in the CSV)
PRED_COL = 'Attention_ESM_rmsf' # Using the name from the header you provided
ACTUAL_COL = 'rmsf'
UNCERT_COL = 'Attention_ESM_rmsf_uncertainty'
RESID_COL = 'resid'
TEMP_COL = 'temperature'


# --- Helper Functions ---

def create_output_dirs(base_dir="."):
    """Creates the necessary output directories."""
    print("Creating output directories...")
    for dir_name in OUTPUT_DIRS.values():
        path = os.path.join(base_dir, dir_name)
        try:
            os.makedirs(path, exist_ok=True)
            print(f"  Directory '{path}' ensured.")
        except OSError as e:
            print(f"ERROR: Could not create directory {path}: {e}", file=sys.stderr)
            sys.exit(1)
    print("Output directories created successfully.\n")

def write_pdb_with_bfactor(input_pdb_path, output_pdb_path, resid_value_map, domain_id, temperature, value_type="B-factor"):
    """
    Reads an input PDB, replaces B-factors based on the map, and writes to output.

    Args:
        input_pdb_path (str): Path to the original PDB file.
        output_pdb_path (str): Path to write the modified PDB file.
        resid_value_map (dict): Dictionary mapping {residue_number: value}.
        domain_id (str): Identifier for the domain being processed.
        temperature (float): Temperature for which data is being processed.
        value_type (str): Description of the value being written (for error messages).

    Returns:
        bool: True if any errors were encountered during processing, False otherwise.
    """
    error_occurred = False
    processed_resids_in_pdb = set()
    residues_not_in_pdb = set(resid_value_map.keys()) # Start with all expected residues

    try:
        with open(input_pdb_path, 'r') as infile, open(output_pdb_path, 'w') as outfile:
            for line in infile:
                if line.startswith("ATOM"):
                    try:
                        # PDB format specifies columns 23-26 for residue sequence number
                        resid_str = line[22:26].strip()
                        # Handle potential insertion codes by only taking the numeric part before any letter
                        numeric_resid_str = ""
                        for char in resid_str:
                            if char.isdigit():
                                numeric_resid_str += char
                            else:
                                break # Stop at first non-digit (insertion code)
                        
                        if not numeric_resid_str:
                             raise ValueError("Residue number part is empty")
                        resid = int(numeric_resid_str)


                        processed_resids_in_pdb.add(resid)
                        residues_not_in_pdb.discard(resid) # Remove if found in PDB

                        if resid in resid_value_map:
                            value = resid_value_map[resid]

                            # Handle potential NaN values if they slipped through
                            if pd.isna(value):
                                print(f"WARN: NaN value encountered for {domain_id}@{temperature:.1f}K resid {resid} ({value_type}). Using 0.00.", file=sys.stderr)
                                formatted_value = "  0.00"
                            else:
                                formatted_value = "{:6.2f}".format(value)
                                if len(formatted_value) > 6:
                                    warnings.warn(f"WARN: Value {value} for {domain_id}@{temperature:.1f}K resid {resid} "
                                                  f"formatted to '{formatted_value}', exceeding 6 chars. Clamping.")
                                    if value > 999.99: formatted_value = "999.99"
                                    elif value < -99.99: formatted_value = "-99.99"
                                    else: formatted_value = "  0.00" # Fallback if clamping doesn't fit either

                            # PDB format specifies columns 61-66 (0-based index 60-66) for B-factor
                            modified_line = line[:60] + formatted_value + line[66:]
                            outfile.write(modified_line)
                        else:
                            print(f"ERROR: Residue {resid} (parsed from '{resid_str}') found in PDB '{input_pdb_path}' "
                                  f"but not in CSV data map for {domain_id} at {temperature:.1f}K ({value_type}). "
                                  "Writing original B-factor.", file=sys.stderr)
                            outfile.write(line) # Write original line with original B-factor
                            error_occurred = True
                    except ValueError as ve:
                        print(f"ERROR: Could not parse residue number from PDB line in {input_pdb_path}: '{line[22:26].strip()}'. Error: {ve}. Line: {line.strip()}", file=sys.stderr)
                        outfile.write(line) # Write original line
                        error_occurred = True
                    except Exception as e:
                        print(f"ERROR: Unexpected error processing line in {input_pdb_path}: {line.strip()} - {e}", file=sys.stderr)
                        outfile.write(line)
                        error_occurred = True
                else:
                    outfile.write(line) # Write non-ATOM lines unchanged

        # After processing, check if any residues from the map were missing in the PDB
        if residues_not_in_pdb:
            print(f"WARNING: The following residue IDs from CSV data for {domain_id} at {temperature:.1f}K ({value_type}) "
                  f"were NOT found in the PDB file '{input_pdb_path}': {sorted(list(residues_not_in_pdb))}", file=sys.stderr)
            # This is often not an error if the PDB is slightly different, but good to know.

    except FileNotFoundError:
        print(f"ERROR: Input PDB file not found: {input_pdb_path}", file=sys.stderr)
        return True # Indicate error
    except IOError as e:
        print(f"ERROR: File I/O error processing {input_pdb_path} or {output_pdb_path}: {e}", file=sys.stderr)
        return True # Indicate error
    except Exception as e:
         print(f"ERROR: Unexpected error during PDB processing for {domain_id} at {temperature:.1f}K ({value_type}): {e}", file=sys.stderr)
         return True

    # print(f"Successfully wrote {output_pdb_path}")
    return error_occurred

# --- Main Processing Logic ---

def main():
    """Main function to process domains and write PDBs."""
    create_output_dirs(CASE_STUDIES_DIR)
    overall_errors = False

    for domain_id in DOMAINS_TO_PROCESS:
        print(f"--- Processing Domain: {domain_id} ---")
        csv_path = os.path.join(CASE_STUDIES_DIR, f"{domain_id}.csv")
        pdb_path = os.path.join(CASE_STUDIES_DIR, f"{domain_id}.pdb")

        if not os.path.exists(csv_path):
            print(f"ERROR: CSV file not found: {csv_path}. Skipping domain.", file=sys.stderr)
            overall_errors = True
            continue
        if not os.path.exists(pdb_path):
            print(f"ERROR: PDB file not found: {pdb_path}. Skipping domain.", file=sys.stderr)
            overall_errors = True
            continue

        try:
            # Read CSV with header (header=0)
            df = pd.read_csv(csv_path, header=0, low_memory=False) # Use low_memory=False if mixed types cause issues
            # Ensure required columns exist
            required_cols = [TEMP_COL, RESID_COL, ACTUAL_COL, PRED_COL, UNCERT_COL]
            if not all(col in df.columns for col in required_cols):
                print(f"ERROR: CSV {csv_path} is missing one or more required columns: {required_cols}", file=sys.stderr)
                overall_errors = True
                continue
            # Ensure numerical types
            df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors='coerce')
            df[RESID_COL] = pd.to_numeric(df[RESID_COL], errors='coerce')
            df[ACTUAL_COL] = pd.to_numeric(df[ACTUAL_COL], errors='coerce')
            df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors='coerce')
            df[UNCERT_COL] = pd.to_numeric(df[UNCERT_COL], errors='coerce')
            df.dropna(subset=[TEMP_COL, RESID_COL], inplace=True) # Drop rows where essential keys are missing
            df[RESID_COL] = df[RESID_COL].astype(int) # Convert resid to int after dropping NaNs

        except Exception as e:
            print(f"ERROR: Failed to read or process CSV {csv_path}: {e}", file=sys.stderr)
            overall_errors = True
            continue

        # Determine which temperatures to process for this domain
        if domain_id == MULTI_TEMP_DOMAIN:
            temps_to_process = ALL_TEMPS
        elif domain_id in SINGLE_TEMP_DOMAINS:
            temps_to_process = [TARGET_TEMP_SINGLE]
        else:
            print(f"WARN: Domain {domain_id} not specified as single or multi-temp. Skipping.", file=sys.stderr)
            continue

        for temp in temps_to_process:
            print(f"  Processing Temperature: {temp:.1f}K")
            # Use numpy.isclose for float comparison to handle potential precision issues
            temp_df = df[pd.to_numeric(df[TEMP_COL], errors='coerce').apply(lambda x: pd.notna(x) and abs(x - temp) < 1e-6)].copy()
            # temp_df = df[np.isclose(df[TEMP_COL], temp)].copy() # Requires numpy import


            if temp_df.empty:
                print(f"  WARN: No data found for temperature {temp:.1f}K in {csv_path}. Skipping this temperature.", file=sys.stderr)
                continue

            # Check for missing essential data AFTER filtering for temperature
            # Use dropna() before creating maps to avoid passing NaNs
            pred_map = dict(zip(temp_df.dropna(subset=[PRED_COL])[RESID_COL],
                                temp_df.dropna(subset=[PRED_COL])[PRED_COL]))
            actual_map = dict(zip(temp_df.dropna(subset=[ACTUAL_COL])[RESID_COL],
                                  temp_df.dropna(subset=[ACTUAL_COL])[ACTUAL_COL]))
            uncert_map = dict(zip(temp_df.dropna(subset=[UNCERT_COL])[RESID_COL],
                                  temp_df.dropna(subset=[UNCERT_COL])[UNCERT_COL]))

            # Create diff_map only from residues present in both pred and actual maps
            valid_resids_for_diff = set(pred_map.keys()) & set(actual_map.keys())
            diff_map = {resid: pred_map[resid] - actual_map[resid] for resid in valid_resids_for_diff}


            # Report missing values AFTER creating maps (to see how many were excluded)
            missing_pred = temp_df[PRED_COL].isnull().sum()
            missing_actual = temp_df[ACTUAL_COL].isnull().sum()
            missing_uncert = temp_df[UNCERT_COL].isnull().sum()
            if missing_pred > 0:
                 print(f"  INFO: {missing_pred} missing predicted RMSF values for {temp:.1f}K were excluded.", file=sys.stderr)
            if missing_actual > 0:
                 print(f"  INFO: {missing_actual} missing actual RMSF values for {temp:.1f}K were excluded.", file=sys.stderr)
            if missing_uncert > 0:
                 print(f"  INFO: {missing_uncert} missing uncertainty values for {temp:.1f}K were excluded.", file=sys.stderr)

            # --- Define output paths ---
            temp_str = f"{int(temp)}K" # Format temperature string without decimal
            pred_path = os.path.join(CASE_STUDIES_DIR, OUTPUT_DIRS['pred'], f"{domain_id}_pred_{temp_str}.pdb")
            actual_path = os.path.join(CASE_STUDIES_DIR, OUTPUT_DIRS['actual'], f"{domain_id}_actual_{temp_str}.pdb")
            diff_path = os.path.join(CASE_STUDIES_DIR, OUTPUT_DIRS['diff'], f"{domain_id}_diff_{temp_str}.pdb")
            uncert_path = os.path.join(CASE_STUDIES_DIR, OUTPUT_DIRS['uncert'], f"{domain_id}_uncert_{temp_str}.pdb")

            # --- Write PDBs ---
            print(f"    Writing Predicted RMSF PDB: {os.path.basename(pred_path)}")
            err1 = write_pdb_with_bfactor(pdb_path, pred_path, pred_map, domain_id, temp, "Predicted RMSF")
            print(f"    Writing Actual RMSF PDB: {os.path.basename(actual_path)}")
            err2 = write_pdb_with_bfactor(pdb_path, actual_path, actual_map, domain_id, temp, "Actual RMSF")
            print(f"    Writing Difference PDB: {os.path.basename(diff_path)}")
            err3 = write_pdb_with_bfactor(pdb_path, diff_path, diff_map, domain_id, temp, "Difference")
            print(f"    Writing Uncertainty PDB: {os.path.basename(uncert_path)}")
            err4 = write_pdb_with_bfactor(pdb_path, uncert_path, uncert_map, domain_id, temp, "Uncertainty")

            if err1 or err2 or err3 or err4:
                overall_errors = True
                print(f"  -> Errors encountered for {domain_id} at {temp:.1f}K.")
            else:
                 print(f"  -> Successfully processed {temp:.1f}K.")

    print("\n--- Script Finished ---")
    if overall_errors:
        print("WARNING: Errors were encountered during processing. Please check the output above.")
    else:
        print("All specified domains processed successfully.")

# --- Run Script ---
if __name__ == "__main__":
    main()