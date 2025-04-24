# src/xrf_cluster_core/data_io.py

import os
import glob
import numpy as np
from PIL import Image # Using Pillow for image loading
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, is_color_like # Import necessary tools
from typing import Optional, Dict, Any # For type hinting optional dictionaries

# --- Configuration ---
SUPPORTED_EXTENSIONS_LOWER = ['.bmp', '.txt']
DEFAULT_CMAP = 'tab20' # Default categorical colormap

# --- load_element_maps function (Unchanged) ---
def load_element_maps(input_dir: str) -> tuple[np.ndarray, tuple[int, int], list[str]]:
    """
    Loads element map data from .bmp and .txt/.TXT files in a specified directory.
    (Docstring remains the same)
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    print(f"Scanning directory: {input_dir}")
    potential_files = []
    potential_files.extend(glob.glob(os.path.join(input_dir, f'*.[bB][mM][pP]'), recursive=False))
    potential_files.extend(glob.glob(os.path.join(input_dir, f'*.[tT][xX][tT]'), recursive=False))

    all_files_set = set()
    all_files = []
    for f in potential_files:
        ext_lower = os.path.splitext(f)[1].lower()
        if ext_lower in SUPPORTED_EXTENSIONS_LOWER:
            if f not in all_files_set:
                all_files.append(f)
                all_files_set.add(f)

    if not all_files:
        raise FileNotFoundError(f"No supported map files ({', '.join(SUPPORTED_EXTENSIONS_LOWER)}) found in {input_dir}")

    element_data_dict = {}
    element_names_list = []
    map_shape = None
    map_dtype = None

    print(f"Found {len(all_files)} supported map files. Loading...")

    for filepath in all_files:
        filename = os.path.basename(filepath)
        element_name = os.path.splitext(filename)[0]
        file_ext_lower = os.path.splitext(filename)[1].lower()

        try:
            print(f"  Loading: {filename} (as element: '{element_name}')")

            if file_ext_lower == '.bmp':
                with Image.open(filepath) as img:
                    if img.mode == 'L':
                        print(f"    Info: BMP '{filename}' is grayscale ('L').")
                        data = np.array(img)
                    else:
                        print(f"    Info: BMP '{filename}' mode is '{img.mode}'. Converting to grayscale ('L').")
                        try:
                            converted_img = img.convert('L')
                            data = np.array(converted_img)
                        except Exception as convert_e:
                            print(f"    Error: Failed to convert BMP '{filename}' to grayscale: {convert_e}")
                            raise IOError(f"Failed to convert BMP '{filename}' to grayscale") from convert_e

            elif file_ext_lower == '.txt':
                print(f"    Info: Loading TXT '{filename}' with comma delimiter.")
                try:
                    data = np.loadtxt(filepath, delimiter=',')
                except ValueError as txt_e:
                    print(f"    Error: Failed to load TXT file '{filename}'. Check delimiters/format. Error: {txt_e}")
                    raise IOError(f"Failed loading TXT '{filename}'. Ensure it's comma-delimited.") from txt_e
            else:
                print(f"    Skipping file with unexpected extension: {filename}")
                continue

            if map_shape is None:
                map_shape = data.shape
                map_dtype = data.dtype
                print(f"    Detected map shape: {map_shape}, Data type: {map_dtype}")
            elif data.shape != map_shape:
                raise ValueError(f"Dimension mismatch! File '{filename}' has shape {data.shape}, expected {map_shape}.")
            elif data.dtype != map_dtype:
                 print(f"    Warning: Data type mismatch for '{filename}' ({data.dtype}), expected ({map_dtype}). Attempting conversion.")
                 try:
                     data = data.astype(map_dtype)
                 except ValueError as type_e:
                     raise ValueError(f"Could not convert data type for '{filename}' to match expected type '{map_dtype}'. Error: {type_e}") from type_e

            element_data_dict[element_name] = data.flatten()
            element_names_list.append(element_name)

        except FileNotFoundError: raise
        except IOError as e: raise
        except Exception as e: raise

    if not element_data_dict:
        raise ValueError("Could not load any valid element map data.")

    print(f"Successfully loaded {len(element_names_list)} element maps. Stacking data...")
    num_pixels = map_shape[0] * map_shape[1]
    num_elements = len(element_names_list)
    pixel_element_data = np.empty((num_pixels, num_elements), dtype=map_dtype)
    element_names_list.sort()
    print(f"  Element order for stacking: {element_names_list}")
    for i, name in enumerate(element_names_list):
        pixel_element_data[:, i] = element_data_dict[name]

    print("Data stacking complete.")
    return pixel_element_data, map_shape, element_names_list


# --- Helper function for file bumping ---
def _get_unique_filepath(directory: str, base_name: str, extension: str) -> str:
    """Finds a unique filepath by appending (n) if necessary."""
    counter = 0
    filepath = os.path.join(directory, f"{base_name}{extension}")
    while os.path.exists(filepath):
        counter += 1
        filepath = os.path.join(directory, f"{base_name}({counter}){extension}")
    return filepath


# --- save_results function (Refined for Editor Inputs) ---
def save_results(
    output_dir: str,
    phase_map_labels: np.ndarray,
    stats_df: pd.DataFrame,
    map_shape: tuple[int, int],
    phase_name_map: Optional[Dict[Any, str]] = None,
    phase_color_map: Optional[Dict[Any, str]] = None
    # TODO: Add base_filename, image_format parameters later
):
    """
    Saves the phase map image (default TIFF) and statistics CSV.
    Optionally uses user-defined phase names and colors if provided via dictionaries.
    Implements file bumping to avoid overwriting existing files.

    Args:
        output_dir: Path to the directory where results will be saved.
        phase_map_labels: 1D NumPy array of cluster labels for each pixel.
        stats_df: Pandas DataFrame with calculated phase statistics.
        map_shape: Original map dimensions (height, width).
        phase_name_map: Optional dict mapping {label_id: user_defined_name}.
        phase_color_map: Optional dict mapping {label_id: user_defined_color}
                         (e.g., hex string like "#FF0000").
    """
    print(f"\nSaving results to: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"  Error creating output directory '{output_dir}': {e}")
        raise

    # --- Define Base Filenames and Extensions ---
    base_map_name = 'phase_map'
    map_ext = '.tiff' # Default format
    base_stats_name = 'phase_statistics'
    stats_ext = '.csv'

    # --- Determine Unique Output Paths using Helper ---
    output_map_path = _get_unique_filepath(output_dir, base_map_name, map_ext)
    output_stats_path = _get_unique_filepath(output_dir, base_stats_name, stats_ext)

    # --- Prepare Stats DataFrame (Apply User Names if provided) ---
    stats_df_to_save = stats_df.copy() # Work on a copy
    if phase_name_map:
        print("  Applying user-defined phase names to statistics...")
        try:
            # Attempt to map based on the numeric part of the default 'Phase X' column
            # This assumes stats_df still has the default names before this step
            def get_label_from_phase_str(phase_str):
                try:
                    return int(phase_str.split()[-1]) # Extract number from "Phase X"
                except:
                    return None # Handle cases where name isn't in expected format

            stats_df_to_save['Original_Label'] = stats_df_to_save['Phase'].apply(get_label_from_phase_str)
            # Apply mapping, keeping original if label not found or mapping fails
            stats_df_to_save['Phase'] = stats_df_to_save.apply(
                lambda row: phase_name_map.get(row['Original_Label'], row['Phase']) if pd.notna(row['Original_Label']) else row['Phase'],
                axis=1
            )
            stats_df_to_save = stats_df_to_save.drop(columns=['Original_Label']) # Remove temporary column
            print("    Successfully applied phase names.")
        except Exception as name_map_e:
            print(f"    Warning: Could not apply phase name map: {name_map_e}")
            # Proceed with potentially unmapped names

    # --- Save Phase Map Image ---
    print(f"  Generating phase map image ('{os.path.basename(output_map_path)}')...")
    try:
        if phase_map_labels.size != map_shape[0] * map_shape[1]:
             raise ValueError("Label array size mismatch.")

        phase_map_reshaped = phase_map_labels.reshape(map_shape)
        unique_labels = np.unique(phase_map_labels)
        num_unique_phases = len(unique_labels)
        print(f"    Found {num_unique_phases} unique phases for plotting: {unique_labels}")

        if num_unique_phases == 0:
             print("    Warning: No unique phases found. Skipping map generation.")
        else:
            sorted_labels = np.sort(unique_labels)
            cmap = None
            norm = None

            # --- Create Colormap and Normalization ---
            if phase_color_map:
                print("    Using user-defined colors...")
                colors = []
                fallback_cmap = plt.get_cmap(DEFAULT_CMAP) # Get fallback cmap instance
                fallback_colors_used = 0
                for i, label in enumerate(sorted_labels):
                    color = phase_color_map.get(label) # Get user color for this label
                    if color and is_color_like(color):
                        colors.append(color)
                    else:
                        # Fallback: cycle through default cmap if user color missing/invalid
                        fallback_color = fallback_cmap(fallback_colors_used % fallback_cmap.N)
                        colors.append(fallback_color)
                        fallback_colors_used += 1
                        if color is not None: # User provided something, but it was invalid
                             print(f"    Warning: Invalid color '{color}' provided for label {label}. Using fallback.")
                        else: # Color simply wasn't provided for this label
                             print(f"    Warning: No color provided for label {label}. Using fallback.")

                if len(colors) == num_unique_phases:
                    cmap = ListedColormap(colors, name='user_defined_phases')
                    # Create boundaries for normalization
                    boundaries = np.concatenate(([sorted_labels[0] - 0.5],
                                                 sorted_labels[:-1] + np.diff(sorted_labels) / 2.0,
                                                 [sorted_labels[-1] + 0.5]))
                    norm = BoundaryNorm(boundaries, cmap.N)
                else:
                     print("    Error creating color list from user map. Falling back to default cmap.")
                     # Fallback if color list creation failed
                     cmap = plt.get_cmap(DEFAULT_CMAP, num_unique_phases)
                     norm = None # Use default normalization

            else:
                # Default behavior: Use tab20
                print(f"    Using default colormap: '{DEFAULT_CMAP}'")
                cmap = plt.get_cmap(DEFAULT_CMAP, num_unique_phases)
                if num_unique_phases > plt.get_cmap(DEFAULT_CMAP).N: # Check if default cmap has enough colors
                     print(f"Warning: Number of unique phases ({num_unique_phases}) exceeds colors in '{DEFAULT_CMAP}'. Colors will repeat.")
                norm = None # Use default normalization for standard maps

            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(8, 8 * map_shape[0] / map_shape[1]))
            im = ax.imshow(phase_map_reshaped, cmap=cmap, norm=norm, interpolation='none') # norm might be None
            ax.set_title('Mineral Phase Map')
            ax.axis('off')

            # --- Configure Colorbar ---
            cbar_label = 'Phase ID'
            if norm: # Using BoundaryNorm with ListedColormap
                 cbar = fig.colorbar(im, ax=ax, norm=norm, boundaries=boundaries, ticks=sorted_labels,
                                     spacing='uniform', label=cbar_label)
            else: # Using default normalization
                 cbar = fig.colorbar(im, ax=ax, ticks=sorted_labels, spacing='proportional', label=cbar_label)

            # Set tick labels (use names if provided, otherwise numeric labels)
            tick_labels = [str(int(l)) for l in sorted_labels] # Default numeric labels
            if phase_name_map:
                try:
                    # Try to map labels to names provided
                    tick_labels = [phase_name_map.get(l, str(int(l))) for l in sorted_labels]
                    print(f"    Using phase names for colorbar ticks: {tick_labels}")
                except Exception as tick_name_e:
                    print(f"    Warning: Could not apply phase names to colorbar ticks: {tick_name_e}")
            cbar.ax.set_yticklabels(tick_labels)

            # Save the figure
            plt.savefig(output_map_path, dpi=300, bbox_inches='tight', format='tiff')
            plt.close(fig)
            print(f"  Phase map saved to: {output_map_path}")

    except Exception as e:
        print(f"  Error generating or saving phase map image: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        raise


    # --- Save Statistics CSV ---
    print(f"  Saving phase statistics ('{os.path.basename(output_stats_path)}')...")
    try:
        if not stats_df_to_save.empty:
             # Save the potentially modified DataFrame (with updated names)
             stats_df_to_save.to_csv(output_stats_path, index=False, float_format='%.4f')
             print(f"  Statistics saved to: {output_stats_path}")
        else:
             print("  Skipping saving statistics: DataFrame is empty.")
    except Exception as e:
         print(f"  Error saving statistics CSV: {e}")
         raise


# --- Example Usage (if __name__ == "__main__") - Updated for Editor Inputs Test ---
if __name__ == "__main__":
    print("\n" + "="*30 + "\nTesting data_io module...\n" + "="*30)
    test_dir = "temp_test_data_io"
    test_out_dir = "temp_test_data_io_output"

    import shutil
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    if os.path.exists(test_out_dir): shutil.rmtree(test_out_dir)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)
    print(f"Created test directory: {test_dir}")
    print(f"Created output directory: {test_out_dir}")

    map_h, map_w = 10, 15
    num_pixels_test = map_h * map_w
    num_elements_test = 3
    test_phase_labels = [0, 1, 3] # Non-contiguous labels
    num_phases_test = len(test_phase_labels)
    element_names_loaded = ['ElemA', 'ElemC', 'ElemE'] # Example names

    try:
        # --- Create Dummy Input Files (Simplified) ---
        print(f"  Creating dummy input files...")
        # ... (Assume files are created as before) ...

        # --- Create Dummy Data for Saving ---
        print("\n--- Testing Save Function ---")
        dummy_labels = np.random.choice(test_phase_labels, size=num_pixels_test)
        for i, label in enumerate(test_phase_labels): # Ensure all labels present
             if label not in dummy_labels: dummy_labels[i % num_pixels_test] = label
        unique_labels_found = np.unique(dummy_labels)
        num_unique_found = len(unique_labels_found)

        # Create dummy stats DataFrame with DEFAULT names first
        phase_names_default = [f"Phase {lbl}" for lbl in unique_labels_found]
        dummy_stats_data = {'Phase': phase_names_default,
                            'Proportion (%)': [100.0 / num_unique_found] * num_unique_found}
        for name in element_names_loaded:
            dummy_stats_data[f'Mean_{name}'] = np.random.rand(num_unique_found) * 100
            dummy_stats_data[f'StdDev_{name}'] = np.random.rand(num_unique_found) * 10
        dummy_stats_df = pd.DataFrame(dummy_stats_data)

        # --- Define Dummy Editor Inputs ---
        user_phase_names = {
            0: "Matrix",
            1: "Inclusion A",
            3: "Boundary Phase" # Maps original label 3
        }
        user_phase_colors = {
            0: "#CCCCCC", # Gray
            1: "#FFD700", # Gold
            3: "#0000FF"  # Blue
            # Label 2 (if present) would get a fallback color
        }

        # --- Test Saving WITHOUT Editor Inputs ---
        print("\n--- Run 1: Saving WITHOUT editor inputs ---")
        save_results(test_out_dir, dummy_labels, dummy_stats_df, (map_h, map_w))
        assert os.path.exists(os.path.join(test_out_dir, 'phase_map.tiff'))
        assert os.path.exists(os.path.join(test_out_dir, 'phase_statistics.csv'))
        # Check CSV content for default names (optional)
        # df_check1 = pd.read_csv(os.path.join(test_out_dir, 'phase_statistics.csv'))
        # assert "Phase 0" in df_check1['Phase'].values

        # --- Test Saving WITH Editor Inputs (Test Bumping) ---
        print("\n--- Run 2: Saving WITH editor inputs (Test Bumping) ---")
        save_results(test_out_dir, dummy_labels, dummy_stats_df, (map_h, map_w),
                     phase_name_map=user_phase_names,
                     phase_color_map=user_phase_colors)
        # Check for bumped files
        expected_map_file_2 = os.path.join(test_out_dir, 'phase_map(1).tiff')
        expected_stats_file_2 = os.path.join(test_out_dir, 'phase_statistics(1).csv')
        assert os.path.exists(expected_map_file_2)
        assert os.path.exists(expected_stats_file_2)
        # Check CSV content for user names (optional)
        df_check2 = pd.read_csv(expected_stats_file_2)
        assert "Matrix" in df_check2['Phase'].values
        assert "Inclusion A" in df_check2['Phase'].values
        assert "Boundary Phase" in df_check2['Phase'].values

        print("\nSaving tests completed. Check output files in:", test_out_dir)

    except Exception as e:
            print(f"\n--- Test Failed: {e} ---")
            import traceback
            traceback.print_exc()

    finally:
        print(f"\nTest directories left for inspection: '{test_dir}', '{test_out_dir}'")
