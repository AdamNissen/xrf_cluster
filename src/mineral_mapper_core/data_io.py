# core/data_io.py

import os
import glob
import numpy as np
from PIL import Image # Using Pillow for image loading
import pandas as pd
import matplotlib.pyplot as plt
import colorcet as cc # <-- Import colorcet

# Set a non-interactive backend for Matplotlib if running without a display (optional)
# import matplotlib
# matplotlib.use('Agg')

# --- Configuration ---
SUPPORTED_EXTENSIONS_LOWER = ['.bmp', '.txt']

# ... load_element_maps function remains the same as the last version ...
# (Make sure the previous version of load_element_maps is here)
def load_element_maps(input_dir: str) -> tuple[np.ndarray, tuple[int, int], list[str]]:
    """Loads element map data... (docstring from previous version)"""
    # --- FUNCTION CODE FROM PREVIOUS STEP ---
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

        except FileNotFoundError:
             print(f"  Error: File disappeared or invalid path: {filename}")
             raise
        except IOError as e:
            print(f"  Error reading or processing file {filename}: {e}")
            raise
        except Exception as e:
            print(f"  An unexpected error occurred processing file {filename}: {e}")
            raise

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
# --- END OF load_element_maps FUNCTION ---


def save_results(output_dir: str, phase_map_labels: np.ndarray, stats_df: pd.DataFrame, map_shape: tuple[int, int]):
                                # TODO: Add image_format parameter later for user selection
    """
    Saves the phase map image (default TIFF) and statistics CSV to the output directory.

    Args:
        output_dir: The path to the directory where results will be saved.
        phase_map_labels: 1D NumPy array containing the cluster label for each pixel.
        stats_df: Pandas DataFrame containing the calculated phase statistics.
                          Expected to have a 'Phase' column for potential labeling.
        map_shape: The original map dimensions (height, width).
    """
    print(f"\nSaving results to: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"  Error creating output directory '{output_dir}': {e}")
        raise

    # --- Save Phase Map Image ---
    # Default to TIFF format, filename reflects this
    output_image_format = 'tiff' # TODO: Make this an argument/option
    output_map_filename = f'phase_map.{output_image_format}'
    output_map_path = os.path.join(output_dir, output_map_filename)
    print(f"  Generating phase map image ({output_image_format})...")

    try:
        if phase_map_labels.size != map_shape[0] * map_shape[1]:
             raise ValueError(f"Label array size ({phase_map_labels.size}) does not match map dimensions ({map_shape[0]}x{map_shape[1]}={map_shape[0] * map_shape[1]})")

        phase_map_reshaped = phase_map_labels.reshape(map_shape)
        unique_labels = np.unique(phase_map_labels)
        num_unique_phases = len(unique_labels)

        # Use colorcet glasbey colormap (handles cycling for many colors)
        # Using glasbey_dark as an example, others exist (glasbey_light, glasbey_bw, etc.)
        cmap = cc.cm.glasbey # Consider cc.cm.glasbey_dark or other variants if needed

        fig, ax = plt.subplots(figsize=(8, 8 * map_shape[0] / map_shape[1])) # Adjust aspect ratio
        # Need to ensure labels map correctly to the colormap indices
        # Map labels (which might be 0, 1, 3...) to range 0..N-1 for cmap indexing
        label_to_cmap_idx = {label: i for i, label in enumerate(unique_labels)}
        cmap_indices = np.vectorize(label_to_cmap_idx.get)(phase_map_reshaped)

        im = ax.imshow(cmap_indices, cmap=cmap, interpolation='none', vmin=0, vmax=num_unique_phases-1)
        ax.set_title('Mineral Phase Map')
        ax.axis('off')

        # Create a colorbar with discrete ticks matching the *original* labels
        # Map cmap indices back to original labels for ticks
        ticks_norm = [label_to_cmap_idx[label] for label in unique_labels] # Normalized tick positions (0..N-1)
        # Center ticks within color blocks
        tick_locs = np.array(ticks_norm) + 0.5 * (ticks_norm[1] - ticks_norm[0]) if len(ticks_norm) > 1 else [0.5] # Handle single phase case

        cbar = fig.colorbar(im, ax=ax, label='Phase ID')
        cbar.set_ticks(ticks_norm) # Set ticks based on the 0..N-1 range used in imshow
        # Set tick labels to the original phase labels (0, 1, 2, -1 etc.)
        cbar.set_ticklabels(unique_labels)


        # --- Optional: Code block for potentially mapping labels to names (remains commented) ---
        # try:
        #     if 'Phase' in stats_df.columns:
        #         phase_names = {int(label): name for label, name in zip(unique_labels, stats_df['Phase'])}
        #         tick_labels = [phase_names.get(int(t), str(int(t))) for t in unique_labels]
        #         cbar.set_ticklabels(tick_labels) # Set labels corresponding to original values
        #         print(f"    Labeled colorbar ticks with names: {tick_labels}")
        #     else:
        #          print("    'Phase' column not found in stats_df, using numeric tick labels.")
        # except Exception as tick_label_e:
        #      print(f"    Warning: Could not map tick labels to phase names: {tick_label_e}")
        # --- End Optional Tick Labeling ---


        # Save as TIFF (or format specified by output_image_format)
        plt.savefig(output_map_path, dpi=300, bbox_inches='tight', format=output_image_format)
        plt.close(fig) # Close the figure to free memory
        print(f"  Phase map saved to: {output_map_path}")

    except Exception as e:
        print(f"  Error generating or saving phase map image: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        raise


    # --- Save Statistics CSV ---
    # Filename default approved
    output_stats_filename = 'phase_statistics.csv'
    output_stats_path = os.path.join(output_dir, output_stats_filename)
    print(f"  Saving phase statistics...")
    try:
        stats_df.to_csv(output_stats_path, index=False, float_format='%.4f')
        print(f"  Statistics saved to: {output_stats_path}")
    except Exception as e:
         print(f"  Error saving statistics CSV: {e}")
         raise


# Example Usage (updated for TIFF output)
if __name__ == "__main__":
    print("\n" + "="*30 + "\nTesting data_io module...\n" + "="*30)
    test_dir = "temp_test_data_io"
    test_out_dir = "temp_test_data_io_output"

    # --- Clean and Create Test Dirs ---
    import shutil
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    if os.path.exists(test_out_dir): shutil.rmtree(test_out_dir)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Created test directory: {test_dir}")
    print(f"Output directory set to: {test_out_dir}")

    map_h, map_w = 10, 15
    num_pixels_test = map_h * map_w
    num_elements_test = 3
    num_phases_test = 4 # Use a few phases for colormap test

    try:
        # --- Create Dummy Input Files ---
        dummy_gray = np.random.randint(0, 256, size=(map_h, map_w), dtype=np.uint8)
        Image.fromarray(dummy_gray, mode='L').save(os.path.join(test_dir, "ElementA.bmp"))
        dummy_txt = np.random.rand(map_h, map_w) * 100
        np.savetxt(os.path.join(test_dir, "ElementC.txt"), dummy_txt, fmt='%.4f', delimiter=',')
        dummy_txt_upper = np.random.rand(map_h, map_w) * 50
        np.savetxt(os.path.join(test_dir, "ElementE_caps.TXT"), dummy_txt_upper, fmt='%.4f', delimiter=',')
        print(f"  Saved dummy input files.")

        # --- Test Loading ---
        print("\n--- Testing Load Function ---")
        pixel_data, map_shape_loaded, element_names_loaded = load_element_maps(test_dir)
        print("  Loading successful.")
        assert map_shape_loaded == (map_h, map_w)
        assert pixel_data.shape == (num_pixels_test, num_elements_test)
        assert len(element_names_loaded) == num_elements_test

        # --- Test Saving ---
        print("\n--- Testing Save Function ---")
        # Create dummy label data (e.g., random labels 0, 1, 2, 3)
        dummy_labels = np.random.randint(0, num_phases_test, size=num_pixels_test)
        # Create dummy stats DataFrame
        phase_names_for_stats = [f"Phase {i}" for i in range(num_phases_test)] # Ensure Phase names match number of labels
        dummy_stats_data = {'Phase': phase_names_for_stats,
                            'Proportion (%)': [100.0 / num_phases_test] * num_phases_test}
        # Add element stats based on loaded names
        for name in element_names_loaded:
            dummy_stats_data[f'Mean_{name}'] = np.random.rand(num_phases_test) * 100
            dummy_stats_data[f'StdDev_{name}'] = np.random.rand(num_phases_test) * 10
        dummy_stats_df = pd.DataFrame(dummy_stats_data)

        save_results(test_out_dir, dummy_labels, dummy_stats_df, map_shape_loaded)

        # Verify output files exist (check for TIFF)
        expected_map_file = os.path.join(test_out_dir, 'phase_map.tiff') # Check for .tiff
        expected_stats_file = os.path.join(test_out_dir, 'phase_statistics.csv')
        assert os.path.exists(expected_map_file)
        assert os.path.exists(expected_stats_file)
        print("  Saving successful. Output files created.")
        print(f"  Check contents of '{test_out_dir}'")

    except Exception as e:
            print(f"\n--- Test Failed: {e} ---")
            import traceback
            traceback.print_exc()

    finally:
        print(f"\nTest directories left for inspection: '{test_dir}', '{test_out_dir}'")