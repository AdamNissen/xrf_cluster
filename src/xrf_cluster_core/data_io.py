# src/xrf_cluster_core/data_io.py

import os
import glob
import numpy as np
from PIL import Image # Using Pillow for image loading
import pandas as pd
import matplotlib.pyplot as plt
# Removed colorcet and specific matplotlib.colors imports for simplification

# --- Configuration ---
SUPPORTED_EXTENSIONS_LOWER = ['.bmp', '.txt']

# --- load_element_maps function (Unchanged from last correct version) ---
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


# --- save_results function (Simplified Version) ---
def save_results(output_dir: str, phase_map_labels: np.ndarray, stats_df: pd.DataFrame, map_shape: tuple[int, int]):
    """
    Saves the phase map image (default TIFF) and statistics CSV.
    Uses a standard matplotlib colormap (viridis) for simplicity.

    Args:
        output_dir: The path to the directory where results will be saved.
        phase_map_labels: 1D NumPy array containing the cluster label for each pixel.
        stats_df: Pandas DataFrame containing the calculated phase statistics.
        map_shape: The original map dimensions (height, width).
    """
    print(f"\nSaving results to: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"  Error creating output directory '{output_dir}': {e}")
        raise

    # --- Save Phase Map Image (Simplified) ---
    output_image_format = 'tiff' # Default format
    output_map_filename = f'phase_map.{output_image_format}' # Base filename
    # TODO: Task 1.5 - Implement file bumping logic here
    output_map_path = os.path.join(output_dir, output_map_filename)
    print(f"  Generating phase map image ({output_image_format}) using 'viridis' cmap...")

    try:
        if phase_map_labels.size != map_shape[0] * map_shape[1]:
             raise ValueError(f"Label array size ({phase_map_labels.size}) does not match map dimensions ({map_shape[0]}x{map_shape[1]}={map_shape[0] * map_shape[1]})")

        phase_map_reshaped = phase_map_labels.reshape(map_shape)
        unique_labels = np.unique(phase_map_labels)
        num_unique_phases = len(unique_labels)
        print(f"    Found {num_unique_phases} unique phases for plotting: {unique_labels}")

        if num_unique_phases == 0:
             print("    Warning: No unique phases found in labels. Skipping map generation.")
        else:
            # --- Use standard Matplotlib colormap ---
            cmap = plt.get_cmap('viridis') # Using viridis as requested

            fig, ax = plt.subplots(figsize=(8, 8 * map_shape[0] / map_shape[1]))
            # Let imshow handle normalization of integer labels for the colormap
            im = ax.imshow(phase_map_reshaped, cmap=cmap, interpolation='none')
            ax.set_title('Mineral Phase Map')
            ax.axis('off')

            # --- Configure Colorbar (Simplified) ---
            # Create ticks based on the unique labels found
            sorted_labels = np.sort(unique_labels)
            cbar = fig.colorbar(im, ax=ax, ticks=sorted_labels, spacing='proportional', label='Phase ID')
            # Setting ticks should usually be sufficient for integer data with imshow's default norm
            print(f"    Colorbar created with ticks: {sorted_labels}")

            # Save the figure
            plt.savefig(output_map_path, dpi=300, bbox_inches='tight', format=output_image_format)
            plt.close(fig) # Close the figure to free memory
            print(f"  Phase map saved to: {output_map_path}")

    except Exception as e:
        print(f"  Error generating or saving phase map image: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        raise

    # --- Save Statistics CSV (Unchanged) ---
    output_stats_filename = 'phase_statistics.csv' # Base filename
    # TODO: Task 1.5 - Implement file bumping logic here
    output_stats_path = os.path.join(output_dir, output_stats_filename)
    print(f"  Saving phase statistics...")
    try:
        if not stats_df.empty:
             stats_df.to_csv(output_stats_path, index=False, float_format='%.4f')
             print(f"  Statistics saved to: {output_stats_path}")
        else:
             print("  Skipping saving statistics: DataFrame is empty.")
    except Exception as e:
         print(f"  Error saving statistics CSV: {e}")
         raise


# --- Example Usage (if __name__ == "__main__") - Unchanged ---
# (Keep the testing block from the previous version here)
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
    test_phase_labels = [0, 2, 4] # Non-contiguous labels
    num_phases_test = len(test_phase_labels)

    try:
        # --- Create Dummy Input Files ---
        elem_names_test = ['ElemA', 'ElemC', 'ElemE']
        Image.fromarray(np.random.randint(0, 256, size=(map_h, map_w), dtype=np.uint8), mode='L').save(os.path.join(test_dir, f"{elem_names_test[0]}.bmp"))
        np.savetxt(os.path.join(test_dir, f"{elem_names_test[1]}.txt"), np.random.rand(map_h, map_w) * 100, fmt='%.4f', delimiter=',')
        np.savetxt(os.path.join(test_dir, f"{elem_names_test[2]}.TXT"), np.random.rand(map_h, map_w) * 50, fmt='%.4f', delimiter=',')
        print(f"  Saved dummy input files.")

        # --- Test Loading ---
        print("\n--- Testing Load Function ---")
        pixel_data, map_shape_loaded, element_names_loaded = load_element_maps(test_dir)
        print("  Loading successful.")
        assert map_shape_loaded == (map_h, map_w)
        assert pixel_data.shape == (num_pixels_test, num_elements_test)
        assert sorted(element_names_loaded) == sorted(elem_names_test)

        # --- Test Saving ---
        print("\n--- Testing Save Function (Simplified Plotting) ---")
        dummy_labels = np.random.choice(test_phase_labels, size=num_pixels_test)
        for i, label in enumerate(test_phase_labels):
             if label not in dummy_labels: dummy_labels[i % num_pixels_test] = label
        print(f"  Unique dummy labels generated: {np.unique(dummy_labels)}")

        phase_names_for_stats = [f"Phase {lbl}" for lbl in np.unique(dummy_labels)]
        dummy_stats_data = {'Phase': phase_names_for_stats,
                            'Proportion (%)': [100.0 / num_phases_test] * len(phase_names_for_stats)}
        for name in element_names_loaded:
            dummy_stats_data[f'Mean_{name}'] = np.random.rand(len(phase_names_for_stats)) * 100
            dummy_stats_data[f'StdDev_{name}'] = np.random.rand(len(phase_names_for_stats)) * 10
        dummy_stats_df = pd.DataFrame(dummy_stats_data)

        save_results(test_out_dir, dummy_labels, dummy_stats_df, map_shape_loaded)

        expected_map_file = os.path.join(test_out_dir, 'phase_map.tiff')
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