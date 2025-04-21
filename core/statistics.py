# core/statistics.py

import numpy as np
import pandas as pd

def calculate_phase_stats(original_pixel_data: np.ndarray,
                          labels: np.ndarray,
                          num_clusters: int, # Total number of unique labels expected/found
                          element_names: list[str],
                          map_shape: tuple[int, int]) -> pd.DataFrame:
    """
    Calculates statistics for each phase identified by clustering.

    Uses the *original (unscaled)* data for calculating means and std devs.

    Args:
        original_pixel_data: The UNscaled pixel data (num_pixels, num_elements).
        labels: The final cluster labels (num_pixels) from perform_clustering.
                May include a label for a null phase.
        num_clusters: The total number of unique cluster labels present in `labels`.
                      (Note: This might differ slightly from the target k if some
                       clusters ended up empty or null phase wasn't created).
        element_names: A list of names corresponding to the columns (elements)
                       in original_pixel_data.
        map_shape: The original map dimensions (height, width).

    Returns:
        A Pandas DataFrame where each row represents a phase and columns contain
        statistics (Phase ID, Proportion, Mean_ElementX, StdDev_ElementX, ...).
        Returns an empty DataFrame if no valid clusters are found.
    """
    print("\nCalculating phase statistics...")
    if original_pixel_data is None or original_pixel_data.size == 0:
        raise ValueError("Original pixel data cannot be None or empty.")
    if labels is None or labels.size == 0:
         raise ValueError("Cluster labels cannot be None or empty.")
    if labels.shape[0] != original_pixel_data.shape[0]:
        raise ValueError(f"Labels array length ({labels.shape[0]}) does not match data length ({original_pixel_data.shape[0]}).")
    if len(element_names) != original_pixel_data.shape[1]:
         raise ValueError(f"Number of element names ({len(element_names)}) does not match number of data columns ({original_pixel_data.shape[1]}).")


    total_pixels = map_shape[0] * map_shape[1]
    if total_pixels <= 0:
         raise ValueError("Total pixels must be positive.")

    stats_list = []
    unique_labels = np.unique(labels)
    print(f"  Found {len(unique_labels)} unique labels in the results: {unique_labels}")

    # Define column names for the output DataFrame
    stat_columns = ['Phase', 'Num_Pixels', 'Proportion (%)']
    for name in element_names:
        stat_columns.extend([f'Mean_{name}', f'StdDev_{name}'])

    # Iterate through each unique cluster label found in the data
    for cluster_label in unique_labels:
        # Decision Point 1: Phase Naming (using generic name for now)
        phase_name = f"Phase {cluster_label}"
        # Note: We could later add logic here or in the calling function
        # to specifically name the 'null phase' if its label index is known.

        cluster_mask = (labels == cluster_label)
        num_pixels_in_cluster = np.sum(cluster_mask)

        # Decision Point 2: Handling Empty Clusters (skip them)
        if num_pixels_in_cluster == 0:
            print(f"  Skipping {phase_name}: No pixels found.")
            continue

        print(f"  Calculating stats for {phase_name} ({num_pixels_in_cluster} pixels)...")

        # Select the original data for this phase
        phase_data = original_pixel_data[cluster_mask]

        # Calculate statistics
        proportion = (num_pixels_in_cluster / total_pixels) * 100
        means = np.mean(phase_data, axis=0)
        stds = np.std(phase_data, axis=0)

        # Store results in a dictionary
        phase_stats = {}
        phase_stats['Phase'] = phase_name
        phase_stats['Num_Pixels'] = num_pixels_in_cluster
        phase_stats['Proportion (%)'] = proportion

        if len(element_names) != len(means) or len(element_names) != len(stds):
             print(f"  Warning: Mismatch between number of element names ({len(element_names)}) and calculated means/stds ({len(means)}/{len(stds)}) for {phase_name}. Skipping stats details.")
             # Handle this case - perhaps fill with NaN or skip adding means/stds
             for name in element_names:
                 phase_stats[f'Mean_{name}'] = np.nan
                 phase_stats[f'StdDev_{name}'] = np.nan
        else:
             for i, name in enumerate(element_names):
                 phase_stats[f'Mean_{name}'] = means[i]
                 phase_stats[f'StdDev_{name}'] = stds[i]

        stats_list.append(phase_stats)

    if not stats_list:
         print("Warning: No statistics were generated (perhaps all clusters were empty?). Returning empty DataFrame.")
         return pd.DataFrame(columns=stat_columns)

    # Convert list of dictionaries to DataFrame
    stats_df = pd.DataFrame(stats_list, columns=stat_columns)
    # Sort by phase label numerically for consistent output
    try:
         # Extract numeric part of phase name for sorting
         stats_df['_sort_key'] = stats_df['Phase'].str.extract(r'(\d+)$').astype(int)
         stats_df = stats_df.sort_values(by='_sort_key').drop(columns=['_sort_key'])
    except Exception as sort_e:
         print(f"Warning: Could not sort statistics DataFrame by phase number: {sort_e}")


    print("Statistics calculation complete.")
    return stats_df


# Example Usage (for testing purposes)
if __name__ == "__main__":
    print("\n" + "="*30 + "\nTesting statistics module...\n" + "="*30)

    # Create dummy data consistent with other modules' tests
    map_h, map_w = 5, 6
    num_pixels = map_h * map_w # 30
    num_elements = 3
    num_phases = 3 # Let's say clustering found 3 phases (0, 1, 2)
    element_names = ['Fe', 'Si', 'Ca']

    # Dummy ORIGINAL (unscaled) pixel data
    np.random.seed(42)
    original_data = np.random.rand(num_pixels, num_elements) * np.array([[100, 50, 10]]) # Different scales

    # Dummy cluster labels
    dummy_labels = np.random.randint(0, num_phases, size=num_pixels)
    # Ensure at least one pixel per label for basic test
    for i in range(num_phases):
        if i not in dummy_labels:
             dummy_labels[i] = i # Force at least one pixel for each label 0, 1, 2

    # Add an empty cluster label (e.g., label 3 has no pixels) - test skipping
    # dummy_labels[5] = 3 # No, num_clusters should match unique labels found
    unique_labels_test = np.unique(dummy_labels)
    num_clusters_test = len(unique_labels_test) # Should be 3

    print(f"Test Setup:")
    print(f"  Map Shape: {(map_h, map_w)}")
    print(f"  Num Pixels: {num_pixels}")
    print(f"  Num Elements: {num_elements} ({element_names})")
    print(f"  Num Clusters: {num_clusters_test}")
    print(f"  Labels (sample): {dummy_labels[:10]}...")
    print(f"  Original Data (sample):\n{original_data[:5,:]}")


    try:
        stats_dataframe = calculate_phase_stats(
            original_pixel_data=original_data,
            labels=dummy_labels,
            num_clusters=num_clusters_test, # Pass the actual number of unique labels found
            element_names=element_names,
            map_shape=(map_h, map_w)
        )

        print("\n--- Statistics Calculation Successful ---")
        print("Resulting DataFrame:")
        # Use pandas options for better console display
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
             print(stats_dataframe)

        # Basic assertions
        assert isinstance(stats_dataframe, pd.DataFrame)
        assert stats_dataframe.shape[0] == num_clusters_test # Should have rows for each non-empty cluster
        assert 'Phase' in stats_dataframe.columns
        assert 'Proportion (%)' in stats_dataframe.columns
        assert 'Mean_Fe' in stats_dataframe.columns
        assert 'StdDev_Si' in stats_dataframe.columns
        # Check if proportions sum roughly to 100
        assert np.isclose(stats_dataframe['Proportion (%)'].sum(), 100.0)

        print("\nBasic assertions passed.")

    except Exception as e:
        print(f"\n--- Statistics Test Failed: {e} ---")
        import traceback
        traceback.print_exc()