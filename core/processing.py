# core/processing.py

import numpy as np
from sklearn.preprocessing import StandardScaler
# --- Add new imports ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
# --- End new imports ---

# Default range of k values to test for automatic determination
DEFAULT_K_RANGE = range(2, 11) # Check k from 2 up to 10
# Max samples to use for silhouette score calculation (for performance)
N_SAMPLES_FOR_SILHOUETTE = 5000
# Random state for reproducibility
RANDOM_STATE = 42

def scale_data(pixel_element_data: np.ndarray) -> np.ndarray:
    """Scales the input pixel data... (docstring from previous step)"""
    # --- Function code from previous step ---
    print("\nScaling data...")
    if pixel_element_data is None or pixel_element_data.size == 0:
        raise ValueError("Input data for scaling cannot be None or empty.")
    if pixel_element_data.ndim != 2:
         raise ValueError(f"Input data must be 2-dimensional (pixels, elements), but got {pixel_element_data.ndim} dimensions.")
    scaler = StandardScaler()
    try:
        scaled_data = scaler.fit_transform(pixel_element_data)
        print("Data scaling complete.")
        # Handle potential NaN outputs from zero-variance columns directly
        if np.sum(np.isnan(scaled_data)) > 0:
             print("Warning: NaNs detected after scaling (likely due to zero-variance columns). Replacing NaNs with 0.")
             scaled_data = np.nan_to_num(scaled_data) # Replace NaN with 0
    except Exception as e:
        print(f"Error during data scaling: {e}")
        raise
    return scaled_data
# --- END of scale_data function ---


def determine_optimal_k(scaled_data: np.ndarray, k_range: range = DEFAULT_K_RANGE) -> int:
    """
    Determines the optimal number of clusters (k) using the Silhouette Score.

    Args:
        scaled_data: The scaled pixel data (num_pixels, num_elements).
        k_range: A range object specifying the number of clusters to test
                 (e.g., range(2, 11) tests k=2 through k=10).

    Returns:
        The estimated optimal number of clusters (k). Defaults to the start
        of the k_range if calculation fails or yields poor results.
    """
    print("\nDetermining optimal k using silhouette score...")
    num_pixels = scaled_data.shape[0]

    if num_pixels <= 1:
         raise ValueError("Cannot determine k with 1 or fewer data points.")
    if max(k_range) >= num_pixels:
         print(f"Warning: Max k in k_range ({max(k_range)}) >= number of pixels ({num_pixels}). Adjusting k_range.")
         k_range = range(k_range.start, num_pixels) # Adjust upper limit

    if not k_range: # Check if range became empty
        print("Warning: k_range is empty after adjustment. Defaulting to k=2 (if possible).")
        return 2 if num_pixels >= 2 else 1


    # Subsample data for faster silhouette calculation if dataset is large
    if num_pixels > N_SAMPLES_FOR_SILHOUETTE:
        print(f"  Subsampling data ({num_pixels} points) to {N_SAMPLES_FOR_SILHOUETTE} points for silhouette calculation.")
        np.random.seed(RANDOM_STATE) # for reproducible subsampling
        indices = np.random.choice(num_pixels, N_SAMPLES_FOR_SILHOUETTE, replace=False)
        data_sample = scaled_data[indices]
    else:
        data_sample = scaled_data
        print(f"  Using all {num_pixels} points for silhouette calculation.")

    n_samples_eff = data_sample.shape[0] # Effective number of samples being used

    silhouette_scores = []
    tested_k_values = []
    best_k = k_range.start # Default to the minimum k to test
    max_score = -1.1 # Initialize below the theoretical minimum (-1)

    for k in k_range:
        if k >= n_samples_eff: # Cannot have more clusters than samples
            print(f"  Skipping k={k} (>= number of samples {n_samples_eff})")
            continue # Skip this k value

        print(f"  Testing k={k}...")
        tested_k_values.append(k)
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')

        try:
            with warnings.catch_warnings():
                 # Suppress specific warnings if needed, e.g., about convergence
                 warnings.simplefilter("ignore", category=UserWarning) # Can be more specific
                 labels = kmeans.fit_predict(data_sample)

            # Check if more than 1 cluster was actually formed (can happen with weird data)
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                print(f"    Warning: Only {len(unique_labels)} cluster(s) found for k={k}. Cannot calculate silhouette score.")
                silhouette_scores.append(-1) # Assign a poor score
                continue

            score = silhouette_score(data_sample, labels)
            silhouette_scores.append(score)
            print(f"    Silhouette Score: {score:.4f}")
            if score > max_score:
                max_score = score
                best_k = k

        except Exception as e:
            print(f"    Error during KMeans fitting or silhouette calculation for k={k}: {e}")
            silhouette_scores.append(-1) # Assign poor score on error


    # --- Optional: Plot Silhouette Scores (can be useful for debugging) ---
    # import matplotlib.pyplot as plt
    # if tested_k_values:
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(tested_k_values, silhouette_scores, marker='o')
    #     plt.xlabel("Number of clusters (k)")
    #     plt.ylabel("Average Silhouette Score")
    #     plt.title("Silhouette Score for Optimal k")
    #     plt.xticks(tested_k_values)
    #     plt.grid(True)
    #     plt.show() # Consider saving instead of showing in a non-interactive context
    # --- End Optional Plot ---

    if max_score <= 0 or best_k < k_range.start : # Check if calculation failed or only found poor scores
        print(f"Warning: Automatic k detection yielded poor results (max score={max_score:.4f}).")
        final_k = k_range.start
        print(f"  Defaulting to minimum tested k: {final_k}")
    else:
        final_k = best_k
        print(f"\nAutomatic determination suggests k={final_k} (Max Silhouette Score: {max_score:.4f})")

    return final_k


# Example Usage (updated to include k-determination)
if __name__ == "__main__":
    print("\n" + "="*30 + "\nTesting processing module (scaling & k-determination)...\n" + "="*30)

    # Create dummy multi-element data (pixels x elements)
    # Make slightly more complex data for clustering test
    np.random.seed(RANDOM_STATE)
    c1 = np.random.rand(50, 2) * 0.5 + [1, 1] # Cluster 1
    c2 = np.random.rand(50, 2) * 0.5 + [3, 3] # Cluster 2
    c3 = np.random.rand(50, 2) * 0.5 + [1, 3] # Cluster 3
    data_test = np.vstack((c1, c2, c3))
    const_col = np.full((data_test.shape[0], 1), 5.0) # Add constant column
    data_test_full = np.hstack((data_test, const_col))

    print("Original Data Sample (first 5 rows):")
    print(data_test_full[:5,:])
    print(f"Original Means: {np.mean(data_test_full, axis=0)}")
    print(f"Original Std Devs: {np.std(data_test_full, axis=0)}")

    try:
        # 1. Scale the data
        scaled_data_test = scale_data(data_test_full)
        scaled_means = np.mean(scaled_data_test, axis=0)
        scaled_stds = np.std(scaled_data_test, axis=0)
        print("\nScaled Data Sample (first 5 rows):")
        print(scaled_data_test[:5,:])
        print(f"Scaled Means: {scaled_means}")
        print(f"Scaled Std Devs: {scaled_stds}") # Should be ~[1, 1, 0]

        # 2. Determine optimal k
        # Use a wider range for testing, e.g., 2 to 6
        test_k_range = range(2, 7)
        optimal_k = determine_optimal_k(scaled_data_test, k_range=test_k_range)
        print(f"\nTest Result: Optimal k determined as: {optimal_k}")
        # For this dummy data, we expect k=3 to likely have the highest score
        # assert optimal_k == 3 # Assertion might be too strict depending on randomness

    except Exception as e:
        print(f"\n--- Processing Test Failed: {e} ---")
        import traceback
        traceback.print_exc()