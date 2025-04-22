# core/processing.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# --- Constants ---
DEFAULT_K_RANGE = range(2, 11) # Check k from 2 up to 10
N_SAMPLES_FOR_SILHOUETTE = 5000 # Max samples for silhouette calculation
RANDOM_STATE = 42 # For reproducibility
OUTLIER_PERCENTILE = 95 # Threshold for null phase identification
# --- End Constants ---


def scale_data(pixel_element_data: np.ndarray) -> np.ndarray:
    """
    Scales the input pixel data using StandardScaler (zero mean, unit variance).

    Args:
        pixel_element_data: A 2D NumPy array (num_pixels, num_elements)
                              containing the raw element data for each pixel.

    Returns:
        A 2D NumPy array (num_pixels, num_elements) containing the scaled data.

    Raises:
        ValueError: If input data is invalid.
    """
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

    Raises:
        ValueError: If input data has 1 or fewer points.
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
        # Ensure we don't return 1 if only 1 pixel exists after adjustment
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
        # Use n_init='auto' for newer scikit-learn versions
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')

        try:
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=UserWarning) # Suppress potential convergence warnings
                 labels = kmeans.fit_predict(data_sample)

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

    # --- Optional Plot (remains commented out) ---
    # import matplotlib.pyplot as plt ...

    if max_score <= 0 or best_k < k_range.start : # Check if calculation failed or only found poor scores
        print(f"Warning: Automatic k detection yielded poor results (max score={max_score:.4f}).")
        final_k = k_range.start
        print(f"  Defaulting to minimum tested k: {final_k}")
    else:
        final_k = best_k
        print(f"\nAutomatic determination suggests k={final_k} (Max Silhouette Score: {max_score:.4f})")

    # Ensure final_k is valid before returning
    if final_k >= num_pixels:
         print(f"Warning: Determined k ({final_k}) >= number of pixels ({num_pixels}). Clamping k to {num_pixels - 1}.")
         final_k = num_pixels - 1
    if final_k < 1: # Should be caught earlier, but double check
        print("Warning: Determined k < 1. Setting k to 1.")
        final_k = 1


    return final_k


def perform_clustering(scaled_data: np.ndarray, n_clusters: int, add_null_phase: bool = False) -> tuple[np.ndarray, int, KMeans]:
    """
    Performs K-Means clustering and optionally adds a null phase for outliers.

    Args:
        scaled_data: The scaled pixel data (num_pixels, num_elements).
        n_clusters: The target number of clusters (k).
        add_null_phase: If True, identify outliers based on distance from
                        centroid and assign them to a new 'null phase' cluster.

    Returns:
        A tuple containing:
        - final_labels: 1D NumPy array (num_pixels) with the final cluster label
                        for each pixel (includes null phase label if created).
        - num_final_clusters: The total number of clusters including the null phase.
        - kmeans_model: The fitted sklearn KMeans object.

    Raises:
        ValueError: If n_clusters is less than 1 or >= number of pixels.
    """
    print(f"\nPerforming K-Means clustering with k={n_clusters}...")
    num_pixels = scaled_data.shape[0]

    if n_clusters < 1:
        raise ValueError("Number of clusters (n_clusters) must be at least 1.")
    # Allow n_clusters == num_pixels (each point is a cluster), but warn? Let's restrict to < num_pixels.
    if n_clusters >= num_pixels:
         raise ValueError(f"Number of clusters ({n_clusters}) must be less than the number of pixels ({num_pixels}).")

    # Perform K-Means clustering on the full scaled dataset
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning) # Suppress potential convergence warnings
            initial_labels = kmeans.fit_predict(scaled_data)
        centroids = kmeans.cluster_centers_
        print(f"Initial K-Means clustering complete. Found {len(np.unique(initial_labels))} initial clusters.")
    except Exception as e:
        print(f"Error during K-Means fitting: {e}")
        raise

    final_labels = initial_labels.copy()
    num_final_clusters = n_clusters # Start with the target number of clusters

    # Optional: Null Phase / Outlier Detection
    if add_null_phase:
        print(f"Attempting to identify outliers for null phase (using {OUTLIER_PERCENTILE}th percentile distance)...")
        if num_pixels == 0:
             print("  Skipping null phase detection: No data points.")
             # Return early? Or let it proceed (should be caught earlier)
             return final_labels, num_final_clusters, kmeans

        # Calculate distance of each point to its assigned centroid
        distances = np.zeros(num_pixels)
        try:
            for i in range(n_clusters):
                cluster_points_mask = (initial_labels == i)
                n_points_in_cluster = np.sum(cluster_points_mask)

                if n_points_in_cluster > 0:
                    point_vectors = scaled_data[cluster_points_mask]
                    centroid_vector = centroids[i]
                    distances[cluster_points_mask] = np.linalg.norm(point_vectors - centroid_vector, axis=1)

            if len(distances) > 0: # Ensure distances were calculated
                distance_threshold = np.percentile(distances, OUTLIER_PERCENTILE)
                outlier_mask = distances > distance_threshold
                num_outliers = np.sum(outlier_mask)
                print(f"  Distance threshold ({OUTLIER_PERCENTILE}th percentile): {distance_threshold:.4f}")
                print(f"  Identified {num_outliers} potential outliers.")

                if num_outliers > 0:
                    null_phase_label = n_clusters # Use k as the label index
                    final_labels[outlier_mask] = null_phase_label
                    num_final_clusters += 1 # Increment the total cluster count
                    print(f"  Assigned label {null_phase_label} to outliers (null phase).")
            else:
                 print("  Warning: Could not calculate distances for outlier detection.")

        except Exception as e:
            print(f"  Error during outlier detection: {e}")
            # Fallback to original labels if outlier detection fails
            final_labels = initial_labels
            num_final_clusters = n_clusters


    print(f"Clustering finished. Final number of clusters: {num_final_clusters}")
    return final_labels, num_final_clusters, kmeans


# Example Usage (updated to include all functions)
if __name__ == "__main__":
    print("\n" + "="*50 + "\nTesting full processing module...\n" + "="*50)

    # Create dummy multi-element data
    np.random.seed(RANDOM_STATE)
    c1 = np.random.rand(50, 2) * 0.5 + [1, 1] # Cluster 1
    c2 = np.random.rand(50, 2) * 0.5 + [3, 3] # Cluster 2
    c3 = np.random.rand(50, 2) * 0.5 + [1, 3] # Cluster 3
    noise = np.random.rand(10, 2) * 4.0 # Outliers
    data_test = np.vstack((c1, c2, c3, noise))
    const_col = np.full((data_test.shape[0], 1), 5.0) # Constant column
    data_test_full = np.hstack((data_test, const_col))
    n_test_pixels = data_test_full.shape[0]

    print(f"Test Data Shape: {data_test_full.shape}")

    try:
        # 1. Scale the data
        scaled_data_test = scale_data(data_test_full)
        assert scaled_data_test.shape == data_test_full.shape

        # 2. Determine optimal k
        optimal_k = determine_optimal_k(scaled_data_test, k_range=range(2, 7))
        print(f"\nOptimal k determined as: {optimal_k}")

        # 3. Perform clustering WITHOUT null phase using optimal k
        print("\n--- Clustering without null phase (using determined k) ---")
        labels_no_null, k_final_no_null, _ = perform_clustering(
            scaled_data_test, n_clusters=optimal_k, add_null_phase=False
        )
        print(f"  Labels shape: {labels_no_null.shape}, Unique labels: {np.unique(labels_no_null)}, Final k: {k_final_no_null}")
        assert len(labels_no_null) == n_test_pixels
        assert k_final_no_null == optimal_k

        # 4. Perform clustering WITH null phase using optimal k
        print("\n--- Clustering WITH null phase (using determined k) ---")
        labels_with_null, k_final_with_null, _ = perform_clustering(
            scaled_data_test, n_clusters=optimal_k, add_null_phase=True
        )
        print(f"  Labels shape: {labels_with_null.shape}, Unique labels: {np.unique(labels_with_null)}, Final k: {k_final_with_null}")
        assert len(labels_with_null) == n_test_pixels
        # Null phase might or might not be created depending on data/threshold
        assert k_final_with_null >= optimal_k

        print("\nFull processing module tests completed successfully.")

    except Exception as e:
        print(f"\n--- Processing Test Failed: {e} ---")
        import traceback
        traceback.print_exc()