# XRF Cluster Core Library (v2.0.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Version](https://img.shields.io/badge/Version-2.0.0-blue)
![Python Version](https://img.shields.io/badge/Python-3.8+-important)

**Owner:** AdamNissen ([github.com/AdamNissen](https://github.com/AdamNissen))
**Repository:** [github.com/AdamNissen/xrf_cluster](https://github.com/AdamNissen/xrf_cluster)

This repository contains the core Python library (version 2.0.0) providing backend functionalities for unsupervised clustering (mineral phase mapping) of elemental map data (e.g., from XRF, EDS, EPMA). This library is designed to be reusable and installable via pip, serving as a dependency for front-end applications like the private `xrf_cluster_gui`.

Version 2.0.0 represents a significant refactoring to improve modularity, testability, and adhere to modern Python packaging standards using a `src/` layout and `pyproject.toml`.

## Core Features

* **Data Loading (`core.data_io`):** Loads elemental maps from `.bmp` (grayscale conversion) or `.txt`/`.TXT` files (comma-delimited). Validates dimensions across all maps in a directory.
* **Data Scaling (`core.processing`):** Scales data using `StandardScaler` (zero mean, unit variance) for optimal clustering performance. Handles zero-variance columns.
* **Clustering (`core.processing`):** Performs K-Means clustering using `scikit-learn`.
* **Automatic k-Determination (`core.processing`):** Estimates the optimal number of clusters ('k') using the Silhouette Score method, with subsampling for efficiency on large datasets.
* **Outlier/Null Phase (`core.processing`):** Optionally identifies outlier pixels based on distance from their assigned cluster centroid and assigns them to a separate "null phase" label.
* **Statistics (`core.statistics`):** Calculates phase statistics (pixel count, proportion, mean elemental value, standard deviation) for each identified phase using the original (unscaled) data.
* **Plotting (`core.plotting`):** Generates summary plots (e.g., line or bar charts) comparing element statistics across different phases, saved as image files.
* **Output Saving (`core.data_io`):** Saves the resulting phase map as an image file (TIFF default) using perceptually uniform and colorblind-friendly colormaps (Colorcet Glasbey default). Saves statistics to a CSV file. Includes logic to prevent overwriting existing files ("file bumping").

## Requirements

* Python 3.8 or higher
* Core Dependencies (see `pyproject.toml` or `requirements.txt` for specific versions):
    * Numpy
    * Pandas
    * Scikit-learn
    * Matplotlib
    * Pillow
    * Colorcet

## Installation

### For Users / Dependent Projects (Recommended)

You can install this library directly from GitHub. Using the `dev` branch is recommended for accessing the latest developments, or use a specific tag (like `v2.0.0` once created) for stable releases.

```bash
# Install from the dev branch
pip install "git+[https://github.com/AdamNissen/xrf_cluster.git@dev#egg=xrf-cluster-core](https://github.com/AdamNissen/xrf_cluster.git@dev#egg=xrf-cluster-core)"

# Install a specific tag/release (replace v2.0.0 with the actual tag)
# pip install "git+[https://github.com/AdamNissen/xrf_cluster.git@v2.0.0#egg=xrf-cluster-core](https://github.com/AdamNissen/xrf_cluster.git@v2.0.0#egg=xrf-cluster-core)"
(Note: The installable package name is xrf-cluster-core as specified by #egg=.... Ensure this matches the name in pyproject.toml if changed.)

## For Developers (Contributing to this Library)

1. Clone the repository:
   git clone https://github.com/AdamNissen/xrf_cluster.git
   cd xrf_cluster

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # .\venv\Scripts\activate # Windows

3. Install dependencies (including development tools if specified):
   pip install -r requirements.txt

4. Install the package in editable mode (recommended for development):
   pip install -e .
   (This command uses pyproject.toml to install the package such that changes in the src/ directory are immediately reflected without needing reinstallation.)

## Usage (as a Library)

Here's a basic example of how to use the core functions:

    import numpy as np
    import pandas as pd
    # Assuming package name is 'xrf_cluster_core' after installation
    # Adjust import if package name in pyproject.toml is different
    from xrf_cluster_core.core import data_io, processing, statistics, plotting

    # --- Example Workflow ---
    input_path = "./path/to/your/map/data" # Directory with .bmp/.txt files
    output_path = "./output_results"
    run_auto_k = True
    manual_k = 4 # Ignored if run_auto_k is True
    add_null = True

    try:
        # 1. Load Data
        original_data, map_shape, elem_names = data_io.load_element_maps(input_path)
        print(f"Loaded {len(elem_names)} elements. Map Shape: {map_shape}")

        # 2. Scale Data
        scaled_data = processing.scale_data(original_data)

        # 3. Determine K
        if run_auto_k:
            k = processing.determine_optimal_k(scaled_data) # Uses default range
            print(f"Automatically determined k = {k}")
        else:
            k = manual_k
            print(f"Using manual k = {k}")

        # 4. Perform Clustering
        labels, final_k, model = processing.perform_clustering(scaled_data,
                                                               n_clusters=k,
                                                               add_null_phase=add_null)
        print(f"Clustering complete. Final number of phases: {final_k}")

        # 5. Calculate Statistics (using original data!)
        stats_df = statistics.calculate_phase_stats(original_data,
                                                    labels,
                                                    final_k, # Pass the actual final count
                                                    elem_names,
                                                    map_shape)
        print("Calculated Statistics:\n", stats_df.head())

        # 6. Save Results (incl. summary plots)
        # 6a. Generate summary plots (implement function in plotting.py first)
        try:
            # Assuming plot_statistics_comparison returns list of saved plot paths
            plot_paths = plotting.plot_statistics_comparison(stats_df, elem_names, output_dir=output_path)
            print(f"Saved summary plots: {plot_paths}")
        except AttributeError:
             print("Skipping summary plots (plotting function likely not implemented yet).")
        except Exception as plot_e:
            print(f"Could not generate summary plots: {plot_e}")

        # 6b. Save map and stats CSV (handles file bumping, uses TIFF default)
        # TODO: Implement refinement to pass edited names/colors if needed
        data_io.save_results(output_path, labels, stats_df, map_shape)

        print(f"\nAnalysis complete. Results saved in {output_path}")

    except FileNotFoundError:
        print(f"Error: Input directory not found at {input_path}")
    except ValueError as ve:
        print(f"Error: Data validation failed - {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


## Testing

Automated tests are located in the tests/ directory. You can run them using pytest (after installing it: pip install pytest):

    python -m pytest tests/


## Contributing

Contributions to this core library are welcome. Please adhere to the following process:

1. Check the Issues page (https://github.com/AdamNissen/xrf_cluster/issues) for existing bug reports or feature requests.
2. Open an issue to discuss significant changes or new features.
3. Fork the repository.
4. Create a feature branch (git checkout -b feature/YourFeature).
5. Make your changes. Include tests for new functionality.
6. Ensure all tests pass.
7. Commit your changes (git commit -am 'Add some feature').
8. Push to the branch (git push origin feature/YourFeature).
9. Open a Pull Request against the dev branch of the main repository (https://github.com/AdamNissen/xrf_cluster).


## License

This project is licensed under the [Specify Chosen License, e.g., MIT License]. See the LICENSE file for full details.

## Contact

For issues, questions, or suggestions regarding this core library, please use the GitHub Issues page (https://github.com/AdamNissen/xrf_cluster/issues) for this repository, or email the following:

adamjnissen@gmail.com