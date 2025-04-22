# Design Plan: Mineral Mapper Core Library

**Author** Adam Nissen
**Version:** 1.0
**Date:** 2025-04-21
**Context:** This plan outlines the development for the public core library component of the Mineral Mapper project, intended for open-source use and as a dependency for other applications.
**LLM Acknowledgement** This program was developed using the Gemini 2.5 large language model by Google.  


## 1. Project Goal

To develop and maintain a reusable, installable Python library that provides the core backend functionalities for unsupervised clustering (mineral phase mapping) of elemental map data.

**Core Functionalities:**
* Loading elemental maps (`.bmp`, `.txt/.TXT` with specified delimiters) and validating dimensions (`core.data_io`).
* Scaling elemental data for clustering (`core.processing`).
* Performing K-Means clustering (`core.processing`).
* Optionally determining optimal 'k' via Silhouette Score (`core.processing`).
* Optionally identifying outliers for a "null phase" (`core.processing`).
* Calculating phase statistics (proportion, mean, std dev) using original data (`core.statistics`).
* Generating statistical summary plots (`core.plotting`).
* Saving phase map images (TIFF default) and statistics (CSV), including robust color mapping and optional file bumping (`core.data_io`).

## 2. Repository Structure (`xrf_cluster`)rs.
xrf_cluster/
├── src/
│   └── mineral_mapper_core/  # Installable package source
│       ├── init.py       # Defines package, exports, holds version
│       ├── core/             # Core logic sub-package
│       │   ├── init.py
│       │   ├── data_io.py      # Loading, Saving (incl. file bumping)
│       │   ├── processing.py   # Scaling, K-Determination, Clustering
│       │   ├── statistics.py   # Stats calculation
│       │   └── plotting.py     # Summary plot generation
│       └── utils/            # Optional helpers
│           └── init.py
├── tests/                    # Unit/integration tests for core functions
│   ├── init.py
│   └── test_*.py
├── data/                     # Example/test data for library testing
│   └── ...
├── pyproject.toml            # Build system definition, metadata, dependencies (PEP 517/518)
├── README.md                 # Documentation for using this LIBRARY (API, setup)
├── LICENSE                   # Open-source license (e.g., MIT)
├── requirements.txt          # Dependencies for DEVELOPMENT of this library
└── .gitignore                # Standard Python ignores

* **Key Files:**
    * `pyproject.toml`: Defines how to build and install this library, its dependencies, and metadata. Crucial for `pip install`.
    * `src/mineral_mapper_core/`: Contains the actual Python code organized as a package.
    * `tests/`: Contains automated tests to ensure core logic works correctly.
    * `README.md`: Explains what the library does, how to install it (`pip install .` or `pip install git+...`), and how to use its functions (API documentation).
    * `LICENSE`: Specifies the open-source terms of use.

## 3. Development Workflow 

### Phase 1: Setup & Core Data Handling

* **Goal:** Establish installable package structure, implement reliable I/O, handle file bumping, fix plotting.
* **Tasks:**
    * 1.1: **Setup Package Structure:** Create `src/` layout, `pyproject.toml`, `tests/`, `LICENSE`, dev `requirements.txt`, `.gitignore`.
    * 1.2: **Implement `core.data_io.load_element_maps`:** Loading, validation, error handling (Done - requires moving to `src/`).
    * 1.3: **Implement `core.data_io.save_results`:** Initial save logic (Done - requires moving to `src/`).
    * 1.4: **Refine `save_results` Plotting:** Fix colorbar (e.g., using `ListedColormap` + Colorcet). Ensure TIFF default. *(Next Step)*
    * 1.5: **Refine `save_results` File Bumping:** Implement logic to avoid overwriting output files by appending `(n)`.
    * 1.6: **Refine `save_results` Signature:** Allow passing optional phase names/color maps for incorporation into outputs.
    * 1.7: **Testing:** Add unit tests for `data_io` functions in `tests/`.

### Phase 2: Core Processing, Stats & Plotting

* **Goal:** Implement core ML, stats calculations, and summary plotting functions.
* **Tasks:**
    * 2.1: **Implement `core.processing.scale_data`:** StandardScaler implementation (Done - requires moving to `src/`).
    * 2.2: **Implement `core.processing.determine_optimal_k`:** Silhouette score method (Done - requires moving to `src/`). Note limitations.
    * 2.3: **Implement `core.processing.perform_clustering`:** K-Means + optional null phase (Done - requires moving to `src/`).
    * 2.4: **Implement `core.statistics.calculate_phase_stats`:** Calculate proportions, means, std devs (Done - requires moving to `src/`).
    * 2.5: **Implement `core.plotting.plot_statistics_comparison`:** Create functions to generate and save summary plots from `stats_df`.
    * 2.6: **Testing:** Add unit/integration tests for `processing`, `statistics`, `plotting` functions in `tests/`.

### Phase 3: Testing, Refinement & Documentation

* **Goal:** Ensure library robustness, refactor, document API.
* **Tasks:**
    * 3.1: **Integration Testing:** Test the full workflow using functions within the library context. Evaluate results on test data (`data/`).
    * 3.2: **Code Review & Refactoring:** Improve clarity, efficiency, comments within the `src/` code. Ensure a stable API.
    * 3.3: **Documentation:** Write/update `README.md` focusing on library installation and API usage (function arguments, return values, examples).

### Phase 4: Versioning & Maintenance

* **Goal:** Manage library versions and provide ongoing maintenance.
* **Tasks:**
    * 4.1: **Versioning:** Update version number (e.g., in `pyproject.toml` or `src/mineral_mapper_core/__init__.py`) according to changes (SemVer recommended). Use Git tags for releases.
    * 4.2: **Maintenance:** Address bugs reported in issues, consider feature requests for the core library.

## 4. Collaboration Plan

* Use Git feature branches -> Pull Requests -> `dev` branch.
* Use GitHub Issues for tracking bugs and features for the core library.
* Ensure automated tests pass before merging.
