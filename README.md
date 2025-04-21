# Unsupervised Mineral Phase Mapper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) A Python application with a graphical user interface (GUI) designed to perform unsupervised machine learning (K-Means clustering) on elemental map data (e.g., from XRF, EDS, SEM) to identify and quantify distinct phases within a sample. Primarily aimed at researchers and analysts in geoscience, materials science, and related fields.

This tool can be run directly from source code or used as a standalone executable (recommended for most users).

## Features

* **Intuitive GUI:** Simple interface built with Tkinter for ease of use.
* **Flexible Input:** Accepts elemental maps as `.bmp` or `.txt` files.
* **Automated Processing:** Reads all compatible maps from a specified input directory.
* **K-Means Clustering:** Applies K-Means algorithm to group pixels with similar elemental compositions into phases. Further clustering methods (e.g., Gaussian Mixture Models, DBSCAN, etc.) will be investigated at a later date
* **Automatic 'k' Detection:** Option to automatically determine the optimal number of clusters (phases) using the Silhouette Score method.
* **Manual 'k' Override:** Allows users to specify the desired number of clusters.
* **Outlier Handling:** Optional feature to identify and isolate outlier/mixed pixels into a distinct "Null Phase".
* **Visual Output:** Generates a color-coded phase map image (`phase_map.png`) for easy visualization.
* **Quantitative Output:** Produces a statistics summary file (`phase_statistics.csv`) detailing phase proportions and elemental statistics (mean, std dev) per phase.
* **Standalone Executable:** Available for download, allowing use without installing Python or dependencies.
* **Modular Codebase:** Designed with separate modules for data handling, processing, statistics, and GUI for better maintainability (for developers).

## Installation and Setup

Choose the option that best suits your needs:

### Option 1: Using the Pre-built Executable (Recommended for End-Users)

1.  **Download:** Obtain the latest executable file for your operating system (Windows, macOS, Linux) from the **[Releases Page](<link-to-your-github-releases-page>)** of this repository.
2.  **Place:** Save the downloaded executable file to a convenient location on your computer.
3.  **Run:** No installation is needed. Simply double-click the executable file to launch the application (see Usage section below).

*(Note: You do not need Python or any specific libraries installed to use the pre-built executable.)*

### Option 2: Running from Source (For Developers or Modification)

1.  **Prerequisites:**
    * Python (3.8 or higher recommended)
    * Git (for cloning the repository)

2.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd mineral_phase_mapper
    ```

3.  **Create and activate a virtual environment (Recommended):**
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

4.  **Install dependencies:**
    Make sure you are in the project's root directory (where `requirements.txt` is located).
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Launching the Application

* **If using the Executable:** Navigate to the folder where you saved the executable and double-click the file (e.g., `mineral_mapper.exe` or similar).
* **If running from Source:**
    1.  Open your terminal or command prompt.
    2.  Navigate to the project's root directory (`mineral_phase_mapper`).
    3.  Activate your virtual environment (if created): `source venv/bin/activate` or `.\venv\Scripts\activate`.
    4.  Run the application script: `python run_mapper.py`.

### Using the GUI

Once the application window is open:

1.  **Select Input Directory:** Click "Browse..." and choose the folder containing all the elemental map files (`.bmp`, `.txt`) for **a single sample**.
2.  **Select Output Directory:** Click "Browse..." and choose a folder where the results (`phase_map.png`, `phase_statistics.csv`) will be saved. It's recommended to use an empty or new directory.
3.  **Number of Phases (k):**
    * Choose **Automatic** (Recommended) for automatic determination of `k`.
    * Choose **Manual** and enter an integer (>= 2) if you want to specify the number of phases.
4.  **Add 'Null Phase':** Check this box to attempt to isolate outlier/mixed pixels.
5.  **Run Analysis:** Click the "Run Analysis" button. Monitor progress in the Status area.
6.  **Results:** A message box will confirm completion. Output files will be in the specified output directory.

## Input Data Format

* Place all elemental map files for a single sample in one input folder.
* **All map files MUST have the exact same pixel dimensions.**
* The base filename (e.g., `Fe` from `Fe.bmp`) is used as the element name.
* **`.bmp` files:** Grayscale preferred. RGB images may be handled by using the first channel *(confirm/modify)*. Palette-based images attempt conversion to grayscale.
* **`.txt` files:** Space- or tab-delimited numerical data representing the map grid.

## Output Description

1.  **`phase_map.png`:** Image showing spatial distribution of phases, colored by phase ID, with a color bar.
2.  **`phase_statistics.csv`:** CSV file with columns: `Phase`, `Proportion (%)`, `Mean_[Element]`, `StdDev_[Element]` for each identified phase.

## Building the Executable from Source (For Developers)

If you want to build the executable yourself:

1.  **Setup:** Follow the steps in "Installation and Setup - Option 2: Running from Source" to set up the environment and install dependencies.
2.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```
3.  **Build Command:** Navigate to the project root directory in your terminal (where `run_mapper.py` is located) and run:
    ```bash
    pyinstaller --windowed --onefile run_mapper.py
    ```
    *(Note: For complex cases, especially with scientific libraries, you might need to modify the command or use a `.spec` file to handle hidden imports or include necessary data files. Refer to PyInstaller documentation.)*
4.  **Output:** The standalone executable will be created in the `dist` directory.

## Contributing

*[Optional: Add contribution guidelines here or link to CONTRIBUTING.md]*

## License

This project is licensed under the MIT License. See the LICENSE file for details. 

## Contact / Support

For support or further information, please email:
adamjnissen@gmail.com
---