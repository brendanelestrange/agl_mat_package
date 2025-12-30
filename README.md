# AGL-MAT-Score (Accelerated)

**Fast, C++ accelerated calculation of Multiscale Weighted Colored Subgraphs (MWCGS) for protein-ligand complexes.**

This package calculates algebraic graph features (AGL-MAT-Score) for transition metal interfaces. It utilizes a hybrid C++/Python architecture (via Pybind11 and Eigen) to achieve orders of magnitude speedup compared to pure Python implementations.

---

## 🛠 Prerequisites

Before building the package, you **must** ensure the following C++ libraries and environments are set up.

### 1. System Dependencies (OpenMP)
OpenMP is required for parallelization support in the C++ layer.

* **macOS:**
    ```bash
    brew install libomp
    ```
* **Ubuntu/Debian:**
    ```bash
    sudo apt-get install libomp-dev
    ```

### 2. Conda Environment (Chemfiles & Python Libs)
The easiest way to handle the dependency on `Chemfiles` (required for PDB/Mol2 parsing) is via Conda/Mamba.

```bash
# Create and activate the environment
conda create -n agl_fast
conda activate agl_fast

# Install C++ dependencies
conda install -c conda-forge chemfiles  compilers llvm-openmp

# Install Python dependencies
conda install xgboost scikit-learn numpy pandas
```

---

## 🚀 Installation & Build

Once dependencies are installed, you need to compile the C++ shared library using the provided build script.

**Note:** You may need to adjust paths in `CMakeLists.txt` depending on your specific system architecture or Conda install path.

1.  **Navigate to the base directory:**
    ```bash
    cd /path/to/agl-score-accelerated
    ```

2.  **Run the build script:**
    ```bash
    chmod +x ./make.sh
    ./make.sh
    ```

---

## 💻 Usage

You can generate features using the `get_agl_mat_features.py` script located in the `src` directory. This implementation mirrors the API of the original AGL-Score but runs significantly faster.

### Example Command
This example assumes you have downloaded data from **PDBBind**.

```bash
python src/get_agl_mat_features.py \
  -k 112 \
  -c 12 \
  -m Adjacency \
  -f '../data/tmQMg_properties_and_targets.csv' \
  -dd '../data/tmQMg' \
  -fd '../Features'
```

### Arguments Key
| Flag | Description |
| :--- | :--- |
| `-k` | Kernel scale/cutoff parameter. |
| `-c` | Number of colors/types for the subgraph. |
| `-m` | Matrix type (e.g., `Adjacency`, `Laplacian`). |
| `-f` | Path to the input CSV file containing PDB IDs. |
| `-dd` | **Data Directory:** Path to the folder containing PDB/Mol2 files. |
| `-fd` | **Feature Directory:** Output path for the generated features. |

---

## 📂 Project Structure

* `src/`: Contains Python drivers and C++ source files.
* `make.sh`: Build script to invoke CMake and Make.
* `CMakeLists.txt`: Configuration for the build system (may require manual editing).
