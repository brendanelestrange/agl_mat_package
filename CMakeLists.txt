cmake_minimum_required(VERSION 3.12)
project(agl_tmc_cpp)

# --- 1. OPTIMIZATION FLAGS ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/O2 /Ob2)
else()
    add_compile_options(-O3 -march=native -fPIC)
endif()


find_package(pybind11 REQUIRED)

# B. Find Eigen3
find_package(Eigen3 REQUIRED)

# C. Find OpenMP
find_package(OpenMP REQUIRED)

# D. Find Chemfiles (Manual Search in Conda)
# We look for the physical file
find_library(CHEMFILES_LIB_FILE 
    NAMES chemfiles 
    # Look in the current Conda environment
    PATHS $ENV{CONDA_PREFIX}/lib 
    NO_DEFAULT_PATH
)

# Also check standard paths if not found in Conda
if(NOT CHEMFILES_LIB_FILE)
    find_library(CHEMFILES_LIB_FILE NAMES chemfiles)
endif()

if(NOT CHEMFILES_LIB_FILE)
    message(FATAL_ERROR "Could not find chemfiles library. Ensure 'conda install -c conda-forge chemfiles' is run.")
else()
    message(STATUS "Found Chemfiles: ${CHEMFILES_LIB_FILE}")
endif()

# --- 3. CREATE PYTHON MODULE ---
pybind11_add_module(${PROJECT_NAME} mat_agl_binding.cpp)

# --- 4. INCLUDE DIRECTORIES ---
# We need to explicitly include the Conda include path for headers
target_include_directories(${PROJECT_NAME} PRIVATE 
    $ENV{CONDA_PREFIX}/include
)

# --- 5. LINK LIBRARIES ---
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${CHEMFILES_LIB_FILE}  # <--- FIX: Link the FILE PATH, not the ::target name
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)