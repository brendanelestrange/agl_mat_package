#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <chemfiles.hpp>
#include <Eigen/Dense>
#include <omp.h>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::string;

// --- 1. DEFINITIONS & CONSTANTS ---

// Optimized set for O(1) lookups
const std::unordered_set<string> TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"
};

const vector<string> LIGAND_ELEMS = {
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"
};

// Merged Bondi (1964) and Alvarez (2013) Radii
std::unordered_map<string, float> vwals_rad {
    {"H", 1.20},  {"C", 1.70},  {"N", 1.55},  {"O", 1.52},  {"F", 1.47},
    {"P", 1.80},  {"S", 1.80},  {"Cl", 1.75}, {"Br", 1.85}, {"I", 1.98},
    {"B", 0.85},  {"Si", 2.10}, {"As", 1.85}, {"Se", 1.90}, {"Te", 1.40},
    // Transition Metals (Alvarez)
    {"Sc", 2.11}, {"Ti", 2.00}, {"V", 2.07},  {"Cr", 2.00}, {"Mn", 2.00}, 
    {"Fe", 2.00}, {"Co", 2.00}, {"Ni", 1.63}, {"Cu", 1.40}, {"Zn", 1.39},
    {"Y",  2.19}, {"Zr", 2.06}, {"Nb", 2.07}, {"Mo", 2.09}, {"Tc", 2.09}, 
    {"Ru", 2.05}, {"Rh", 2.00}, {"Pd", 2.10}, {"Ag", 2.11}, {"Cd", 2.18},
    {"La", 2.40}, {"Hf", 2.23}, {"Ta", 2.22}, {"W",  2.18}, {"Re", 2.05}, 
    {"Os", 2.00}, {"Ir", 2.00}, {"Pt", 1.75}, {"Au", 2.17}, {"Hg", 1.50}
};

const float SIGMA = 0.2127; 

struct Point { float x, y, z; };

struct trajFile {
    chemfiles::Frame frame;
    chemfiles::Topology topology;
    auto get_positions() const { return frame.positions(); }
};


float get_radius(const string& atom_type) {
    if (vwals_rad.count(atom_type)) return vwals_rad.at(atom_type);
    return 1.70f;
}

VectorXd filter_pos(const VectorXd &values) {
    std::vector<double> temp_list;
    for (int i = 0; i < values.size(); i++) {
        if (values[i] > 1e-9) temp_list.push_back(values[i]);
    }
    return Eigen::Map<VectorXd>(temp_list.data(), temp_list.size());
}

float distance_calc(const Point& first, const Point& second) {
    double dx = first.x - second.x;
    double dy = first.y - second.y;
    double dz = first.z - second.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float kernel(float dist, float vdw_rad, float kappa, float tau, const string& k_type) {
    float eta = tau * vdw_rad;
    char type_char = std::tolower(k_type[0]); 

    if (type_char == 'e') return std::exp( -std::pow(dist / eta, kappa) );
    else if (type_char == 'l') return 1.0f / (1.0f + std::pow(dist / eta, kappa));
    else if (type_char == 'r') return std::exp( -std::pow(dist, 2) / (2.0f * std::pow(eta, 2)) );
    else return std::exp( -std::pow(dist / eta, kappa) );
}

vector<float> calculate_stats(const VectorXd &raw_eigens, float counts) {
    VectorXd eigens = filter_pos(raw_eigens);
    
    if (eigens.size() == 0) {
        vector<float> empty_vec(10, 0.0f);
        empty_vec[0] = counts; 
        return empty_vec;
    }

    float mean = eigens.mean();
    float sum = eigens.sum();
    float variance = (eigens.array() - mean).square().sum() / (eigens.size()); 
    float std_dev = std::sqrt(variance);

    float median_val;
    int n = eigens.size();
    VectorXd sorted = eigens;
    std::sort(sorted.data(), sorted.data() + sorted.size());

    if (n % 2 == 1) median_val = sorted(n / 2);
    else median_val = (sorted(n / 2 - 1) + sorted(n / 2)) / 2.0;

    return { counts, sum, (float)eigens.minCoeff(), (float)eigens.maxCoeff(), mean, median_val, std_dev, variance, (float)eigens.size(), (float)eigens.squaredNorm() };
}

vector<float> analyze_metal_ligand_pair(
    const trajFile &file, 
    int metal_idx, 
    const vector<int>& ligand_indices,
    string metal_type, 
    string ligand_type, 
    float cutoff, float kappa, float tau, string kernel_type
) {
    if (ligand_indices.empty()) return vector<float>(10, 0.0f);

    auto positions = file.get_positions();
    Point p_metal = { (float)positions[metal_idx][0], (float)positions[metal_idx][1], (float)positions[metal_idx][2] };
    
    vector<int> valid_ligands;
    for (int idx : ligand_indices) {
        Point p_lig = { (float)positions[idx][0], (float)positions[idx][1], (float)positions[idx][2] };
        if (distance_calc(p_metal, p_lig) <= cutoff) {
            valid_ligands.push_back(idx);
        }
    }

    float total_pair_count = (float)valid_ligands.size();
    if (valid_ligands.empty()) return vector<float>(10, 0.0f);

    int N = 1 + valid_ligands.size();
    MatrixXd adj_mat(N, N);
    adj_mat.setZero();

    float r_metal = get_radius(metal_type);
    float r_lig = get_radius(ligand_type);
    float eta = r_metal + r_lig;
    float covalent_restriction = eta + SIGMA;

    bool has_interaction = false;

    for (int j = 0; j < valid_ligands.size(); j++) {
        int lig_real_idx = valid_ligands[j];
        Point p_lig = { (float)positions[lig_real_idx][0], (float)positions[lig_real_idx][1], (float)positions[lig_real_idx][2] };
        
        float d_calc = distance_calc(p_metal, p_lig);

        if (d_calc <= covalent_restriction) {
            float weight = kernel(d_calc, eta, kappa, tau, kernel_type);
            adj_mat(0, j + 1) = -weight; 
            adj_mat(j + 1, 0) = -weight; 
            has_interaction = true;
        }
    }

    if (!has_interaction) {
        vector<float> zeros(10, 0.0f);
        zeros[0] = total_pair_count;
        return zeros;
    }

    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(adj_mat); 
    return calculate_stats(eigensolver.eigenvalues(), total_pair_count);
}

vector<float> get_tmc_scores(string xyz_file, float cutoff, float kappa, float tau, string kernel_type) {
    chemfiles::set_warning_callback([](std::string){});

    chemfiles::Trajectory trajectory(xyz_file);
    trajFile file;
    file.frame = trajectory.read(); 
    file.topology = file.frame.topology();

    vector<int> metal_indices;
    std::map<string, vector<int>> ligand_groups;
    
    for (const string& l : LIGAND_ELEMS) {
        ligand_groups[l] = vector<int>{};
    }

    for (size_t i = 0; i < file.frame.size(); i++) {
        string atom_type = file.topology[i].name(); 
        
        atom_type.erase(std::remove_if(atom_type.begin(), atom_type.end(), ::isdigit), atom_type.end());
        
        if (atom_type.length() > 0) atom_type[0] = toupper(atom_type[0]);
        if (atom_type.length() > 1) atom_type[1] = tolower(atom_type[1]);

        if (TRANSITION_METALS.count(atom_type)) {
            metal_indices.push_back(i);
        } else if (ligand_groups.count(atom_type)) {
            ligand_groups[atom_type].push_back(i);
        }
    }

    vector<std::tuple<int, string, string>> tasks;
    
    for (int m_idx : metal_indices) {
        string m_type = file.topology[m_idx].name();

        if (m_type.length() > 0) m_type[0] = toupper(m_type[0]);
        if (m_type.length() > 1) m_type[1] = tolower(m_type[1]);

        for (const string& l_type : LIGAND_ELEMS) {
            tasks.push_back(std::make_tuple(m_idx, m_type, l_type));
        }
    }

    vector<float> global_features(tasks.size() * 10);

    py::gil_scoped_release release;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < tasks.size(); i++) {
        int m_idx = std::get<0>(tasks[i]);
        string m_type = std::get<1>(tasks[i]);
        string l_type = std::get<2>(tasks[i]);

        vector<float> feats = analyze_metal_ligand_pair(
            file, m_idx, ligand_groups[l_type], 
            m_type, l_type, cutoff, kappa, tau, kernel_type
        );

        size_t offset = i * 10;
        for (int k = 0; k < 10; k++) {
            global_features[offset + k] = feats[k];
        }
    }
    
    py::gil_scoped_acquire acquire;
    return global_features;
}

PYBIND11_MODULE(agl_tmc_cpp, m) {
    m.doc() = "TMC Spectral Features (AGL-Score adaptation for XYZ)";

    m.def("get_tmc_scores", &get_tmc_scores, "Calculate Metal-Ligand Spectral Features",
          py::arg("xyz_file"), 
          py::arg("cutoff") = 6.0,
          py::arg("kappa") = 6.0,
          py::arg("tau") = 2.0,
          py::arg("kernel_type") = "exponential");
}