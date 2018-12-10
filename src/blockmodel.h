#ifndef BP_SELECTION_BLOCKMODEL_H
#define BP_SELECTION_BLOCKMODEL_H

#include <random>
#include <utility>
#include <algorithm> // std::shuffle
#include <vector>
#include "types.h"

class blockmodel_t {
protected:
    std::uniform_real_distribution<> random_real;

public:
    // Ctor
    blockmodel_t() : random_real(0, 1) { ; }

    blockmodel_t(const uint_vec_t& memberships,
                 unsigned int Q,
                 unsigned int N,
                 unsigned int deg_corr_flag,
                 const adj_list_t* adj_list_ptr);

    /*
     * Note that we treat these input variables as first-class citizens for the blockmodel
     *
     * To do statistical inference, cab, pa will join in as new players.
     * There are several problems associated with these variables:
     * (1) Given cab, pa, Q, N, adj_list, but unknown memberships -> inference
     * (2) Given Q, adj_list, but unknown cab, pa -> learning
     *
     */

    size_t get_N() const noexcept;

    size_t get_Q() const noexcept;

    size_t get_E() const noexcept;

    unsigned int get_deg_corr_flag() const noexcept;

    const adj_list_t* get_adj_list_ptr() const noexcept;

    const uint_mat_t* get_m() const noexcept;

    /// BP
    unsigned int get_graph_max_degree() const noexcept;

    void shuffle(std::mt19937& engine) noexcept;

    int_vec_t get_k(unsigned int vertex) const noexcept;

    bool are_connected(unsigned int vertex_a, unsigned int vertex_b) const noexcept;

    int_vec_t get_size_vector() const noexcept;

    int_vec_t get_degree_size_vector() const noexcept;

    int_vec_t get_degree() const noexcept;

    uint_vec_t get_memberships() const noexcept;


    double get_entropy_from_degree_correction() const noexcept;

private:
    /// State variable
    const adj_list_t* adj_list_ptr_;
    int_mat_t k_;
    uint_mat_t p_;
    int_vec_t n_;
    int_vec_t deg_n_;
    int_vec_t deg_;
    uint_mat_t m_;
    uint_vec_t memberships_;
    unsigned int num_edges_;
    unsigned int graph_max_degree_;
    unsigned int deg_corr_flag_;
    double entropy_from_degree_correction_;

    /// Internal distribution. Generator must be passed as a service
    std::uniform_int_distribution<> random_block_;
    std::uniform_int_distribution<> random_node_;

    /* Compute the degree matrix from scratch. */
    void compute_k() noexcept;
    void compute_m() noexcept;

    void compute_bp_params_from_memberships() noexcept;

    uint_mat_t compute_e_rs() noexcept;

    // BP
    double_mat_t cab_;
    double_mat_t logcab_;
    double_mat_t pab_;
    uint_vec_t na_;
    uint_vec_t nna_;
    double_vec_t eta_;
    double_vec_t logeta_;

};

/* parameters for bp inference and learning */
bp_blockmodel_state bp_param_from_rand(const blockmodel_t& blockmodel, std::mt19937& engine) noexcept;

bp_blockmodel_state bp_param_from_epsilon_c(const blockmodel_t& blockmodel, double epsilon, double c) noexcept;

bp_blockmodel_state bp_param_from_direct(const blockmodel_t& blockmodel, const double_vec_t& pa, const double_vec_t& cab) noexcept;

bp_blockmodel_state bp_param_from_file(const blockmodel_t& blockmodel, std::string cab_file) noexcept;

#endif //BP_SELECTION_BLOCKMODEL_H
