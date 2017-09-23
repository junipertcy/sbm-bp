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

    blockmodel_t(const uint_vec_t &memberships,
                 unsigned int Q,
                 unsigned int N,
                 unsigned int deg_corr_flag,
                 adj_list_t *adj_list_ptr);

    /*
     * Note that we treat these input variables as first-class citizens for the blockmodel
     *
     * To do statistical inference, cab, pa will join in as new players.
     * There are several problems associated with these variables:
     * (1) Given cab, pa, Q, N, adj_list, but unknown memberships -> inference
     * (2) Given Q, adj_list, but unknown cab, pa -> learning
     *
     */

    unsigned int get_N() const;

    unsigned int get_Q() const;

    unsigned int get_E() const;

    unsigned int get_deg_corr_flag() const;

    adj_list_t *get_adj_list_ptr() const;

    uint_mat_t get_m() const;

    /// BP
    unsigned int get_graph_max_degree() const;

    void shuffle(std::mt19937 &engine);

    int_vec_t get_k(unsigned int vertex) const;

    bool are_connected(unsigned int vertex_a, unsigned int vertex_b) const;

    int_vec_t get_size_vector() const;

    int_vec_t get_degree_size_vector() const;

    int_vec_t get_degree() const;

    uint_vec_t get_memberships() const;


    double get_entropy_from_degree_correction() const;

private:
    /// State variable
    adj_list_t *adj_list_ptr_;
    int_mat_t k_;
    uint_mat_t p_;
    int_vec_t n_;
    int_vec_t deg_n_;
    int_vec_t deg_;
    uint_vec_t memberships_;
    unsigned int num_edges_;
    unsigned int graph_max_degree_;
    unsigned int deg_corr_flag_;
    double entropy_from_degree_correction_;

    /// Internal distribution. Generator must be passed as a service
    std::uniform_int_distribution<> random_block_;
    std::uniform_int_distribution<> random_node_;

    /* Compute the degree matrix from scratch. */
    void compute_k();

    void compute_bp_params_from_memberships();

    uint_mat_t compute_e_rs();

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
bp_blockmodel_state bp_param_from_rand(const blockmodel_t &blockmodel, std::mt19937 &engine);

bp_blockmodel_state bp_param_from_epsilon_c(const blockmodel_t &blockmodel, double epsilon, double c);

bp_blockmodel_state bp_param_from_direct(const blockmodel_t &blockmodel, double_vec_t pa, double_vec_t cab);

bp_blockmodel_state bp_param_from_file(const blockmodel_t &blockmodel, std::string cab_file);

#endif //BP_SELECTION_BLOCKMODEL_H
