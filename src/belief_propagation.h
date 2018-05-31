//
// Created by Tzu-Chi Yen on 3/13/17.
//

#ifndef BP_SELECTION_BELIEF_PROPAGATION_H
#define BP_SELECTION_BELIEF_PROPAGATION_H

#include <cmath>
#include <vector>
#include <iostream>
#include "types.h"
#include "blockmodel.h"
#include "output_functions.h"

static double entropy(double_vec_t dist, unsigned int size) noexcept;


class belief_propagation {
protected:
    std::uniform_real_distribution<> random_real;

    /// blockmodel parameters;
    unsigned int N_;  // number of nodes
    unsigned int Q_;  // number of blocks
    double beta_;  // temperature
    unsigned int deg_corr_flag_;
    const adj_list_t *adj_list_ptr_;  // edgelist
    double_mat_t cab_;
    double_mat_t logcab_;
    double_mat_t pab_;
    uint_vec_t na_;
    uint_vec_t nna_;
    double_vec_t eta_;
    double_vec_t logeta_;
    double_mat_t cab_expect_;
    double_vec_t na_expect_;
    double_vec_t nna_expect_;
    uint_vec_t conf_true_;
    int_vec_t conf_planted_;  // used when initiating messages; equals -1 means the planted configuration is unknown.

    uint_mat_t graph_neis_inv_;  // TODO: rename it with a better name
    uint_mat_t graph_neis_;

    // belief propagation parameters; temporary variables to ensure normalization
    double_mat_t real_psi_;  // the marginal probability of node i (in N) at block q (in Q), normalized
    double _real_psi_total_;  // renamed from normtot_real
    double_vec_t _real_psi_q_;  // renamed from pom_psi

    double_vec_t _diff_real_psi_q; // temporary variable to do numerical differential
    const double _diff_beta = 1.000001; // temporary variable to do numerical differential

    double_vec_t h_;
    double_vec_t diff_h_;

    // These variables are allocated before BP starts
    double_vec_t conf_infer_;
    double_vec_t argmax_marginals_;
    double_vec_t exph_;  // appears in Eq-26
    double_vec_t maxpom_psii_iter_;
    double_vec_t field_iter_;
    double_vec_t _mmap_total_;  // renamed from normtot_psi_
    double_mat_t _mmap_q_nb_;  // renamed from psii_iter_

    // the marginal probability of node i (in N)'s neighbor j (in neighbor[i]) at block q (in Q), normalized
    double_tensor_t mmap_;  // message_map, renamed from psi

    const unsigned int LARGE_DEGREE = 50;
    const double EPS = 1.0e-50;

    // special needs
    bool output_marginals_;
    bool output_conditional_pairwise_entropies_;
    bool output_free_energy_;
    bool output_weighted_free_energy_;
    bool output_entropy_;
    bool output_weighted_entropy_;

    // EM_learning
    void learning_step(float learning_rate) noexcept;

    void clean_mmap_total_at_node_i_(unsigned int node_idx) noexcept;

    /// step-3 and step-11
    bool init_h() noexcept;

    bool update_h(unsigned int node_idx, int mode) noexcept;

    bool update_exph_with_h() noexcept;

    void compute_diff_h(double beta) noexcept;

public:
    belief_propagation() : random_real(0, 1) { ; }

    // Virtual methods: bogus virtual implementation
    virtual double bp_iter_update_psi(unsigned int i, double dumping_rate) noexcept { return 0; }

    // Common methods
    void init_messages(const blockmodel_t &blockmodel,
                       unsigned int bp_messages_init_flag,
                       const int_vec_t &conf,
                       const uint_vec_t &true_conf,
                       std::mt19937 &engine) noexcept;

    void init_special_needs(bool if_output_marginals,
                            bool if_output_conditional_pairwise_entropies,
                            bool if_output_free_energy,
                            bool if_output_weighted_free_energy,
                            bool if_output_entropy,
                            bool if_output_weighted_entropy) noexcept;

    void bp_allocate(const blockmodel_t &blockmodel) noexcept;

    void expand_bp_params(const bp_blockmodel_state &state) noexcept;

    void sum_all_messages_to_i(unsigned int i) noexcept;

    double norm_m_at_i(unsigned int i, double dumping_rate) noexcept;

    void learning(const blockmodel_t &blockmodel,
                  const bp_blockmodel_state &state,
                  float learning_conv_crit,
                  unsigned int learning_max_time,
                  float learning_rate,
                  float dumping_rate,
                  std::mt19937 &engine) noexcept;

    int converge(float conv_crit,
                 unsigned int time_conv,
                 float dumping_rate,
                 std::mt19937 &engine) noexcept;

    void inference(const blockmodel_t &blockmodel,
                   const bp_blockmodel_state &state,
                   float conv_crit,
                   unsigned int time_conv,
                   float dumping_rate,
                   std::mt19937 &engine) noexcept;


    double bp_iter_update_psi_large_degree(unsigned int i,
                                           double dumping_rate) noexcept;

    void set_beta(double beta) noexcept;

    double compute_free_energy(bool by_site) noexcept;

    double compute_entropy(bool by_site) noexcept;

    double compute_overlap() noexcept;

    /// EM methods will be listed here!!!

private:
    void compute_cab_expect() noexcept;

    void compute_na_expect() noexcept;

    double compute_free_energy_site(bool by_site) noexcept;

    double compute_free_energy_edge(bool by_site) noexcept;

    double compute_free_energy_nonedge(bool by_site) noexcept;

    double compute_entropy_site(bool by_site) noexcept;

    double compute_entropy_edge(bool by_site) noexcept;

    double compute_entropy_nonedge(bool by_site) noexcept;

    void compute_marg_entropy() noexcept;

    double compute_e_nishimori() noexcept;

    // in `converge` function
    double maxdiffm_ = 0.;

    // in `compute_entropy` functions
    double numerator_ = 0;
    double_vec_t numerator_edge_;  // to store the conditional pairwise entropy (experimental)
    double denominator_ = 0.;

    double a_ = 0;
    double b_ = 0;

    // collecting outputs
    double_vec_t bethe_ent_;  // Bethe entropy for each node
    double_vec_t marg_ent_;  // marginal entropy for the uncertainty of the labels for each node
    double_vec_t free_ene_;  // equilibrated Bethe free energy of each node
    double_vec_t free_ene_site_;
    double_vec_t free_ene_edge_;
    double_vec_t bethe_ent_site_;
    double_vec_t bethe_ent_edge_;
//    double_mat_t cond_pair_ent_;  // to store the conditional pairwise entropy, in same size as `graph_neis_`

};

/* Inherited classes with specific definitions */
class bp_basic : public belief_propagation {
public:
    double bp_iter_update_psi(unsigned int i, double dumping_rate) noexcept override;
};

class bp_conditional : public belief_propagation {
public:
    double bp_iter_update_psi(unsigned int i, double dumping_rate) noexcept override;
};

#endif //BP_SELECTION_BELIEF_PROPAGATION_H
