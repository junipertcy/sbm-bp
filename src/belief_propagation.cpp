#include "belief_propagation.h"
#include <cassert>

static double entropy(double_vec_t dist, unsigned int size) noexcept {
    double temp = 0.0;
    for (unsigned int i = 0; i < size; ++i) {
        if (dist[i] > 0) {
            temp -= dist[i] * log(dist[i]);// log: natural logarithm
        }
    }
    return temp;
};

void belief_propagation::learning(const blockmodel_t &blockmodel,
                                  const bp_blockmodel_state &state,
                                  float learning_conv_crit,
                                  unsigned int learning_max_time,
                                  float learning_rate,
                                  float dumping_rate,
                                  std::mt19937 &engine) noexcept {
    expand_bp_params(state);

    double fold = 0.0;
    int learning_time = 0;

    double fdiff = 1.0;
    for (learning_time = 0; learning_time < learning_max_time; learning_time++) {
        if (fdiff < learning_conv_crit) learning_conv_crit *= 0.1;
        converge(learning_conv_crit, learning_max_time, dumping_rate, engine);
        compute_na_expect();
        compute_cab_expect();

        double fnew = compute_free_energy();

        fdiff = fabs(fnew - fold);
        fold = fnew;

        if (std::isnan(fold) || std::isinf(fold)) {
            std::clog << "Bethe energy is calculated as nan.\n";
            break;
        }
        if (fdiff < learning_conv_crit) {
            std::clog << "Algorithm stop because of fdiff < learning_conv_crit. [which is good]\n";
            break;
        }
        learning_step(learning_rate);
    }
    output_vec<double_vec_t>(eta_, std::cout);
    output_mat<double_mat_t>(cab_, std::cout);
    std::clog << "overlap:" << compute_overlap() << "\n";
}

void belief_propagation::learning_step(float learning_rate) noexcept {
    /*
     * Due to the incorporation of the learning rate, one may not expect na_ to sum to one.
     * We fix this issue here.
     */
    auto _N = N_;
    for (unsigned int i = 0; i < Q_ - 1; ++i) {
        na_[i] = unsigned(int(learning_rate * na_expect_[i] + (1.0 - learning_rate) * na_[i]));
        _N -= na_[i];
    }
    na_[Q_ - 1] = _N;

    for (unsigned int i = 0; i < Q_; ++i) {
        eta_[i] = double(na_[i]) / N_;
        logeta_[i] = std::log(eta_[i]);

        for (unsigned int j = 0; j < Q_; ++j) {
            cab_[i][j] = learning_rate * cab_expect_[i][j] + (1.0 - learning_rate) * cab_[i][j];
            logcab_[i][j] = std::log(cab_[i][j]);
            pab_[i][j] = cab_[i][j] / N_;
        }
    }
}

void belief_propagation::inference(const blockmodel_t &blockmodel,
                                   const bp_blockmodel_state &state,
                                   float conv_crit,
                                   unsigned int time_conv,
                                   float dumping_rate,
                                   std::mt19937 &engine) noexcept {

    expand_bp_params(state);
    int niter = converge(conv_crit, time_conv, dumping_rate, engine);
    double f = compute_free_energy();
    double e = compute_entropy();
    std::cout << e << " " << f << " " << compute_overlap() << " " << niter << " \n";
    if (if_output_marginals_) {
        // This outputs the vertex marginals when BP is converged.
        // The marginal distribution can be used to compute the mean-field entropy
        // AND
        // the entropy of the average conditional distribution, i.e. margEntropy, or H(v).
        output_mat<double_mat_t>(real_psi_, std::cout);
        for (unsigned int v = 0; v < N_ ; ++v) {
            std::clog << "Node-" << v << "; margEntropy H(v) is " << entropy(real_psi_[v], Q_) << "\n";
        }
    }
}

void belief_propagation::init_messages(const blockmodel_t &blockmodel,
                                       unsigned int bp_messages_init_flag,
                                       const int_vec_t &conf,
                                       const uint_vec_t &true_conf,
                                       std::mt19937 &engine) noexcept {
    assert(bp_messages_init_flag < 4);
    bp_allocate(blockmodel);
    conf_true_ = true_conf;

    for (unsigned int i = 0; i < N_; ++i) {
        if (bp_messages_init_flag == 0) {  // initialize by random messages
            double norm = 0.0;
            for (unsigned int q = 0; q < Q_; ++q) {
                real_psi_[i][q] = random_real(engine);
                norm += real_psi_[i][q];
            }
            for (unsigned int q = 0; q < Q_; ++q) {
                real_psi_[i][q] /= norm;
            }
            for (unsigned int idxij = 0; idxij < adj_list_ptr_->at(i).size(); ++idxij) {
                unsigned int j = graph_neis_[i][idxij];
                unsigned int idxji = graph_neis_inv_[i][idxij];
                norm = 0.0;
                for (unsigned int q = 0; q < Q_; ++q) {
                    mmap_[j][idxji][q] = random_real(engine);
                    norm += mmap_[j][idxji][q];
                }
                for (unsigned int q = 0; q < Q_; ++q) {
                    mmap_[j][idxji][q] /= norm;
                }
            }
        } else if (bp_messages_init_flag == 1) { // initialize by partly planted configuration, others left random
            conf_planted_ = conf;
            double norm = 0.0;
            if (conf_planted_[i] != -1) {
                for (unsigned int q = 0; q < Q_; ++q) {
                    if (q == conf_planted_[i]) {
                        real_psi_[i][q] = 1.0;
                    } else {
                        real_psi_[i][q] = 0.0;
                    }
                }
            } else {
                for (unsigned int q = 0; q < Q_; ++q) {
                    real_psi_[i][q] = random_real(engine);
                    norm += real_psi_[i][q];
                }

                for (unsigned int q = 0; q < Q_; ++q) {
                    real_psi_[i][q] /= norm;
                }
            }
            for (unsigned int idxij = 0; idxij < adj_list_ptr_->at(i).size(); ++idxij) {
                unsigned int j = graph_neis_[i][idxij];
                unsigned int idxji = graph_neis_inv_[i][idxij];
                if (conf_planted_[i] != -1) {
                    for (unsigned int q = 0; q < Q_; ++q) {
                        if (q == conf_planted_[i]) {
                            mmap_[j][idxji][q] = 1.0;
                        } else {
                            mmap_[j][idxji][q] = 0.0;
                        }
                    }
                } else {
                    norm = 0.0;
                    for (unsigned int q = 0; q < Q_; ++q) {
                        mmap_[j][idxji][q] = random_real(engine);
                        norm += mmap_[j][idxji][q];
                    }
                    for (unsigned int q = 0; q < Q_; ++q) {
                        mmap_[j][idxji][q] /= norm;
                    }
                }
            }
        } else if (bp_messages_init_flag == 2) { // planted configuration, with random noise on all sites
            conf_planted_ = conf;
            float planted_noise = 0.1;  // TODO:
            for (unsigned int q = 0; q < Q_; ++q) {
                assert(conf_planted_[i] != 1);
                if (q == conf_planted_[i]) {
                    real_psi_[i][q] = planted_noise + (1.0 - planted_noise) * random_real(engine);
                } else {
                    real_psi_[i][q] = random_real(engine) * (1.0 - planted_noise);
                }
                for (unsigned int idxij = 0; idxij < adj_list_ptr_->at(i).size(); ++idxij) {
                    if (q == conf_planted_[i]) {
                        mmap_[i][idxij][q] = planted_noise + (1.0 - planted_noise) * random_real(engine);
                    } else {
                        mmap_[i][idxij][q] = random_real(engine) * (1.0 - planted_noise);
                    }
                }
                // TODO: check if the messages are normalized.
            }
        } else if (bp_messages_init_flag == 3) {  // fixed planted configuration
            conf_planted_ = conf;
            for (unsigned int q = 0; q < Q_; ++q) {
                assert(conf_planted_[i] != 1);
                if (q == conf_planted_[i]) {
                    real_psi_[i][q] = 1.0;
                } else {
                    real_psi_[i][q] = 0.000;
                }
            }
            for (unsigned int idxij = 0; idxij < adj_list_ptr_->at(i).size(); ++idxij) {
                unsigned int j = graph_neis_[i][idxij];
                unsigned int idxji = graph_neis_inv_[i][idxij++];
                for (unsigned int q = 0; q < Q_; ++q) {
                    if (q == conf_planted_[i]) {
                        mmap_[j][idxji][q] = 1.0;
                    } else {
                        mmap_[j][idxji][q] = 0.0;
                    }
                }
            }
        }
    }
}

void belief_propagation::init_special_needs(bool if_output_marginals) noexcept {
    if_output_marginals_ = if_output_marginals;
}

void belief_propagation::bp_allocate(const blockmodel_t &blockmodel) noexcept {
    Q_ = blockmodel.get_Q();
    N_ = blockmodel.get_N();
    adj_list_ptr_ = blockmodel.get_adj_list_ptr();
    deg_corr_flag_ = blockmodel.get_deg_corr_flag();

    unsigned int graph_max_degree = blockmodel.get_graph_max_degree();

    conf_infer_.resize(N_);
    argmax_marginals_.resize(N_);
    h_.resize(Q_);
    diff_h_.resize(Q_);

    /// temporary variables to ensure normalization
    _real_psi_q_.resize(Q_);
    _diff_real_psi_q.resize(Q_);

    /// Done
    exph_.resize(Q_, 0);
    mmap_.resize(N_);
    graph_neis_inv_.resize(N_);
    graph_neis_.resize(N_);

    unsigned int vertex_j;
    unsigned int idxji;
    for (unsigned int i = 0; i < N_; ++i) {
        mmap_[i].resize(adj_list_ptr_->at(i).size());

        graph_neis_inv_[i].resize(adj_list_ptr_->at(i).size());
        graph_neis_[i].resize(adj_list_ptr_->at(i).size());

        auto nb_of_i(adj_list_ptr_->at(i).begin());
        for (unsigned int idxij = 0; idxij < adj_list_ptr_->at(i).size(); ++idxij) {

            mmap_[i][idxij].resize(Q_);
            vertex_j = *nb_of_i;

            auto nb_of_j(adj_list_ptr_->at(vertex_j).begin());
            idxji = unsigned(int(std::distance(nb_of_j, adj_list_ptr_->at(vertex_j).find(i))));
            graph_neis_inv_[i][idxij] = idxji;
            graph_neis_[i][idxij] = vertex_j;
            advance(nb_of_i, 1);
        }
    }
    maxpom_psii_iter_.resize(graph_max_degree);
    field_iter_.resize(graph_max_degree);
    _mmap_total_.resize(graph_max_degree);

    psii_.resize(Q_);
    real_psi_.resize(N_);
    for (unsigned int i = 0; i < N_; i++) {
        real_psi_[i].resize(Q_);
    }

    _mmap_q_nb_.resize(Q_);
    cab_expect_.resize(Q_);
    for (unsigned int q = 0; q < Q_; q++) {
        _mmap_q_nb_[q].resize(graph_max_degree);
        cab_expect_[q].resize(Q_, 0.0);
    }

    conf_planted_.resize(N_, -1);
    na_expect_.resize(Q_);
    nna_expect_.resize(Q_);

}

void belief_propagation::expand_bp_params(const bp_blockmodel_state &state) noexcept {

    cab_ = state.cab;
    na_ = state.na;

    logcab_.resize(Q_);
    pab_.resize(Q_);
    nna_.resize(Q_);  // TODO: nna_ is not done yet
    eta_.resize(Q_);
    logeta_.resize(Q_);

    for (unsigned int i = 0; i < Q_; ++i) {
        logcab_[i].resize(Q_);
        pab_[i].resize(Q_);
    }

    for (unsigned int q = 0; q < Q_; ++q) {
        eta_[q] = 1.0 * na_[q] / N_;
        logeta_[q] = std::log(eta_[q]);

        for (unsigned int j = 0; j < Q_; ++j) {
            pab_[q][j] = cab_[q][j] / N_;
            pab_[j][q] = cab_[j][q] / N_;
            logcab_[q][j] = std::log(cab_[q][j]);
            logcab_[j][q] = std::log(cab_[j][q]);
        }
    }
}


bool belief_propagation::init_h() noexcept {
    /*
     * Attention: this function should only be executed once in "infer" mode.
     */
    for (unsigned int i = 0; i < Q_; ++i) {
        h_[i] = 0.0;
    }
    for (unsigned int i = 0; i < N_; ++i) {
        update_h(i, +1);
    }
    update_exph_with_h();
    return true;
}

bool belief_propagation::update_h(unsigned int node_idx, int mode) noexcept {
    assert(mode == -1 || mode == +1);
    double di = adj_list_ptr_->at(node_idx).size();

    if (mode == -1) {
        for (unsigned int q1 = 0; q1 < Q_; ++q1) {
            for (unsigned int q2 = 0; q2 < Q_; ++q2) {
                if (deg_corr_flag_ == 0) {
                    h_[q1] -= cab_[q2][q1] * real_psi_[node_idx][q2];
                } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
                    h_[q1] -= di * cab_[q2][q1] * real_psi_[node_idx][q2];
                }
            }
        }
    } else if (mode == +1) {
        for (unsigned int q1 = 0; q1 < Q_; ++q1) {
            for (unsigned int q2 = 0; q2 < Q_; ++q2) {
                if (deg_corr_flag_ == 0) {
                    h_[q1] += cab_[q2][q1] * real_psi_[node_idx][q2];
                } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
                    h_[q1] += di * cab_[q2][q1] * real_psi_[node_idx][q2];
                }
            }
        }
    }
    return true;
}


bool belief_propagation::update_exph_with_h() noexcept {
    for (unsigned int q = 0; q < Q_; ++q) {
        exph_[q] = std::exp(-beta_ * h_[q] / N_);  // TODO: check myexp and exp equivalance
    }
    return true;
}

void belief_propagation::compute_diff_h(double beta) noexcept {
    for (unsigned int i = 0; i < Q_; ++i) {
        diff_h_[i] = 0.0;
    }

    for (unsigned int i = 0; i < N_; ++i) {
        for (unsigned int q1 = 0; q1 < Q_; ++q1) {
            for (unsigned int q2 = 0; q2 < Q_; ++q2) {
                if (deg_corr_flag_ == 0) {
                    diff_h_[q1] += std::pow(cab_[q2][q1], beta) * real_psi_[i][q2];
                }
            }
        }
    }
}

int belief_propagation::converge(float bp_err,
                                 unsigned int max_iter_time,
                                 float dumping_rate,
                                 std::mt19937 &engine) noexcept {
    init_h();

    for (int iter_time = 0; iter_time < max_iter_time; ++iter_time) {
        double maxdiffm = -100.0;
        for (unsigned int iter_inter_time = 0; iter_inter_time < N_; ++iter_inter_time) {
            auto i = unsigned(int(random_real(engine) * N_));   // TODO: unify the expression
            double diffm;
            if (adj_list_ptr_->at(i).size() >= LARGE_DEGREE) {
                diffm = bp_iter_update_psi_large_degree(i, dumping_rate);
            } else {
                diffm = bp_iter_update_psi(i, dumping_rate);
            }
            if (diffm > maxdiffm) {
                maxdiffm = diffm;
            }
        }
        if (maxdiffm < bp_err) {
            // TODO: put them as __private__ variable
            // int bp_last_conv_time = iter_time;
            // double bp_last_diff = maxdiffm;
            return iter_time;
        }
    }

    return -1;
}

void belief_propagation::set_beta(double beta) noexcept {
    beta_ = beta;
}


void belief_propagation::clean_mmap_total_at_node_i_(unsigned int node_idx) noexcept {
    for (unsigned int j = 0; j < adj_list_ptr_->at(node_idx).size(); ++j) {
        _mmap_total_[j] = 0.;
    }
};

void belief_propagation::compute_na_expect() noexcept {

    for (unsigned int j = 0; j < Q_; ++j) {
        na_expect_[j] = 0.0;
        nna_expect_[j] = 0.0;
    }
    for (unsigned int i = 0; i < N_; ++i) {
        for (unsigned int q = 0; q < Q_; ++q) {
            na_expect_[q] += real_psi_[i][q];
            nna_expect_[q] += adj_list_ptr_->at(i).size() * real_psi_[i][q];
        }
    }
}

double belief_propagation::compute_f_site() noexcept {
    double f_site = 0;
    double _diff_f_site = 0;

    for (unsigned int i = 0; i < N_; ++i) {
        double di = adj_list_ptr_->at(i).size();
        double _rescaling_param_ = -100000.;  // Quite ugly, 0 should be good...
        double _diff_rescaling_param_ = -100000.;

        for (unsigned int q = 0; q < Q_; ++q) {
            double a = 0.0;
            double _diff_a = 0.0;
            for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
                double b = 0;
                double _diff_b = 0;

                unsigned int nb = graph_neis_[i][l];
                for (unsigned int t = 0; t < Q_; ++t) {
                    if (deg_corr_flag_ == 0) {
                        b += std::pow(cab_[t][q], beta_) * mmap_[i][l][t];
                        _diff_b += std::pow(cab_[t][q], _diff_beta) * mmap_[i][l][t];
                    } else if (deg_corr_flag_ == 1) {
                        b += di * adj_list_ptr_->at(nb).size() * cab_[t][q] * mmap_[i][l][t];
                    } else if (deg_corr_flag_ == 2) {
                        double tmp = di * adj_list_ptr_->at(nb).size() * pab_[t][q];
                        b += tmp / (1.0 + tmp) * mmap_[i][l][t];  // sum over messages from l -> i
                    }
                }
                a += std::log(b);
                _diff_a += std::log(_diff_b);
            }
            if (deg_corr_flag_ == 0) {
                _real_psi_q_[q] = a + logeta_[q] - beta_ * h_[q] / N_;
                _diff_real_psi_q[q] = _diff_a + logeta_[q] - _diff_beta * h_[q] / N_;
            } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
                _real_psi_q_[q] = a + logeta_[q] - di * h_[q] / N_;
            }

            if (_real_psi_q_[q] > _rescaling_param_) {
                _rescaling_param_ = _real_psi_q_[q];
            }

            if (_diff_real_psi_q[q] > _diff_rescaling_param_) {
                _diff_rescaling_param_ = _diff_real_psi_q[q];
            }
        }

        double _norm_scaling_param_ = 0.;
        double _diff_norm_scaling_param_ = 0.;
        for (unsigned int q = 0; q < Q_; ++q) {
            _norm_scaling_param_ += std::exp(_real_psi_q_[q] - _rescaling_param_);
            _diff_norm_scaling_param_ += std::exp(_diff_real_psi_q[q] - _diff_rescaling_param_);
        }

        _norm_scaling_param_ = _rescaling_param_ + std::log(_norm_scaling_param_);
        _diff_norm_scaling_param_ = _diff_rescaling_param_ + std::log(_diff_norm_scaling_param_);

        f_site += _norm_scaling_param_;
        _diff_f_site += _diff_norm_scaling_param_;
    }
    f_site /= N_;
    return f_site;
}

double belief_propagation::compute_entropy_site() noexcept {
    double e_site = 0;
    for (unsigned int i = 0; i < N_; ++i) {
        double numerator = 0.;
        double denominator = 0.;
        for (unsigned int q = 0; q < Q_; ++q) {
            double a = 0.0;
            double numerator_a_1 = 0.;
            for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
                double b = 0;
                double numerator_b_1 = 0.;
                for (unsigned int t = 0; t < Q_; ++t) {
                    if (deg_corr_flag_ == 0) {
                        b += cab_[t][q] * mmap_[i][l][t];
                        numerator_b_1 += cab_[t][q] * mmap_[i][l][t];
                    } else {
                        std::clog << "Should raise NotImplementedError\n";
                    }
                }
                a += std::log(b);
                numerator_a_1 += std::log(numerator_b_1);
            }

            double numerator_a_2 = 0.;
            for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {

                double numerator_b_2_l = 0.;
                for (unsigned int t = 0; t < Q_; ++t) {
                    numerator_b_2_l += std::log(cab_[t][q]) * cab_[t][q] * mmap_[i][l][t];
                }

                double sum_logs_numerator_b_2_except_l = 0;
                for (unsigned int _l = 0; _l < adj_list_ptr_->at(i).size(); ++_l) {
                    double numerator_b_2_except_l = 0.;
                    for (unsigned int t = 0; t < Q_; ++t) {
                        if (_l != l) {
                            numerator_b_2_except_l += cab_[t][q] * mmap_[i][_l][t];
                        }
                    }
                    sum_logs_numerator_b_2_except_l += std::log(numerator_b_2_except_l);
                }
                numerator_a_2 += numerator_b_2_l * std::exp(sum_logs_numerator_b_2_except_l);
            }
            // TODO: think about it! What if the term is very large??
            if (deg_corr_flag_ == 0) {
                denominator += std::exp(a + logeta_[q] - h_[q] / N_);
                numerator += std::exp(numerator_a_1 + logeta_[q] - h_[q] / N_) * (-h_[q] / N_);  // this term is larger
                numerator += numerator_a_2 * std::exp(logeta_[q]) / std::exp(h_[q] / N_);  // this term is smaller
            }
        }
        e_site += numerator / denominator;
    }
    e_site /= N_;
    return e_site;
}

double belief_propagation::compute_f_edge() noexcept {
    double f_link = 0;
    double _diff_f_link = 0;

    for (unsigned int i = 0; i < N_; ++i) {
        double di = adj_list_ptr_->at(i).size();
        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
            double norm_L = 0;
            double _diff_norm_L = 0;

            unsigned int i2 = graph_neis_[i][l];
            unsigned int l2 = graph_neis_inv_[i][l];
            double dl = adj_list_ptr_->at(i2).size();

            for (unsigned int q1 = 0; q1 < Q_; ++q1) {
                for (unsigned int q2 = q1; q2 < Q_; ++q2) {
                    if (q1 == q2) {
                        if (deg_corr_flag_ == 0) {
                            norm_L += std::pow(cab_[q1][q2], beta_) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                            _diff_norm_L += std::pow(cab_[q1][q2], _diff_beta) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        } else if (deg_corr_flag_ == 1) {
                            norm_L += di * dl * cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            norm_L += tmp / (1.0 + tmp) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        }
                    } else {
                        if (deg_corr_flag_ == 0) {
                            norm_L += std::pow(cab_[q1][q2], beta_) *
                                      (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                            _diff_norm_L += std::pow(cab_[q1][q2], _diff_beta) *
                                            (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);

                        } else if (deg_corr_flag_ == 1) {
                            norm_L += di * dl * cab_[q1][q2] *
                                      (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            norm_L += tmp / (1.0 + tmp) *
                                      (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        }
                    }
                }
            }
            f_link += std::log(norm_L);
            _diff_f_link += std::log(_diff_norm_L);
        }
    }
    f_link /= 2. * N_;  // READ the paper! It counts for all edges, but in our calculation, we did it twice!
    return f_link;
}

double belief_propagation::compute_entropy_edge() noexcept {

    // TODO: check all degree-corrected terms
    double s_link = 0;
    for (unsigned int i = 0; i < N_; ++i) {
        double di = adj_list_ptr_->at(i).size();
        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
            unsigned int i2 = graph_neis_[i][l];
            unsigned int l2 = graph_neis_inv_[i][l];
            double dl = adj_list_ptr_->at(i2).size();

            double numerator = 0.;
            double denominator = 0.;

            for (unsigned int q1 = 0; q1 < Q_; ++q1) {
                for (unsigned int q2 = q1; q2 < Q_; ++q2) {
                    if (q1 == q2) {
                        if (deg_corr_flag_ == 0) {
                            denominator += cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                            numerator += cab_[q1][q2] * std::log(cab_[q1][q2]) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        } else if (deg_corr_flag_ == 1) {
                            denominator += di * dl * cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                            numerator += di * dl * cab_[q1][q2] * std::log(cab_[q1][q2]) *
                                         (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            denominator += tmp / (1.0 + tmp) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                            numerator +=
                                    tmp / (1.0 + tmp) * std::log(cab_[q1][q2]) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        }
                    } else {
                        if (deg_corr_flag_ == 0) {
                            denominator += cab_[q1][q2] *
                                           (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                            numerator += cab_[q1][q2] * std::log(cab_[q1][q2]) *
                                         (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        } else if (deg_corr_flag_ == 1) {
                            denominator += di * dl * cab_[q1][q2] *
                                           (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);

                            numerator += di * dl * cab_[q1][q2] * std::log(cab_[q1][q2]) *
                                         (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            denominator += tmp / (1.0 + tmp) *
                                           (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);

                            numerator += tmp / (1.0 + tmp) * std::log(cab_[q1][q2]) *
                                         (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        }
                    }
                }
            }
            s_link += numerator / denominator;
        }
    }
    s_link /= 2. * N_;
    return s_link;
}


double belief_propagation::compute_f_non_edge() noexcept {
    double log_f_non_edge_ = 0;
    double _diff_log_f_non_edge_ = 0;
    for (unsigned int i = 0; i < N_; ++i) {
        for (unsigned int l = 0; l < N_; ++l) {
            if (std::find(graph_neis_[i].begin(), graph_neis_[i].end(), l) == graph_neis_[i].end()) {
                // l is not a neighbor to i
                double f_non_edge_ = 0;
                double _diff_f_non_edge_ = 0;

                for (unsigned int q1 = 0; q1 < Q_; ++q1) {
                    for (unsigned int q2 = 0; q2 < Q_; ++q2) {
                        if (deg_corr_flag_ == 0) {
                            f_non_edge_ +=
                                    std::pow((1 - cab_[q1][q2] / N_), beta_) * real_psi_[i][q1] * real_psi_[l][q2];
                            _diff_f_non_edge_ +=
                                    (1 - std::pow(cab_[q1][q2], _diff_beta) / N_) * real_psi_[i][q1] * real_psi_[l][q2];
                        } else {
                            // should raise error or something (e.g. NotImplementedError)
                        }
                    }
                }
                if (f_non_edge_ != 0.) {
                    log_f_non_edge_ += std::log(f_non_edge_);
                    _diff_log_f_non_edge_ += std::log(_diff_f_non_edge_);
                } else {
                    // this could happen, esp. when epsilon is close to 0, where the network structure is strong
                    // then, real_psi_[i] has some zero components.
                }
            }
        }
    }
    log_f_non_edge_ /= 2. * N_;
    return log_f_non_edge_;
}

double belief_propagation::compute_entropy_non_edge() noexcept {
    double f_non_edge = 0;
    for (unsigned int i = 0; i < N_; ++i) {
        for (unsigned int l = 0; l < N_; ++l) {
            if (std::find(graph_neis_[i].begin(), graph_neis_[i].end(), l) == graph_neis_[i].end()) {
                double numerator = 0.;
                double denominator = 0.;
                // l is not a neighbor to i
                for (unsigned int q1 = 0; q1 < Q_; ++q1) {
                    for (unsigned int q2 = 0; q2 < Q_; ++q2) {
                        if (deg_corr_flag_ == 0) {
                            denominator += (1 - cab_[q1][q2] / N_) * real_psi_[i][q1] * real_psi_[l][q2];
                            numerator +=
                                    (cab_[q1][q2] / N_) * std::log(cab_[q1][q2]) * real_psi_[i][q1] * real_psi_[l][q2];
                        } else {
                            // should raise error or something (e.g. NotImplementedError)
                        }
                    }
                }
                if (numerator * denominator != 0) {
                    f_non_edge += numerator / denominator;
                } else {
                    // this could happen, esp. when epsilon is close to 0, where the network structure is strong
                    // then, real_psi_[i] has some zero components.
                }
            }
        }
    }
    f_non_edge /= 2. * N_;
    return f_non_edge;
}


double belief_propagation::compute_free_energy() noexcept {
    double f_site = -compute_f_site();  // ~eq.29
    double f_link = compute_f_edge();  // ~eq. 30
    double f_non_edge = compute_f_non_edge();  // ~eq. 31
    double totalf = (f_site + f_link + f_non_edge);
    return totalf;
}

double belief_propagation::compute_entropy() noexcept {
    double e_site = -compute_entropy_site();
    double e_link = +compute_entropy_edge();
    double e_non_edge = -compute_entropy_non_edge();
    double totale = (e_site + e_link + e_non_edge);
    return totale;
}

double belief_propagation::compute_e_nishimori() noexcept {
    double e_nishimori = 0.;

    for (unsigned int q1 = 0; q1 < Q_; ++q1) {
        e_nishimori -= eta_[q1] * logeta_[q1];
        e_nishimori -= eta_[q1] * eta_[q1] * cab_[q1][q1] * logcab_[q1][q1] / 2.;
        for (unsigned int q2 = q1 + 1; q2 < Q_; ++q2) {
            e_nishimori -= eta_[q1] * eta_[q2] * cab_[q1][q2] * logcab_[q1][q2];
        }
    }
    e_nishimori += -compute_f_non_edge();
    return e_nishimori;
}


double belief_propagation::compute_overlap() noexcept {
    uint_mat_t perms;
    perms.clear();
    uint_vec_t myperm;
    myperm.resize(Q_);
    for (unsigned int i = 0; i < Q_; ++i) {
        myperm[i] = i;
    }
    std::sort(myperm.begin(), myperm.end());
    if (Q_ > 8) {
        perms.push_back(myperm);
    } else {
        do {
            perms.push_back(myperm);
        } while (std::next_permutation(myperm.begin(), myperm.end()));
    }
    double max_ov = -1.0;
    double connect = 0.0;
    // TODO: check if it's valid
    compute_na_expect();
    for (unsigned int q = 0; q < Q_; ++q) {
        connect += pow((double) (na_expect_[q]) / N_, 2);
    }
    for (auto pe: perms) {
        double ov = 0.0;
        for (unsigned int i = 0; i < N_; ++i) {
            ov += (real_psi_[i][pe[conf_true_[i]]]);
        }
        ov /= N_;
#ifdef OVL_NORM
        ov=(ov-connect)/(1-connect);
#endif
        if (ov > max_ov) max_ov = ov;
    }

    return max_ov;
}

double belief_propagation::bp_iter_update_psi_large_degree(unsigned int i,
                                                           double dumping_rate) noexcept {
    double di = adj_list_ptr_->at(i).size();
    std::clog << "bp_iter_update_psi_large_degree used!" << "\n";
    double a, b;
    _real_psi_total_ = 0.0;

    clean_mmap_total_at_node_i_(i);

    double maxpom_psi = -100000000.0;
    double xxx = -100000000.0;
    for (unsigned int j = 0; j < adj_list_ptr_->at(i).size(); ++j) {
        maxpom_psii_iter_[j] = xxx;
    }
    for (unsigned int q = 0; q < Q_; ++q) {
        a = 0.0;//log value
        // sum of all graphbors of i
        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
            b = 0.0;
            unsigned int neighbor = graph_neis_[i][l];
            for (unsigned int t = 0; t < Q_; ++t) {
                if (deg_corr_flag_ == 0) {
                    b += cab_[t][q] * mmap_[i][l][t];
                } else if (deg_corr_flag_ == 1) {
                    b += di * adj_list_ptr_->at(neighbor).size() * cab_[t][q] * mmap_[i][l][t];
                    //sum over messages from l -> i
                } else if (deg_corr_flag_ == 2) {
                    double tmp = di * adj_list_ptr_->at(neighbor).size() * pab_[t][q];
                    b += tmp / (1.0 + tmp) * mmap_[i][l][t];//sum over messages from l -> i
                }
            }
            double tmp = log(b);
            a += tmp;
            field_iter_[l] = tmp;
        }

        if (deg_corr_flag_ == 0) {
            _real_psi_q_[q] = a + logeta_[q] - h_[q] / N_;
        } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
            _real_psi_q_[q] = a + logeta_[q] - 1.0 * di * h_[q] / N_;
        }

        if (_real_psi_q_[q] > maxpom_psi) {
            maxpom_psi = _real_psi_q_[q];
        }
        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
            _mmap_q_nb_[q][l] = _real_psi_q_[q] - field_iter_[l];
            if (_mmap_q_nb_[q][l] > maxpom_psii_iter_[l]) maxpom_psii_iter_[l] = _mmap_q_nb_[q][l];
        }
    }
    for (unsigned int q = 0; q < Q_; ++q) {
        _real_psi_total_ += exp(_real_psi_q_[q] - maxpom_psi);
        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
            _mmap_total_[l] += exp(_mmap_q_nb_[q][l] - maxpom_psii_iter_[l]);
        }
    }

    update_h(i, -1);

    double mymaxdiff = -100.0;
    // normalization
    for (unsigned int q = 0; q < Q_; ++q) {
        real_psi_[i][q] = exp(_real_psi_q_[q] - maxpom_psi) / _real_psi_total_;
        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
            unsigned int i2 = graph_neis_[i][l];
            unsigned int l2 = graph_neis_inv_[i][l];
            double thisvalue = exp(_mmap_q_nb_[q][l] - maxpom_psii_iter_[l]) / _mmap_total_[l];
            double mydiff = fabs(mmap_[i2][l2][q] - thisvalue);
            if (mydiff > mymaxdiff) mymaxdiff = mydiff;
            mmap_[i2][l2][q] = (dumping_rate) * thisvalue + (1 - dumping_rate) * mmap_[i2][l2][q];

        }
    }

    update_h(i, +1);
    update_exph_with_h();
    return mymaxdiff;
}

void belief_propagation::compute_cab_expect() noexcept {
    for (unsigned int q1 = 0; q1 < Q_; ++q1) {
        for (unsigned int q2 = 0; q2 < Q_; ++q2) {
            cab_expect_[q1][q2] = 0.;
        }
    }

    for (unsigned int i = 0; i < N_; ++i) {
        double di = adj_list_ptr_->at(i).size();
        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); l++) {
            unsigned int i2 = graph_neis_[i][l];
            unsigned int l2 = graph_neis_inv_[i][l];
            double dl = adj_list_ptr_->at(i2).size();

            double norm_L = 0.0;
            for (unsigned int q1 = 0; q1 < Q_; ++q1) {
                for (unsigned int q2 = q1; q2 < Q_; ++q2) {
                    if (q1 == q2) {
                        if (deg_corr_flag_ == 0) {
                            norm_L += cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        } else if (deg_corr_flag_ == 1) {
                            norm_L += di * dl * cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            norm_L += tmp / (1.0 + tmp) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        }
                    } else {
                        if (deg_corr_flag_ == 0) {
                            norm_L += cab_[q1][q2] *
                                      (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        } else if (deg_corr_flag_ == 1) {
                            norm_L += di * dl * cab_[q1][q2] *
                                      (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            norm_L += tmp / (1.0 + tmp) *
                                      (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        }
                    }
                }
            }
            for (unsigned int q1 = 0; q1 < Q_; ++q1) {
                for (unsigned int q2 = q1; q2 < Q_; ++q2) {
                    if (q1 == q2) {
                        if (deg_corr_flag_ == 0) {
                            cab_expect_[q1][q2] += 0.5 * cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2]) / norm_L;
                        } else if (deg_corr_flag_ == 1) {
                            cab_expect_[q1][q2] += 0.5 * di * dl * cab_[q1][q2] *
                                                   (mmap_[i][l][q1] * mmap_[i2][l2][q2]) / norm_L;
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            cab_expect_[q1][q2] += 0.5 * tmp / (1.0 + tmp) *
                                                   (mmap_[i][l][q1] * mmap_[i2][l2][q2]) / norm_L;
                        }
                    } else {
                        if (deg_corr_flag_ == 0) {
                            cab_expect_[q1][q2] += 0.5 * cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2] +
                                                                         mmap_[i][l][q2] * mmap_[i2][l2][q1]) / norm_L;
                        } else if (deg_corr_flag_ == 1) {
                            cab_expect_[q1][q2] += 0.5 * di * dl * cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2] +
                                                                                   mmap_[i][l][q2] *
                                                                                   mmap_[i2][l2][q1]) / norm_L;
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            cab_expect_[q1][q2] += 0.5 * tmp / (1.0 + tmp) *
                                                   (mmap_[i][l][q1] * mmap_[i2][l2][q2] +
                                                    mmap_[i][l][q2] * mmap_[i2][l2][q1])
                                                   / norm_L;
                        }
                        cab_expect_[q2][q1] = cab_expect_[q1][q2];  // TODO: check if this is really necessary
                    }
                }
            }
        }
    }
    for (unsigned int q1 = 0; q1 < Q_; ++q1) {
        for (unsigned int q2 = q1; q2 < Q_; ++q2) {
//            cab_expect_[q1][q2] /= (N_ * na_expect_[q1] * na_expect_[q2]);
            if ((na_expect_[q1] > EPS) && (na_expect_[q2] > EPS)) {
                // TODO: check what this is for???
                if (q1 != q2) {
                    if (deg_corr_flag_ == 0) {
                        cab_expect_[q1][q2] *= N_ / (na_expect_[q1] * na_expect_[q2]);
                    } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
                        cab_expect_[q1][q2] *= N_ / (nna_expect_[q1] * nna_expect_[q2]);
                    }
                    cab_expect_[q2][q1] = cab_expect_[q1][q2];
                } else {
                    if (deg_corr_flag_ == 0) {
                        cab_expect_[q1][q2] *= 2. * N_ / (na_expect_[q1] * na_expect_[q2]);
                    } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
                        cab_expect_[q1][q2] *= 2. * N_ / (nna_expect_[q1] * nna_expect_[q2]);
                    }
                }
            }
        }
    }
}

void belief_propagation::sum_all_messages_to_i(unsigned int i) noexcept {
    double di = adj_list_ptr_->at(i).size();
    double a, b;
    _real_psi_total_ = 0.0;

    for (unsigned int q = 0; q < Q_; ++q) {
        a = 1.0;
        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {

            b = 0.0;
            unsigned int neighbor = graph_neis_[i][l];
            for (unsigned int t = 0; t < Q_; ++t) {  //they are all to sum over messages from l -> i
                if (deg_corr_flag_ == 0) {
                    b += std::pow(cab_[t][q], beta_) * mmap_[i][l][t];
                } else if (deg_corr_flag_ == 1) {
                    b += di * adj_list_ptr_->at(neighbor).size() * cab_[t][q] * mmap_[i][l][t];
                    //sum over messages from l -> i
                } else if (deg_corr_flag_ == 2) {
                    double tmp = di * adj_list_ptr_->at(neighbor).size() * pab_[t][q];
                    b += tmp / (1.0 + tmp) * mmap_[i][l][t];//sum over messages from l -> i
                }
            }
            if (b == 0.) {
                std::clog << "sanity check 1: this should not happen.\n";
                continue;
            }
            a *= b;
            field_iter_[l] = b;
        }
        if (deg_corr_flag_ == 0) {
            _real_psi_q_[q] = a * eta_[q] * exph_[q];
        } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
            _real_psi_q_[q] = a * eta_[q] * exp(-1.0 * di * h_[q] / N_);
        }

        _real_psi_total_ += _real_psi_q_[q];

        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
            if (field_iter_[l] < EPS) {  // to cure the problem that field_iter_[l] may be very small
                double tmprob = 1.0;
                for (unsigned int lx = 0; lx < adj_list_ptr_->at(i).size(); ++lx) {
                    if (lx == l) {
                        continue;
                    }
                    if (field_iter_[lx] != 0) {
                        tmprob *= field_iter_[lx];
                    } else {
                        std::clog << "sanity check 2: this should not happen.\n";
                    }

                }
                _mmap_q_nb_[q][l] = tmprob;
            } else {
                _mmap_q_nb_[q][l] = _real_psi_q_[q] / field_iter_[l];
            }
            _mmap_total_[l] += _mmap_q_nb_[q][l];
        }
    }
}

double belief_propagation::norm_m_at_i(unsigned int i, double dumping_rate) noexcept {
    // normalization
    double mymaxdiff = -100.0;
    for (unsigned int q = 0; q < Q_; ++q) {
        real_psi_[i][q] = _real_psi_q_[q] / _real_psi_total_;
        for (unsigned int l = 0; l < adj_list_ptr_->at(i).size(); ++l) {
            unsigned int i2 = graph_neis_[i][l];
            unsigned int l2 = graph_neis_inv_[i][l];
            double mydiff = fabs(mmap_[i2][l2][q] - _mmap_q_nb_[q][l] / _mmap_total_[l]);

            if (mydiff > mymaxdiff) {
                mymaxdiff = mydiff;
            }

            mmap_[i2][l2][q] = (dumping_rate) * _mmap_q_nb_[q][l] / _mmap_total_[l] +
                               (1.0 - dumping_rate) * mmap_[i2][l2][q];//update psi_{i\to j}
        }
    }

    return mymaxdiff;
}

double belief_propagation::get_marginal_entropy() noexcept {
    // TODO
    return 0.;
}


double bp_basic::bp_iter_update_psi(unsigned int i, double dumping_rate) noexcept {
/*
 * This function implements Step-7 to Step-11 of Algm III.B of Aurelien Decelle's PRE 84, 066106 (2011) paper
 */

    clean_mmap_total_at_node_i_(i);
    sum_all_messages_to_i(i);

    /// Update h_ (Part I. Substract the old psi_j)
    update_h(i, -1);

    double mymaxdiff = norm_m_at_i(i, dumping_rate);
    /// Update h_ (Part II. Add the new psi_j)

    update_h(i, +1);

    update_exph_with_h();

    return mymaxdiff;
}

double bp_conditional::bp_iter_update_psi(unsigned int i, double dumping_rate) noexcept {
/*
 * This function implements Step-7 to Step-11 of Algm III.B of Aurelien Decelle's PRE 84, 066106 (2011) paper
 */
    if (conf_planted_[i] == -1) {  // random
        clean_mmap_total_at_node_i_(i);
        sum_all_messages_to_i(i);
        // Update h_ (Part I. Substract the old psi_j)
        update_h(i, -1);
        double mymaxdiff = norm_m_at_i(i, dumping_rate);
        // Update h_ (Part II. Add the new psi_j)
        update_h(i, +1);
        update_exph_with_h();

        return mymaxdiff;
    } else {
        /*
         * In bp_conditional mode, if the belief at a certain node is not assigned random,
         * then, we will believe that it is the true partition, which
         * emits constant messages to adjacent nodes.
         *
         * That is, we don't update messages on this node.
         */
        return 0;
    }

}

