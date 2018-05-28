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

        double fnew = compute_free_energy(false);

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

//    std::clog << "(marg_ent_, bethe_ent_) = " << marg_ent_[1] << ", " << bethe_ent_[1] << "\n";
//    output_vec<double_vec_t>(bethe_ent_, std::cout);

//    double f = compute_free_energy();
    if (output_entropy_) {
        double e = compute_entropy(true);
        output_vec<double_vec_t>(bethe_ent_, std::cout);
        std::clog << "Bethe entropy | overlap | niter \n";
        std::cout << e << " " << compute_overlap() << " " << niter << " \n";
    } else if (output_free_energy_) {
        double f = compute_free_energy(true);
        output_vec<double_vec_t>(free_ene_, std::cout);
        std::clog << "Bethe free energy | overlap | niter \n";
        std::cout << f << " " << compute_overlap() << " " << niter << " \n";
    } else if (output_weighted_free_energy_) {
        compute_marg_entropy();
        double f = compute_free_energy(true);
        double_vec_t score;
        score.resize(N_, 0.);
        for (unsigned int idx = 0; idx < N_; ++idx) {
            score[idx] = marg_ent_[idx] * free_ene_[idx];
        }
        output_vec<double_vec_t>(score, std::cout);
        std::clog << "Weighted Bethe free energy | overlap | niter \n";
        std::cout << f << " " << compute_overlap() << " " << niter << " \n";
    } else if (output_weighted_entropy_) {
        compute_marg_entropy();
        double e = compute_entropy(true);
        double_vec_t score;
        score.resize(N_, 0.);
        for (unsigned int idx = 0; idx < N_; ++idx) {
            score[idx] = marg_ent_[idx] * bethe_ent_[idx];
        }
        output_vec<double_vec_t>(score, std::cout);
        std::clog << "Weighted Bethe entropy | overlap | niter \n";
        std::cout << e << " " << compute_overlap() << " " << niter << " \n";
    } else {  // normal mode
        double e = compute_entropy(true);
        double f = compute_free_energy(true);
//        output_vec<double_vec_t>(free_ene_, std::cout);
        std::clog << "Bethe entropy | Bethe free energy | overlap | niter \n";
        std::cout << e << " " << f << " " << compute_overlap() << " " << niter << " \n";
    }

//    output_vec<double_vec_t>(free_ene_, std::cout);

//    output_vec(marg_ent_, std::clog);
//    std::clog << "-----\n";

//    output_vec(bethe_ent_, std::clog);
//    std::clog << "free_ene_: \n";
//    output_vec(free_ene_, std::clog);
//    std::clog << "bethe_ent_: \n";
//    output_vec(bethe_ent_, std::clog);
//    std::clog << "--- --- --- --- \n";
//    std::clog << "--- --- --- --- \n";
//    std::clog << "free_ene_site_: \n";
//    output_vec(free_ene_site_, std::clog);
//    std::clog << "free_ene_edge_: \n";
//    output_vec(free_ene_edge_, std::clog);
//    std::clog << "bethe_ent_site_: \n";
//    output_vec(bethe_ent_site_, std::clog);
//    std::clog << "bethe_ent_edge_: \n";
//    output_vec(bethe_ent_edge_, std::clog);


    if (output_marginals_) {
        // This outputs the vertex marginals when BP is converged.
        // The marginal distribution can be used to compute the mean-field entropy
        // AND
        // the entropy of the average conditional distribution, i.e. margEntropy, or H(v).

        output_mat<double_mat_t>(real_psi_, std::cout);
        // MI(v) = H(v) − H(v | G \ v)
        //       = entropy of the average conditional distribution (margEntropy) - average of the conditional entropy (condEntropy);
        for (unsigned int v = 0; v < N_ ; ++v) {
            std::clog << "n_idx: " << v << "; margEnt H(v) is " << entropy(real_psi_[v], Q_) << "; n_nbs: " << adj_list_ptr_->at(v).size() << "; nb = ";
            output_vec<>(adj_list_ptr_->at(v), std::clog);
        }
    }

    if (output_conditional_pairwise_entropies_) {
        for (unsigned int n = 0; n < N_; ++n) {  // TODO: improve this block!
            auto nb = graph_neis_[n].size();
            for (unsigned int m = 0; m < N_; ++m) {
                std::cout << n << " " << m << " " << entropy(real_psi_[n], 2) << " \n";
            }

            for (unsigned int nb_ = 0; nb_ < nb; ++nb_) {
                std::cout << n << " " << graph_neis_[n][nb_] << " " << entropy(mmap_[n][nb_], 2) << " \n";
            }
        }
        double e = compute_entropy(true);
        for (unsigned int n = 0; n < N_; ++n) {
//            std::cout << entropy(real_psi_[n], 2) << " ";
            std::cout << bethe_ent_[n] << " ";
        }
        std::cout << "\n";
    }
}

int belief_propagation::converge(float bp_err,
                                 unsigned int max_iter_time,
                                 float dumping_rate,
                                 std::mt19937 &engine) noexcept {
    init_h();
    for (int iter_time = 0; iter_time < max_iter_time; ++iter_time) {
        maxdiffm_ = -1;
        for (unsigned int iter_inter_time = 0; iter_inter_time < N_; ++iter_inter_time) {
            double diffm = -1;
            while (diffm == -1) {
                auto i = unsigned(int(random_real(engine) * N_));   // TODO: unify the expression
//                std::clog << "now updating messages to node " << i << "... \n";
                if (adj_list_ptr_->at(i).size() >= LARGE_DEGREE) {
                    diffm = bp_iter_update_psi_large_degree(i, dumping_rate);
                } else {
                    diffm = bp_iter_update_psi(i, dumping_rate);
                }
            }
            if (diffm > maxdiffm_) {
                maxdiffm_ = diffm;
            }
        }

        if (maxdiffm_ < bp_err) {
            std::clog << "End-of-BP: maxdiffm (" << maxdiffm_ << ") < bp_err (" << bp_err << "); iter_times = " << iter_time + 1 << "\n";

            // TODO: put them as __private__ variable
            // int bp_last_conv_time = iter_time;
            // double bp_last_diff = maxdiffm;
            return iter_time;
        }
    }
    std::clog << "End-of-BP: to-few-BP-time; the _maxdiffm is " << maxdiffm_ << "\n";
    return -1;
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
            auto n = adj_list_ptr_->at(i).size();
            for (unsigned int idxij = 0; idxij < n; ++idxij) {
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
            auto n = adj_list_ptr_->at(i).size();
            for (unsigned int idxij = 0; idxij < n; ++idxij) {
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
                auto n = adj_list_ptr_->at(i).size();
                for (unsigned int idxij = 0; idxij < n; ++idxij) {
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
            auto n = adj_list_ptr_->at(i).size();
            for (unsigned int idxij = 0; idxij < n; ++idxij) {
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

void belief_propagation::init_special_needs(bool if_output_marginals,
                                            bool if_output_conditional_pairwise_entropies,
                                            bool if_output_free_energy,
                                            bool if_output_weighted_free_energy,
                                            bool if_output_entropy,
                                            bool if_output_weighted_entropy) noexcept {
    output_marginals_ = if_output_marginals;
    output_conditional_pairwise_entropies_ = if_output_conditional_pairwise_entropies;
    output_free_energy_ = if_output_free_energy;
    output_weighted_free_energy_ = if_output_weighted_free_energy;
    output_entropy_ = if_output_entropy;
    output_weighted_entropy_ = if_output_weighted_entropy;
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

    // conditional entropies
    bethe_ent_.resize(N_, 0.);
    marg_ent_.resize(N_, 0.);
    free_ene_.resize(N_, 0.);
    free_ene_site_.resize(N_, 0.);
    free_ene_edge_.resize(N_, 0.);
    bethe_ent_site_.resize(N_, 0.);
    bethe_ent_edge_.resize(N_, 0.);

    /// temporary variables to ensure normalization
    _real_psi_q_.resize(Q_);
    _diff_real_psi_q.resize(Q_);

    /// Done
    exph_.resize(Q_, 0);
    mmap_.resize(N_);
    graph_neis_inv_.resize(N_);
    graph_neis_.resize(N_);
    cond_pair_ent_.resize(N_);

    unsigned int vertex_j;
    unsigned int idxji;
    for (unsigned int i = 0; i < N_; ++i) {
        mmap_[i].resize(adj_list_ptr_->at(i).size());

        graph_neis_inv_[i].resize(adj_list_ptr_->at(i).size());
        graph_neis_[i].resize(adj_list_ptr_->at(i).size());
        cond_pair_ent_[i].resize(adj_list_ptr_->at(i).size());

        auto nb_of_i(adj_list_ptr_->at(i).begin());
        auto n = adj_list_ptr_->at(i).size();
        for (unsigned int idxij = 0; idxij < n; ++idxij) {

            mmap_[i][idxij].resize(Q_);
            vertex_j = *nb_of_i;

            auto nb_of_j(adj_list_ptr_->at(vertex_j).begin());
            idxji = unsigned(int(std::distance(nb_of_j, adj_list_ptr_->at(vertex_j).find(i))));
            graph_neis_inv_[i][idxij] = idxji;
            graph_neis_[i][idxij] = vertex_j;
            cond_pair_ent_[i][idxij] = 0.;
            advance(nb_of_i, 1);
        }
    }
    maxpom_psii_iter_.resize(graph_max_degree);
    field_iter_.resize(graph_max_degree);
    _mmap_total_.resize(graph_max_degree);

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

void belief_propagation::set_beta(double beta) noexcept {
    beta_ = beta;
}


void belief_propagation::clean_mmap_total_at_node_i_(unsigned int node_idx) noexcept {
    auto n = adj_list_ptr_->at(node_idx).size();
    for (unsigned int j = 0; j < n; ++j) {
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

double belief_propagation::compute_free_energy_site(bool by_site) noexcept {
    double f_site = 0;
//    double _diff_f_site = 0;

    for (unsigned int i = 0; i < N_; ++i) {
        const double di = adj_list_ptr_->at(i).size();
        double _rescaling_param_ = -1.;  // Quite ugly, 0 should be good...
//        double _diff_rescaling_param_ = -1.;

        for (unsigned int q = 0; q < Q_; ++q) {
            a_ = 0.;
//            double _diff_a = 0.0;
            auto nb_of_i = adj_list_ptr_->at(i).size();
            for (unsigned int l = 0; l < nb_of_i; ++l) {
                b_ = 0.;
//                double _diff_b = 0;

                unsigned int nb = graph_neis_[i][l];
                for (unsigned int t = 0; t < Q_; ++t) {
                    if (deg_corr_flag_ == 0) {
                        b_ += std::pow(cab_[t][q], beta_) * mmap_[i][l][t];
//                        _diff_b += std::pow(cab_[t][q], _diff_beta) * mmap_[i][l][t];
                    } else if (deg_corr_flag_ == 1) {
                        b_ += di * adj_list_ptr_->at(nb).size() * cab_[t][q] * mmap_[i][l][t];
                    } else if (deg_corr_flag_ == 2) {
                        double tmp = di * adj_list_ptr_->at(nb).size() * pab_[t][q];
                        b_ += tmp / (1.0 + tmp) * mmap_[i][l][t];  // sum over messages from l -> i
                    }
                }
                a_ += std::log(b_);
//                _diff_a += std::log(_diff_b);
            }
            if (deg_corr_flag_ == 0) {

                _real_psi_q_[q] = a_ + logeta_[q] - beta_ * h_[q] / N_;
//                _diff_real_psi_q[q] = _diff_a + logeta_[q] - _diff_beta * h_[q] / N_;
            } else if (deg_corr_flag_ == 1 || deg_corr_flag_ == 2) {
                _real_psi_q_[q] = a_ + logeta_[q] - di * h_[q] / N_;
            }

            if (_real_psi_q_[q] > _rescaling_param_) {
                _rescaling_param_ = _real_psi_q_[q];
            }

//            if (_diff_real_psi_q[q] > _diff_rescaling_param_) {
//                _diff_rescaling_param_ = _diff_real_psi_q[q];
//            }
        }

        double _norm_scaling_param_ = 0.;
//        double _diff_norm_scaling_param_ = 0.;
        for (unsigned int q = 0; q < Q_; ++q) {
            _norm_scaling_param_ += std::exp(_real_psi_q_[q] - _rescaling_param_);
//            _diff_norm_scaling_param_ += std::exp(_diff_real_psi_q[q] - _diff_rescaling_param_);
        }

        _norm_scaling_param_ = _rescaling_param_ + std::log(_norm_scaling_param_);
//        _diff_norm_scaling_param_ = _diff_rescaling_param_ + std::log(_diff_norm_scaling_param_);

        f_site += _norm_scaling_param_ / N_;
//        _diff_f_site += _diff_norm_scaling_param_ / N_;
//        std::clog << (_diff_norm_scaling_param_ - _norm_scaling_param_) / (_diff_beta - beta_) / N_ << ",";
//        std::clog << "(Correct) Z_" << i << ": " << std::exp(_norm_scaling_param_) << "\n";
        if (by_site) {
            free_ene_[i] += -_norm_scaling_param_ / N_;
            free_ene_site_[i] += -_norm_scaling_param_ / N_;
        }


    }

//    std::clog << "the numerical entropy (site) is " << (_diff_f_site - f_site) / (_diff_beta - beta_) << "\n";

    return -f_site;
}

double belief_propagation::compute_entropy_site(bool by_site) noexcept {
    double e_site = 0;
    double_vec_t numerator_a_2_edge_;

    for (unsigned int i = 0; i < N_; ++i) {
        numerator_ = 0.;
        denominator_ = 0.;
        const auto n = adj_list_ptr_->at(i).size();
        numerator_edge_.clear();
        numerator_edge_.resize(n, 0.);
        for (unsigned int q = 0; q < Q_; ++q) {
            a_ = 0.0;
            double numerator_a_1 = 0.;

            for (unsigned int l = 0; l < n; ++l) {
                b_ = 0.;
                double numerator_b_1 = 0.;
                for (unsigned int t = 0; t < Q_; ++t) {
                    if (deg_corr_flag_ == 0) {
                        b_ += cab_[t][q] * mmap_[i][l][t];
                        numerator_b_1 += cab_[t][q] * mmap_[i][l][t];
                    } else {
                        std::clog << "Should raise NotImplementedError\n";
                    }
                }
                a_ += std::log(b_);
                numerator_a_1 += std::log(numerator_b_1);
            }

            double numerator_a_2 = 0.;

            numerator_a_2_edge_.clear();
            numerator_a_2_edge_.resize(n, 0.);
//            std::clog << "before the for-loop, n is " << n << "\n";
            for (unsigned int l = 0; l < n; ++l) {
//                std::clog << "the differential term (with the new log term) -- l is " << l << "\n";
                double numerator_b_2_l = 0.;
                for (unsigned int t = 0; t < Q_; ++t) {
                    numerator_b_2_l += std::log(cab_[t][q]) * cab_[t][q] * mmap_[i][l][t];
                }

                double sum_logs_numerator_b_2_except_l = 0;
                for (unsigned int _l = 0; _l < n; ++_l) {
                    if (_l != l) {
//                        std::clog << "l is " << l << "; non_l_member: " << _l << ",";
                        double numerator_b_2_except_l = 0.;
                        for (unsigned int t = 0; t < Q_; ++t) {
                            numerator_b_2_except_l += cab_[t][q] * mmap_[i][_l][t];
                        }
                        sum_logs_numerator_b_2_except_l += std::log(numerator_b_2_except_l);
                    }
                }
//                std::clog << "\n";
                numerator_a_2 += numerator_b_2_l * std::exp(sum_logs_numerator_b_2_except_l);
                numerator_a_2_edge_[l] += numerator_b_2_l * std::exp(sum_logs_numerator_b_2_except_l);
            }
            // TODO: think about it! What if the term is very large??
            if (deg_corr_flag_ == 0) {

                denominator_ += std::exp(a_ + logeta_[q] - h_[q] / N_);
                numerator_ += std::exp(numerator_a_1 + logeta_[q] - h_[q] / N_) * (- h_[q] / N_);  // this term is larger
                numerator_ += numerator_a_2 * std::exp(logeta_[q] - h_[q] / N_);  // this term is smaller
                for (unsigned int l = 0; l < n; ++l) {
                    numerator_edge_[l] += numerator_a_2_edge_[l] * std::exp(logeta_[q] - h_[q] / N_);
                }

            }
        }
//        std::clog << "Anaytical Z_" << i << ": " << denominator_ << "\n";
//        std::clog << numerator_ << " , " << denominator_ << "\n";
        e_site += numerator_ / denominator_ / N_;
        if (by_site) {
            bethe_ent_[i] += numerator_ / denominator_ / N_;
            bethe_ent_site_[i] += numerator_ / denominator_ / N_;

            for (unsigned int l = 0; l < n; ++l) {
                cond_pair_ent_[i][l] += numerator_edge_[l] / denominator_ / N_;
            }
        }
//        std::clog << "node entropy [" << i << "] = " << numerator_ / denominator_ / N_ << "\n";
//        std::clog << numerator_ / denominator_ / N_ << ",";
    }
//    output_mat(cab_, std::clog);
    return e_site;
}

double belief_propagation::compute_free_energy_edge(bool by_site) noexcept {
    double f_link = 0;
//    double _diff_f_link = 0;

    for (unsigned int i = 0; i < N_; ++i) {
        double di = adj_list_ptr_->at(i).size();
        auto n = adj_list_ptr_->at(i).size();
        for (unsigned int l = 0; l < n; ++l) {
            double norm_L = 0;
//            double _diff_norm_L = 0;

            unsigned int i2 = graph_neis_[i][l];
            unsigned int l2 = graph_neis_inv_[i][l];
            double dl = adj_list_ptr_->at(i2).size();

            for (unsigned int q1 = 0; q1 < Q_; ++q1) {
                for (unsigned int q2 = q1; q2 < Q_; ++q2) {
                    if (q1 == q2) {
                        if (deg_corr_flag_ == 0) {
                            norm_L += std::pow(cab_[q1][q2], beta_) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
//                            _diff_norm_L += std::pow(cab_[q1][q2], _diff_beta) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
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
//                            _diff_norm_L += std::pow(cab_[q1][q2], _diff_beta) *
//                                            (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);

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
            // READ the paper! It counts for all edges, but in our calculation, we did it twice!
            f_link += std::log(norm_L) / 2. / N_;
//            _diff_f_link += std::log(_diff_norm_L) / 2. / N_;
            if (by_site) {
                free_ene_[i] += std::log(norm_L) / 2. / N_;
                free_ene_edge_[i] += std::log(norm_L) / 2. / N_;
            }
        }
    }
//    std::clog << "the numerical entropy (edge) is " << (_diff_f_link-f_link) / (_diff_beta - beta_) << "\n";

    return f_link;
}

double belief_propagation::compute_entropy_edge(bool by_site) noexcept {

    // TODO: check all degree-corrected terms
    double s_link = 0;
    for (unsigned int i = 0; i < N_; ++i) {
        double di = adj_list_ptr_->at(i).size();
        auto n = adj_list_ptr_->at(i).size();
        for (unsigned int l = 0; l < n; ++l) {
            unsigned int i2 = graph_neis_[i][l];
            unsigned int l2 = graph_neis_inv_[i][l];
            double dl = adj_list_ptr_->at(i2).size();

            numerator_ = 0.;
            denominator_ = 0.;

            for (unsigned int q1 = 0; q1 < Q_; ++q1) {
                for (unsigned int q2 = q1; q2 < Q_; ++q2) {
                    if (q1 == q2) {
                        if (deg_corr_flag_ == 0) {
                            denominator_ += cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                            numerator_ += cab_[q1][q2] * std::log(cab_[q1][q2]) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        } else if (deg_corr_flag_ == 1) {
                            denominator_ += di * dl * cab_[q1][q2] * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                            numerator_ += di * dl * cab_[q1][q2] * std::log(cab_[q1][q2]) *
                                         (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            denominator_ += tmp / (1.0 + tmp) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                            numerator_ +=
                                    tmp / (1.0 + tmp) * std::log(cab_[q1][q2]) * (mmap_[i][l][q1] * mmap_[i2][l2][q2]);
                        }
                    } else {
                        if (deg_corr_flag_ == 0) {
                            denominator_ += cab_[q1][q2] *
                                           (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                            numerator_ += cab_[q1][q2] * std::log(cab_[q1][q2]) *
                                         (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        } else if (deg_corr_flag_ == 1) {
                            denominator_ += di * dl * cab_[q1][q2] *
                                           (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);

                            numerator_ += di * dl * cab_[q1][q2] * std::log(cab_[q1][q2]) *
                                         (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        } else if (deg_corr_flag_ == 2) {
                            double tmp = di * dl * pab_[q1][q2];
                            denominator_ += tmp / (1.0 + tmp) *
                                           (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);

                            numerator_ += tmp / (1.0 + tmp) * std::log(cab_[q1][q2]) *
                                         (mmap_[i][l][q1] * mmap_[i2][l2][q2] + mmap_[i][l][q2] * mmap_[i2][l2][q1]);
                        }
                    }
                }
            }

            s_link += numerator_ / denominator_ / 2. / N_;
            if (by_site) {
                bethe_ent_[i] += numerator_ / denominator_ / 2. / N_;
                bethe_ent_edge_[i] += numerator_ / denominator_ / 2. / N_;
            }
//            std::clog << "edge entropy [" << i << "] = " << numerator_ / denominator_ / 2. / N_ << "\n";
        }
    }

    return s_link;
}


double belief_propagation::compute_free_energy_nonedge(bool by_site) noexcept {
    double log_f_non_edge_ = 0;
//    double _diff_log_f_non_edge_ = 0;

    double f_non_edge_ = 0;
//    double _diff_f_non_edge_ = 0;
    compute_na_expect();
    for (unsigned int q1 = 0; q1 < Q_; ++q1) {
        for (unsigned int q2 = q1; q2 < Q_; ++q2) {
            if (q1 == q2) {
                f_non_edge_ += -0.5*std::pow(cab_[q1][q2], beta_)*na_expect_[q1]/N_*na_expect_[q2]/N_;
//                _diff_f_non_edge_ += -0.5*std::pow(cab_[q1][q2], _diff_beta)*na_expect_[q1]/N_*na_expect_[q2]/N_;
//                        std::pow((1 - cab_[q1][q2] / N_), beta_) * real_psi_[i][q1] * real_psi_[l][q2];
            } else {
                f_non_edge_ += -1.*std::pow(cab_[q1][q2], beta_)*na_expect_[q1]/N_*na_expect_[q2]/N_;
//                _diff_f_non_edge_ += -1.*std::pow(cab_[q1][q2], _diff_beta)*na_expect_[q1]/N_*na_expect_[q2]/N_;
            }
        }
    }
    log_f_non_edge_ = f_non_edge_;
//    _diff_log_f_non_edge_ = _diff_f_non_edge_;

//
//    for (unsigned int i = 0; i < N_; ++i) {
//        auto nb_of_i = adj_list_ptr_->at(i);
//        for (unsigned int l = 0; l < N_; ++l) {
//            if (nb_of_i.find(l) == nb_of_i.end()) {  // l is not a neighbor to i
//                double f_non_edge_ = 0;
//                double _diff_f_non_edge_ = 0;
//
//                for (unsigned int q1 = 0; q1 < Q_; ++q1) {
//                    for (unsigned int q2 = 0; q2 < Q_; ++q2) {
//                        if (deg_corr_flag_ == 0) {
////
////                            f_non_edge_ +=
////                                    std::pow((1 - cab_[q1][q2] / N_), beta_) * real_psi_[i][q1] * real_psi_[l][q2];
////                            _diff_f_non_edge_ +=
////                                    (1 - std::pow(cab_[q1][q2], _diff_beta) / N_) * real_psi_[i][q1] * real_psi_[l][q2];
//                        } else {
//                            // should raise error or something (e.g. NotImplementedError)
//                        }
//                    }
//                }
//                if (f_non_edge_ != 0.) {
//                    log_f_non_edge_ += std::log(f_non_edge_);
//                    _diff_log_f_non_edge_ += std::log(_diff_f_non_edge_);
//                } else {
//                    // this could happen, esp. when epsilon is close to 0, where the network structure is strong
//                    // then, real_psi_[i] has some zero components.
//                }
//            }
//        }
//    }
//    log_f_non_edge_ /= 2. * N_;
//    _diff_log_f_non_edge_ /= 2. * N_;
//
//    std::clog << "the numerical entropy (NON-edge) is " << (_diff_log_f_non_edge_-log_f_non_edge_) / (_diff_beta - beta_) << "\n";
    return log_f_non_edge_;
}

double belief_propagation::compute_entropy_nonedge(bool by_site) noexcept {
    double f_non_edge = 0;
    for (unsigned int i = 0; i < N_; ++i) {
        auto nb_of_i = adj_list_ptr_->at(i);
        for (unsigned int l = 0; l < N_; ++l) {
//            if (std::find(graph_neis_[i].begin(), graph_neis_[i].end(), l) == graph_neis_[i].end()) {
            if (nb_of_i.find(l) == nb_of_i.end()) {
                numerator_ = 0.;
                denominator_ = 0.;
                // l is not a neighbor to i
                for (unsigned int q1 = 0; q1 < Q_; ++q1) {
                    for (unsigned int q2 = 0; q2 < Q_; ++q2) {
                        if (deg_corr_flag_ == 0) {
                            denominator_ += (1 - cab_[q1][q2] / N_) * real_psi_[i][q1] * real_psi_[l][q2];
                            numerator_ +=
                                    (cab_[q1][q2] / N_) * std::log(cab_[q1][q2]) * real_psi_[i][q1] * real_psi_[l][q2];
                        } else {
                            // should raise error or something (e.g. NotImplementedError)
                        }
                    }
                }
                if (numerator_ * denominator_ != 0) {
                    f_non_edge += -numerator_ / denominator_ / 2. / N_;
                    if (by_site) {
                        bethe_ent_[i] += -numerator_ / denominator_ / 2. / N_;
                    }
                } else {
                    // this could happen, esp. when epsilon is close to 0, where the network structure is strong
                    // then, real_psi_[i] has some zero components.
                }
            }
        }
    }
    return f_non_edge;
}


double belief_propagation::compute_free_energy(bool by_site) noexcept {
    double f_site = compute_free_energy_site(by_site);  // ~eq.29
    double f_link = compute_free_energy_edge(by_site);  // ~eq. 30
    double f_non_edge = compute_free_energy_nonedge(by_site);  // ~eq. 31
    //std::clog << "(f_site, f_link, f_non_edge) = (" << f_site << ", " << f_link << ", " << f_non_edge << ") \n";
    double totalf = (f_site + f_link + f_non_edge);
    std::clog << "(total_f, f_site, f_link, f_non_edge) = (" << totalf << ", " << f_site << ", " << f_link << ", " << f_non_edge << ") \n";

    return totalf;
}

double belief_propagation::compute_entropy(bool by_site) noexcept {
    double e_site = compute_entropy_site(by_site);
    double e_link = compute_entropy_edge(by_site);
//    double e_non_edge = compute_entropy_nonedge(by_site);  // computing this term is slow
    double e_non_edge = 0.;  // approx.: e_non_edge doesn't change much during the course of the BP algorithm.
    double total_e = (e_site + e_link + e_non_edge);
    std::clog << "(total_e, e_site, e_link, e_non_edge) = (" << total_e << ", " << e_site << ", " << e_link << ", " << e_non_edge << ") \n";
    return total_e;
}

void belief_propagation::compute_marg_entropy() noexcept {
    for (unsigned int i = 0; i < N_; ++i) {
        if (conf_planted_[i] == -1) {
            for (unsigned int q = 0; q < Q_; ++q) {
                marg_ent_[i] += -real_psi_[i][q] * std::log(real_psi_[i][q]);
            }
        } else {
            marg_ent_[i] = 0.;
        }

    }
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
    e_nishimori += -compute_free_energy_nonedge(false);
    return e_nishimori;
}


double belief_propagation::compute_overlap() noexcept {
//    output_vec<uint_vec_t>(conf_true_, std::clog);
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
    const auto n = adj_list_ptr_->at(i).size();
    for (unsigned int j = 0; j < n; ++j) {
        maxpom_psii_iter_[j] = xxx;
    }
    for (unsigned int q = 0; q < Q_; ++q) {
        a = 0.0;//log value
        // sum of all graphbors of i
        for (unsigned int l = 0; l < n; ++l) {
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
        for (unsigned int l = 0; l < n; ++l) {
            _mmap_q_nb_[q][l] = _real_psi_q_[q] - field_iter_[l];
            if (_mmap_q_nb_[q][l] > maxpom_psii_iter_[l]) maxpom_psii_iter_[l] = _mmap_q_nb_[q][l];
        }
    }
    for (unsigned int q = 0; q < Q_; ++q) {
        _real_psi_total_ += exp(_real_psi_q_[q] - maxpom_psi);
        for (unsigned int l = 0; l < n; ++l) {
            _mmap_total_[l] += exp(_mmap_q_nb_[q][l] - maxpom_psii_iter_[l]);
        }
    }

    update_h(i, -1);

    double mymaxdiff = -100.0;
    // normalization
    for (unsigned int q = 0; q < Q_; ++q) {
        real_psi_[i][q] = exp(_real_psi_q_[q] - maxpom_psi) / _real_psi_total_;
        for (unsigned int l = 0; l < n; ++l) {
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
        auto n = adj_list_ptr_->at(i).size();
        for (unsigned int l = 0; l < n; l++) {
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
    const double di = adj_list_ptr_->at(i).size();
    const auto n = adj_list_ptr_->at(i).size();
    double a, b;
    _real_psi_total_ = 0.0;

    for (unsigned int q = 0; q < Q_; ++q) {
        a = 1.0;
        for (unsigned int l = 0; l < n; ++l) {

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

                double norm = 0.0;
                for (unsigned int q_ = 0; q_ < Q_; ++q_) {
                    if (cab_[q_][q] != 0) {
                        mmap_[i][l][q_] = 1.;
                        norm += mmap_[i][l][q_];
                    } else {
                        mmap_[i][l][q_] = 0.;
                    }
                }
                for (unsigned int q_ = 0; q_ < Q_; ++q_) {
                    mmap_[i][l][q_] /= norm;
                }

                b = 0.0;
                neighbor = graph_neis_[i][l];
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

//                std::clog << "sanity check 1: this should not happen.\n";
//                std::clog << "i = " << i << "; l = " << l << "; q = "<< q << "\n";
//                output_vec(mmap_[i][l], std::clog);
//                std::clog << cab_[0][q] << " " << cab_[1][q]<< " " << cab_[2][q]<< " " << cab_[3][q]<< " " << cab_[4][q]<< "\n";
//                continue;
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

        for (unsigned int l = 0; l < n; ++l) {
            if (field_iter_[l] < EPS) {  // to cure the problem that field_iter_[l] may be very small
                double tmprob = 1.0;
                for (unsigned int lx = 0; lx < n; ++lx) {
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
    double mymaxdiff = -1;
    const auto n = adj_list_ptr_->at(i).size();
    for (unsigned int q = 0; q < Q_; ++q) {
        real_psi_[i][q] = _real_psi_q_[q] / _real_psi_total_;

//        if (n == 0) {
//            break;  // isolated nodes -->  we'll ask converge() to randomly select a new node.
//        }
        for (unsigned int l = 0; l < n; ++l) {
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
    // must return values that is >= 0;
    if (mymaxdiff != -1) {
//        std::clog << "mymaxdiff: " << mymaxdiff << " -- \n";
    }

    return mymaxdiff;
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
         * i.e. we propagate bp conditioned on knowing the correct label of the node,
         * then, we will believe that it is the true partition, which
         * emits constant messages to adjacent nodes.
         *
         * That is, we don't update messages on this node.
         */
        return 0;
    }

}

