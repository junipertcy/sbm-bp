#include <iostream>
#include "blockmodel.h"
#include "output_functions.h"  // TODO: debug use

using namespace std;

blockmodel_t::blockmodel_t(const uint_vec_t &memberships,
                           unsigned int Q,
                           unsigned int N,
                           unsigned int deg_corr_flag,
                           adj_list_t *adj_list_ptr) :

        random_block_(0, Q - 1),
        random_node_(0, N - 1) {
    deg_corr_flag_ = deg_corr_flag;
    memberships_ = memberships;
    adj_list_ptr_ = adj_list_ptr;
    n_.resize(Q, 0);
    deg_n_.resize(memberships.size(), 0);
    deg_.resize(memberships.size(), 0);
    num_edges_ = 0;
    entropy_from_degree_correction_ = 0.;
    graph_max_degree_ = 0;

    unsigned int deg_at_j_ = 0;
    double log_factorial = 0;

    for (unsigned int j = 0; j < memberships.size(); ++j) {
        ++n_[memberships[j]];
        deg_at_j_ = 0;
        for (auto nb = adj_list_ptr_->at(j).begin(); nb != adj_list_ptr_->at(j).end(); ++nb) {
            ++deg_at_j_;
            ++deg_[j];
            ++num_edges_;
        }
        if (deg_at_j_ >= graph_max_degree_) {
            graph_max_degree_ = deg_at_j_;
        }
        ++deg_n_[deg_at_j_];
    }
    for (unsigned int degree = 0; degree < memberships.size(); ++degree) {
        if (degree != 0) {
            log_factorial += std::log(degree);
        }
        entropy_from_degree_correction_ += deg_n_[degree] * log_factorial;
    }
    num_edges_ /= 2;
    compute_k();
}

adj_list_t *blockmodel_t::get_adj_list_ptr() const {
    return adj_list_ptr_;
}

int_vec_t blockmodel_t::get_k(unsigned int vertex) const { return k_[vertex]; }

bool blockmodel_t::are_connected(unsigned int vertex_a, unsigned int vertex_b) const {
    if (adj_list_ptr_->at(vertex_a).find(vertex_b) != adj_list_ptr_->at(vertex_a).end()) {
        return true;
    }
    return false;
}

int_vec_t blockmodel_t::get_size_vector() const { return n_; }

int_vec_t blockmodel_t::get_degree_size_vector() const { return deg_n_; }

int_vec_t blockmodel_t::get_degree() const { return deg_; }

uint_vec_t blockmodel_t::get_memberships() const { return memberships_; }

uint_mat_t blockmodel_t::get_m() const {
    uint_mat_t m(get_Q(), uint_vec_t(get_Q(), 0));
    for (auto vertex = 0; vertex < adj_list_ptr_->size(); ++vertex) {
        for (auto neighbour = adj_list_ptr_->at(vertex).begin();
             neighbour != adj_list_ptr_->at(vertex).end(); ++neighbour) {
            ++m[memberships_[vertex]][memberships_[*neighbour]];
        }
    }
    for (unsigned int r = 0; r < get_Q(); ++r) {
        for (unsigned int s = 0; s < get_Q(); ++s) {
            m[r][s] /= 2;  // edges are counted twice (the adj_list is symmetric)
            m[r][s] = m[s][r];  // symmetrize m matrix.
        }
    }
    return m;
}

unsigned int blockmodel_t::get_N() const { return unsigned(int(memberships_.size())); }

unsigned int blockmodel_t::get_Q() const { return unsigned(int(n_.size())); }


unsigned int blockmodel_t::get_E() const { return num_edges_; }

unsigned int blockmodel_t::get_deg_corr_flag() const { return deg_corr_flag_; }

double blockmodel_t::get_entropy_from_degree_correction() const { return entropy_from_degree_correction_; }

/// BP
unsigned int blockmodel_t::get_graph_max_degree() const { return graph_max_degree_; }

void blockmodel_t::shuffle(std::mt19937 &engine) {
    std::shuffle(memberships_.begin(), memberships_.end(), engine);
    compute_k();
}


void blockmodel_t::compute_k() {
    k_.clear();
    k_.resize(adj_list_ptr_->size());
    for (unsigned int i = 0; i < adj_list_ptr_->size(); ++i) {
        k_[i].resize(this->n_.size(), 0);
        for (auto nb = adj_list_ptr_->at(i).begin(); nb != adj_list_ptr_->at(i).end(); ++nb) {
            ++k_[i][memberships_[*nb]];
        }
    }
}

void blockmodel_t::compute_bp_params_from_memberships() {
    /*
     * calculate cab_, pab_, na_ from scratch (currently left un-used)
     */
    cab_.clear();
    pab_.clear();
    pab_.resize(get_Q());
    uint_mat_t _cab_ = get_m();  // now, this cab_ is the number of edges between group-a and group-b

    na_.clear();
    na_.resize(get_Q());
    nna_.clear();
    nna_.resize(get_Q());

    for (unsigned int i = 0; i < memberships_.size(); ++i) {
        na_[memberships_[i]] += 1.0;
        nna_[memberships_[i]] += deg_[i];
    }

    for (unsigned int q = 0; q < get_Q(); ++q) {
        eta_[q] = 1.0 * na_[q] / get_N();
        logeta_[q] = log(eta_[q]);
        pab_[q].resize(get_Q());
        for (unsigned int t = 0; t < get_Q(); ++t) {
            cab_[q][t] = _cab_[q][t] / nna_[q] / nna_[t] * get_N();
            pab_[q][t] = cab_[q][t] / get_N();
            logcab_[q][t] = std::log(cab_[q][t]);
            logcab_[t][q] = std::log(cab_[q][t]);
        }
    }
}

// TODO: check this function, it could be wrong
uint_mat_t blockmodel_t::compute_e_rs() {
    p_.clear();
    p_.resize(this->n_.size());
    for (unsigned int i = 0; i < n_.size(); ++i)  // this goes from i=0 to i=9;
    {
        p_[i].resize(this->n_.size(), 0);
    }
    for (unsigned int i = 0; i < adj_list_ptr_->size(); ++i)  // this goes from i=0 to i=1000;
    {
        for (auto nb = adj_list_ptr_->at(i).begin(); nb != adj_list_ptr_->at(i).end(); ++nb) {
            ++p_[memberships_[i]][memberships_[*nb]];
        }
    }
    return p_;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Non class methods
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bp_blockmodel_state bp_param_from_rand(const blockmodel_t &blockmodel, std::mt19937 &engine) {
    std::clog << "in bp_param_from_rand" << "\n";

    bp_blockmodel_state state;
    unsigned int N = blockmodel.get_N();
    unsigned int Q = blockmodel.get_Q();
    unsigned int M = blockmodel.get_E();
    state.cab.resize(Q);
    state.na.resize(Q);

    vector<double> pa_;
    for (unsigned int q = 0; q < Q; ++q) {
        pa_.push_back(1.0 / Q);
    }
    unsigned int tot_size = 0;

    for (unsigned int q = 0; q < Q - 1; ++q) {
        state.na[q] = unsigned(int(pa_[q] * N));  // TODO: check!
        tot_size += state.na[q];
    }
    state.na[Q - 1] = N - tot_size;

    vector<double> cab_;
    for (unsigned int q = 0; q < (Q + 1) * Q * 0.5; ++q) {
        cab_.push_back(std::uniform_real_distribution<>(0, 1)(engine));
    }
    output_vec<double_vec_t>(cab_, std::clog);
    unsigned int num = 0;
    std::clog << "Q " << Q << "\n";
    for (unsigned int i = 0; i < Q; ++i) {
        state.cab[i].resize(Q);
    }
    for (unsigned int i = 0; i < Q; ++i) {

        for (unsigned int j = i; j < Q; ++j) {
            state.cab[i][j] = cab_[num++];
            state.cab[j][i] = state.cab[i][j];
        }
    }
    // rescale cab start
    double cx = 0.0;
    for (unsigned int q = 0; q < Q; ++q) {
        for (unsigned int t = q; t < Q; ++t) {
            if (q == t) cx += state.cab[q][t] * 0.5 / Q / Q;
            else cx += state.cab[q][t] / Q / Q;
        }
    }
    for (unsigned int q = 0; q < Q; ++q) {
        for (unsigned int t = q; t < Q; ++t) {
            state.cab[q][t] = 2.0 * N / M * N / M * state.cab[q][t] / cx / 4.0;
            state.cab[t][q] = state.cab[q][t];
        }
    }
    // rescale cab end
    return state;
}

bp_blockmodel_state bp_param_from_epsilon_c(const blockmodel_t &blockmodel, double epsilon, double c) {
    unsigned int N = blockmodel.get_N();
    unsigned int Q = blockmodel.get_Q();

    bp_blockmodel_state state;
    state.na.resize(Q);
    state.cab.resize(Q);

    double cin = 0, co = 0;

    vector<double> pa_;

    unsigned int tot_size = 0;
    for (unsigned int q = 0; q < Q; ++q) {
        if (q == Q - 1) {
            state.na[q] = N - tot_size;
        }
        pa_.push_back(1.0 / Q);
        tot_size += pa_[q] * N;
        state.na[q] = unsigned(int(pa_[q] * N));
    }

    if (epsilon < 0) {  // make it a completely dis-assortative network
        std::clog << "epsilon < 0; we enforce it to a completely disassortative network.";
        cin = 0;
        co = c * Q / (Q - 1);
    } else {
        cin = c * Q / ((Q - 1) * epsilon + 1);  // TODO: check with original code...
        co = epsilon * cin;
    }

    for (unsigned int q = 0; q < Q; ++q) {
        state.cab[q].resize(Q);
    }

    for (unsigned int q = 0; q < Q; ++q) {
        state.cab[q][q] = cin;  // TODO: different from original definition, check
        for (unsigned int t = q + 1; t < Q; ++t) {
            state.cab[q][t] = co;
            state.cab[t][q] = state.cab[q][t];
        }
    }
    return state;
}

bp_blockmodel_state bp_param_from_direct(const blockmodel_t &blockmodel, double_vec_t pa, double_vec_t cab) {
    unsigned int N = blockmodel.get_N();
    unsigned int Q = blockmodel.get_Q();

    bp_blockmodel_state state;
    state.na.resize(Q);
    unsigned int tot_size = 0;
    for (unsigned int q = 0; q < Q; ++q) {
        if (q == Q - 1) {
            state.na[q] = N - tot_size;
        }
        tot_size += unsigned(int(pa[q] * N));
        state.na[q] = unsigned(int(pa[q] * N));
    }

    state.cab.resize(Q);
    for (unsigned int q = 0; q < Q; ++q) {
        state.cab[q].resize(Q);
    }

    for (unsigned int q = 0; q < Q; ++q) {
        state.cab[q][q] = cab[q * Q - q * (q - 1) / 2];
        for (unsigned int t = q + 1; t < Q; ++t) {
            state.cab[q][t] = cab[q * Q - q * (q - 1) / 2 + t - q];
            state.cab[t][q] = state.cab[q][t];
        }
    }
    return state;
}



