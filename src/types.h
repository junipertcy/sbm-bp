//
// Created by Tzu-Chi Yen on 3/13/17.
//

#ifndef BP_SELECTION_TYPES_H
#define BP_SELECTION_TYPES_H

#include <vector>
#include <set>
#include <utility>

using edge_t = std::pair<unsigned int, unsigned int>;
using edge_list_t = std::vector<edge_t>;
using neighbourhood_t = std::set<unsigned int>;
using adj_list_t = std::vector<neighbourhood_t>;

using mcmc_move_t = struct mcmc_move_t {
    unsigned int vertex;
    unsigned int source;
    unsigned int target;
};

using bp_blockmodel_state = struct bp_blockmodel_state {
    std::vector<std::vector<double> > cab;
    std::vector<unsigned int> na;
};

using bp_message = struct bp_message {
    std::vector<std::vector<double> > real_psi;
    std::vector<std::vector<std::vector<double> > > mmap;
};

using uint_vec_t = std::vector<unsigned int>;
using int_vec_t = std::vector<int>;
using float_vec_t = std::vector<float>;
using double_vec_t = std::vector<double>;
using uint_mat_t = std::vector<std::vector<unsigned int> >;
using int_mat_t = std::vector<std::vector<int> >;
using float_mat_t = std::vector<std::vector<float> >;
using double_mat_t = std::vector<std::vector<double> >;
using double_tensor_t = std::vector<std::vector<std::vector<double> > >;

#endif //BP_SELECTION_TYPES_H
