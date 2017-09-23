//
// Created by Tzu-Chi Yen on 3/13/17.
//

#ifndef BP_SELECTION_GRAPH_UTILITIES_H
#define BP_SELECTION_GRAPH_UTILITIES_H

#include <string>
#include <fstream>
#include <sstream>
#include "types.h"

/* Load beliefs of memberships of each node. Returns true on success. */
bool load_beliefs(int_vec_t & beliefs, const std::string beliefs_path) noexcept;

bool load_confs(uint_vec_t & beliefs, const std::string beliefs_path) noexcept;

/* Load an edge list. Result passed by reference. Returns true on success. */
bool load_edge_list(edge_list_t & edge_list, const std::string edge_list_path) noexcept;

/* Convert adjacency list to edge list. Result passed by reference. */
adj_list_t edge_to_adj(const edge_list_t & edge_list, unsigned int num_vertices=0) noexcept;

#endif //BP_SELECTION_GRAPH_UTILITIES_H
