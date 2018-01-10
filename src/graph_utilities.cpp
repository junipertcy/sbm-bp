//
// Created by Tzu-Chi Yen on 3/13/17.
//

#include <iostream>
#include "graph_utilities.h"

bool load_beliefs(int_vec_t & beliefs, const std::string beliefs_path) noexcept
{
    beliefs.clear();
    std::ifstream beliefs_file(beliefs_path.c_str());
    if (!beliefs_file.is_open()) return false;
    std::string line_buffer;
    int membership;
    while (getline(beliefs_file, line_buffer))
    {
        std::stringstream linestream(line_buffer);
        linestream >> membership;
        beliefs.push_back(membership);
    }
    beliefs_file.close();
    return true;
}

bool load_confs(uint_vec_t & beliefs, const std::string beliefs_path) noexcept
{
    beliefs.clear();
    std::ifstream beliefs_file(beliefs_path.c_str());
    if (!beliefs_file.is_open()) return false;
    std::string line_buffer;
    unsigned int membership;
    while (getline(beliefs_file, line_buffer))
    {
        std::stringstream linestream(line_buffer);
        linestream >> membership;
        beliefs.push_back(membership);
    }
    beliefs_file.close();
    return true;
}

bool load_edge_list(edge_list_t & edge_list, const std::string edge_list_path) noexcept
{
    edge_list.clear();
    std::ifstream edge_list_file(edge_list_path.c_str());
    if (!edge_list_file.is_open()) return false;
    std::string line_buffer;
    unsigned int node_a, node_b;
    while (getline(edge_list_file, line_buffer))
    {
        std::stringstream linestream(line_buffer);
        linestream >> node_a;
        linestream >> node_b;
        edge_list.push_back(std::make_pair(node_a, node_b));
    }
    edge_list_file.close();
    return true;
}

adj_list_t edge_to_adj(const edge_list_t & edge_list, unsigned int num_vertices) noexcept
{
    adj_list_t adj_list(num_vertices);
    for (auto edge = edge_list.begin(); edge != edge_list.end(); ++edge)
    {
        if (edge->first >= adj_list.size())
        {
            adj_list.resize(edge->first + 1);
        }
        if (edge->second >= adj_list.size())
        {
            adj_list.resize(edge->second + 1);
        }
        adj_list[edge->first].insert(edge->second);
        adj_list[edge->second].insert(edge->first);
    }
    return adj_list;
}