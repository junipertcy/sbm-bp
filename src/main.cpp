/* Program to perform BP on the degree-corrected SBM via reading from an edgelist file
 *
 * Written by Tzu-Chi Yen 13 MAR 2017
 */

/* ~~~~~~~~~~~~~~~~ Notes ~~~~~~~~~~~~~~~~ */
// A few base assumption go into this program:
//
// - We assume that node identifiers are zero indexed contiguous integers.
// - We assume that block memberships are zero indexed contiguous integers.
// - We assume that the SBM is of the undirected and simple variant.

// Standard Template Library (STL)
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

// Boost
#include <boost/program_options.hpp>
#include <curses.h>

// Program headers
#include "types.h"
#include "blockmodel.h"
#include "output_functions.h"
#include "graph_utilities.h"
#include "belief_propagation.h"
#include "config.h"

namespace po = boost::program_options;

int main(int argc, char const *argv[]) {
    /* ---- Program options ---- */

    std::string edge_list_path;  // -l
    std::string mode; // -m
    /// Setting memberships; cases from argument var_map (mb_rand | mb_n | mb | mb_path)
    std::string memberships_status;
    bool memberships_randomize = false;
    uint_vec_t mb;  // -mb
    std::string mb_path;

    /// cab and na
    double_vec_t epsilon_c;  // TODO: check its size (must be only 2!)
    unsigned int seed = 0;
    uint_vec_t num_nodes_per_type;
    uint_vec_t n;
    double beta;

    /// belief propagation message
    unsigned int bp_messages_init_flag;
    std::string beliefs_path;
    std::string true_conf_path;

    /// degree correction
    unsigned int deg_corr_flag;

    /// rates and criterions for BP
    float learning_rate;
    float dumping_rate;
    float bp_conv_crit;
    float learning_conv_crit;
    unsigned int time_conv;


    std::string bm_params_string;
    /// Setting cab and pa; case I
    bool cab_ppm = false;
    bool cab_ec = false;
    uint_vec_t num_nodes_per_block;
    float_vec_t probabilities;   // if (use_ppm | use_eps_c) is on, it will be interpreted differently
    uint_vec_t fixed_nodes;
    /// Setting cab and pa; case II
    double_vec_t pa;
    double_vec_t cab;

    /// for init_special_needs
    bool output_marginals = false;
    bool output_free_energy = false;
    bool output_weighted_free_energy = false;
    bool output_entropy = false;
    bool output_weighted_entropy = false;

    /// Setting cab and pa; case III – read from file
    std::string cab_file;
    /// Setting cab and pa; case IV – random, just do shuffling
    bool cab_rand = false;

    po::options_description description("Options");
    description.add_options()
            /// file handles
            ("edge_list_path,l", po::value<std::string>(&edge_list_path), "Path to the input edgelist file.")
            ("n,n", po::value<uint_vec_t>(&n)->multitoken(), "Block sizes vector.\n")
            ("beta,b", po::value<double>(&beta)->default_value(1.), "beta, the inverse temperature")
            ("mb_rand", "Randomize initial block memberships.")
            ("mb_n", "Initialize membership from n [DEFAULT].")
            ("mb", po::value<uint_vec_t>(&mb)->multitoken(), "Directly initialize membership from input vector.")
            ("mb_path", po::value<std::string>(&mb_path), "use an external file to define the memberships.")
            ("epsilon_c", po::value<double_vec_t>(&epsilon_c)->multitoken(),
             "Assign epsilon and c to define cab and pa [DEFAULT].")

            // bp message
            ("bp_messages_init_flag,i", po::value<unsigned int>(&bp_messages_init_flag)->default_value(0),
             "flag to initialize BP. Valid values are 0, 1, 2, and 3. Defualt 0."\
            "0 means initialized by random messages, "\
            "1 means initialized by partly planted configuration, others left random"\
            "2 means initialized by planted configuration, with random noise on all sites"\
            "3 means initialized by fixed planted configuration.")
            ("beliefs_path", po::value<std::string>(&beliefs_path), "Path to planted membership.")
            ("true_conf_path", po::value<std::string>(&true_conf_path), "Path to true membership.")
            ("deg_corr_flag", po::value<unsigned int>(&deg_corr_flag)->default_value(0),
             "flag to indicate whether there is degree correction in the SBM"\
            "0 means no degree correction, "\
            "1 means degree corection, "\
            "2 means yet another version of degree correction. (to be checked)")

            // rates and criterions
            ("learning_rate,r", po::value<float>(&learning_rate)->default_value(0.2),
             "set learning_rate, from 0.0 to 1.0. The larger value, the faster learning. Default 0.2.")
            ("dumping_rate,R", po::value<float>(&dumping_rate)->default_value(1.0),
             "set dumping_rate, from 0.0 to 1.0. The larger value, the faster converge. default 1.0 (no dumping).")
            ("bp_conv_crit,e", po::value<float>(&bp_conv_crit)->default_value(5.0e-6),
             "set bp_conv_crit, convergence criterium of BP and MF, from 0.0 to 1.0. Default 5.0e-6.")
            ("learning_conv_crit,E", po::value<float>(&learning_conv_crit)->default_value(1.0e-6),
             "set learning_conv_crit, convergence criterium of learning, from 0.0 to 1.0. Default 1.0e-6.")
            ("time_conv,t", po::value<unsigned int>(&time_conv)->default_value(100),
             "set time_conv, maximum time for BP to converge, default 100.")


            ("probabilities,P", po::value<float_vec_t>(&probabilities)->multitoken(),
             "In normal mode (SBM): probability matrix in row major order. In PPM mode: p_in followed by p_out.")
            ("fixed_nodes,f", po::value<uint_vec_t>(&fixed_nodes)->multitoken(), "Fixed nodes with known labels.")
            ("cab_rand,u0", "Randomized cab and pa (defaults to SBM).")
            ("cab_ppm,u1",
             "Use planted partition probabilities to define cab and pa (defaults to SBM).")
            ("cab_ec,u2",
             "Use epsilon and c to define cab and pa (defaults to SBM).")
            ("cab_file,u3", po::value<std::string>(&cab_file),
             "use an external file to define cab and pa (defaults to SBM).")
            ("pa", po::value<double_vec_t>(&pa)->multitoken(), "pa vector.\n")
            ("cab", po::value<double_vec_t>(&cab)->multitoken(), "cab vector.\n")

            ("marginals", "whether output marginals in the infer mode")
            ("free_energy", "Output free_energy in the infer mode")
            ("weighted_free_energy", "Output weighted_free_energy in the infer mode")
            ("entropy", "Output entropy in the infer mode")
            ("weighted_entropy", "Output weighted_entropy in the infer mode")


            ("mode,m", po::value<std::string>(&mode), "Mode for the algorithm; valid values: infer | learn.")
            ("seed,d", po::value<unsigned int>(&seed),
             "Seed of the pseudo random number generator (Mersenne-twister 19937). "\
    "A random seed is used if seed is not specified.")
            ("help,h",
             "Produce this help message.");

    po::variables_map var_map;
    po::store(po::parse_command_line(argc, argv, description), var_map);
    po::notify(var_map);


    if (var_map.count("help") > 0 || argc == 1) {
        std::clog << "BP algorithms for the SBM (final output only)\n";
        std::clog << "Usage:\n"
                  << "  " + std::string(argv[0]) + " [--option_1=value] [--option_s2=value] ...\n";
        std::clog << description;
        return 0;
    }

    if (var_map.count("edge_list_path") == 0) {
        std::clog << "edge_list_path is required (-e flag)\n";
        return 1;
    }

    if (var_map.count("mode") == 0) {
        std::clog << "mode is required (-m flag)\n";
        return 1;
    }

    if (var_map.count("n") == 0) {
        std::clog << "n is required (-n flag)\n";
        return 1;
    }

    // control membership assignment parameters
    if (var_map.count("mb_n") + var_map.count("mb") + var_map.count("mb_path") > 1) {
        std::clog << "Error! Please just select one option to assign the membership vector.\n";
        return 1;
    } else if (var_map.count("mb_n") + var_map.count("mb") + var_map.count("mb_path") == 0) {
        // default
        memberships_status = "from_n";
    }

    if (var_map.count("mb_rand") > 0) {
        memberships_randomize = true;
    } else if (var_map.count("mb_n") > 0) {
        memberships_status = "from_n";
    } else if (var_map.count("mb") > 0) {
        memberships_status = "direct";
    } else if (var_map.count("mb_path") > 0) {
        memberships_status = "from_file";
    }

    if (var_map.count("epsilon_c") + (var_map.count("pa") * var_map.count("cab")) > 1) {
        std::clog << "Error! Please just choose one way to initialize the pa/cab parameter.\n";
        return 1;
    } else if (var_map.count("epsilon_c") == 0 && (var_map.count("pa") + var_map.count("cab")) < 2) {
        std::clog << "Error! Please just input both pa/cab parameters.\n";
        return 1;
    } else if (var_map.count("epsilon_c") > 0) {
        bm_params_string = "cab_ec";
    } else if ((var_map.count("pa") + var_map.count("cab")) == 2) {
        bm_params_string = "cab_direct";
    }

    if (var_map.count("bp_messages_init_flag") > 0) {
        if (bp_messages_init_flag != 0 && var_map.count("fixed_nodes") == 0) {
            if (var_map.count("beliefs_path") == 0) {
                std::clog << "Error! Please assign the file path of the initial belief of node membership.\n";
                return 1;
            }
        } else if (var_map.count("fixed_nodes") > 0) {
            std::clog << "Randomly assign initial messages, except certain fixed nodes.\n";
        } else {
            std::clog << "Randomly assign initial messages!\n";
        }
    }

    if (var_map.count("marginals") > 0) {
        output_marginals = true;
    }
    if (var_map.count("free_energy") > 0) {
        output_free_energy = true;
    }
    if (var_map.count("weighted_free_energy") > 0) {
        output_weighted_free_energy = true;
    }
    if (var_map.count("entropy") > 0) {
        output_entropy = true;
    }
    if (var_map.count("weighted_entropy") > 0) {
        output_weighted_entropy = true;
    }

    if (var_map.count("randomize") > 0) {
        bool randomize = true;
    }


    if (var_map.count("seed") == 0) {
        // seeding based on the clock
        seed = (unsigned int) std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }

    /* ---- Setup objects ---- */
    std::mt19937 engine(seed);

    /// Primary parameters for a network (whichever generative model it is generated)
    uint_vec_t memberships_init;  // memberships from block sizes; TODO: add option – user input
    if (memberships_status == "from_n") {
        unsigned int accu = 0;
        for (auto it: n) {
            accu += it;
        }
        memberships_init.resize(accu, 0);
        unsigned shift = 0;
        for (unsigned int r = 0; r < n.size(); ++r) {
            for (unsigned int i = 0; i < n[r]; ++i) {
                memberships_init[shift + i] = r;
            }
            shift += n[r];
        }
    } else if (memberships_status == "direct") {
        unsigned int accu = 0;
        for (auto it: n) {
            accu += it;
        }
        memberships_init.resize(accu, 0);
        for (unsigned int m = 0; m < accu; ++m) {
            memberships_init[m] = mb[m];
        }
        // sanity check!
        if (mb.size() != accu) {
            std::clog << "Error! Size of assigned membership vector does not fit the number of nodes assigned by n.\n";
            return 1;
        }
    } else if (memberships_status == "from_file") {
        // TODO!
    }

    auto Q = unsigned(int(n.size()));  // number of blocks

    unsigned int N = 0;  // number of vertices
    for (unsigned int i = 0; i < Q; ++i) {
        N += n[i];
    }
    edge_list_t edge_list;  // Graph structure
    load_edge_list(edge_list, edge_list_path);

    adj_list_t adj_list = edge_to_adj(edge_list, N);
    edge_list.clear();

    uint_vec_t true_conf;
    if (var_map.count("true_conf_path") == 0) {
        std::clog << "Warning! Assign true conf using ordered node membership.\n";
        true_conf = memberships_init;
    } else {
        bool load_confs_res = load_confs(true_conf, true_conf_path);
        if (!load_confs_res) {
            std::clog << "Warning! Reading true_conf_path error. Assign true conf using ordered node membership.\n";
            true_conf = memberships_init;
        }
    }

    // Now, initiate the blockmodel instance
    blockmodel_t blockmodel(memberships_init, Q, (unsigned) adj_list.size(), deg_corr_flag, &adj_list);
    memberships_init.clear();

    if (memberships_randomize) {
        blockmodel.shuffle(engine);
    }

    /*
     * As the (1) memberships; (2) Q; (3) E (total num of edges); (4) adj_list are four primary parameters to
     * specify a network. The secondary parameters such as (1) pa, and (2) cab, are important for passing it
     * to algorithms for inference and learning.
     */

    /// Secondary parameters for particular generative model for inference and learning
    /*
     * There are four ways to define pa and cab:
     * (1) Read from file. TODO: to implement
     * (2) Set by -P parameter. (epsilon,c mode)
     * (3) Set by -p and -c parameter.
     * (4) Random.
     */

    std::unique_ptr<belief_propagation> algorithm;
    if (mode == "learn") {
        algorithm = std::make_unique<bp_basic>();
    } else if (mode == "infer") {
        algorithm = std::make_unique<bp_conditional>();
    }

    int_vec_t beliefs;
    // Note that we will read the beliefs file anyway; although there could be no such file, it will be okay.
    load_beliefs(beliefs, beliefs_path);


    // The labels of certain nodes can be set fixed here.
    if (var_map.count("fixed_nodes") > 0) {
        beliefs.resize(true_conf.size(), -1);
        for (auto vtx: fixed_nodes) {
            beliefs[vtx] = true_conf[vtx];
        }
    }

    algorithm->init_messages(blockmodel, bp_messages_init_flag, beliefs, true_conf, engine);

    algorithm->init_special_needs(output_marginals, output_free_energy, output_weighted_free_energy, output_entropy, output_weighted_entropy);

    algorithm->set_beta(beta);

    beliefs.clear();
    true_conf.clear();

    bp_blockmodel_state state;
    if (bm_params_string == "cab_ec") {
        state = bp_param_from_epsilon_c(blockmodel, epsilon_c[0], epsilon_c[1]);
        epsilon_c.clear();
    } else if (bm_params_string == "cab_direct") {
        state = bp_param_from_direct(blockmodel, pa, cab);
        pa.clear();
        cab.clear();
    }

    /// Configuration outputs
#if LOGGING == 1
    // TODO
#endif

    /// actual algorithm
    if (mode == "infer") {
        algorithm->inference(blockmodel, state, bp_conv_crit, time_conv, dumping_rate, engine);
    } else if (mode == "learn") {
        algorithm->learning(blockmodel, state, learning_conv_crit, time_conv, learning_rate, dumping_rate, engine);
    }
    return 0;
}