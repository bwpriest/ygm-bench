// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <kronecker_edge_generator.hpp>
#include <ygm/comm.hpp>

int main(int argc, char** argv) {
  ygm::comm world(&argc, &argv);

  {  // Construct from graph in memory
    std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> g1{{0, 1, 1},
                                                             {1, 0, 2}};
    std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> g2{
        {0, 1, 2}, {1, 1, 3}, {2, 0, 1}};

    kronecker_edge_generator kron(world, g1, g2, 2, 3);

    kron.for_all([&world](const auto row, const auto col, const auto val) {
      world.cout() << "(" << row << ", " << col << ") : " << val << std::endl;
    });
  }

  if (argc == 3) {  // Construct from graph in file
    std::string graphA_filename(argv[1]);
    std::string graphB_filename(argv[2]);

    kronecker_edge_generator kron(world, graphA_filename, graphB_filename);

    uint64_t num_edges{0};

    kron.for_all([&num_edges](const auto row, const auto col, const auto val) {
      ++num_edges;
    });

    world.barrier();

    world.cout0("Kronecker edges: ", world.all_reduce_sum(num_edges));
  }

  return 0;
}
