// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <kronecker_edge_generator.hpp>
#include <ygm/comm.hpp>

int main(int argc, char** argv) {
  ygm::comm world(&argc, &argv);

  std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> g1{{0, 1, 1},
                                                           {1, 0, 2}};
  std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> g2{
      {0, 1, 2}, {1, 1, 3}, {2, 0, 1}};

  kronecker_edge_generator kron(world, g1, g2, 2, 3);

  kron.for_all([&world](const auto row, const auto col, const auto val) {
    world.cout() << "(" << row << ", " << col << ") : " << val << std::endl;
  });

  return 0;
}
