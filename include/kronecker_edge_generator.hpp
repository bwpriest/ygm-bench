// Copyright 2019-2021 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <fstream>

#include <ygm/comm.hpp>

namespace kronecker {

template <typename T, typename W>
T read_replicated_graph_file(ygm::comm& world, std::string filename,
                             std::vector<std::tuple<T, T, W>>& edge_list) {
  edge_list.clear();
  T    num_vertices(0);
  auto num_vertices_ptr = world.make_ygm_ptr(num_vertices);
  auto edge_list_ptr    = world.make_ygm_ptr(edge_list);

  if (world.rank0()) {
    std::ifstream filestream(filename);

    if (filestream.is_open()) {
      std::string line;
      if (!std::getline(filestream, line)) {
        std::cerr << "Empty file\n";
        exit(-1);
      }
      std::istringstream iss(line);
      iss >> num_vertices;
      if (line.find(" ") != line.npos) {
        std::cout << iss.str() << std::endl;
        std::cerr << "First line of input has too many values\n";
        exit(-1);
      }
      while (std::getline(filestream, line)) {
        std::istringstream iss2(line);
        T                  src, dest;
        W                  wgt;
        if (!(iss2 >> src >> dest >> wgt)) {
          std::cerr << "Malformed line in input\n";
          exit(-1);
        } else {
          edge_list.push_back(std::make_tuple(src, dest, wgt));
          // Forcing to be symmetric, at least for now...
          edge_list.push_back(std::make_tuple(dest, src, wgt));
        }
      }
      filestream.close();
    } else {
      std::cerr << "Unable to open file " << filename << std::endl;
      exit(-1);
    }

    world.async_bcast(
        [](const auto num_vertices, const auto& edge_list,
           auto num_vertices_ptr, auto edge_list_ptr) {
          *num_vertices_ptr = num_vertices;
          *edge_list_ptr    = edge_list;
        },
        num_vertices, edge_list, num_vertices_ptr, edge_list_ptr);
  }

  world.barrier();

  return num_vertices;
}

}  // namespace kronecker

template <
    typename edge_data_type = uint32_t,
    typename C1 = std::vector<std::tuple<uint64_t, uint64_t, edge_data_type>>,
    typename C2 = std::vector<std::tuple<uint64_t, uint64_t, edge_data_type>>>
class kronecker_edge_generator {
 public:
  using vertex_t    = uint64_t;
  using edge_data_t = edge_data_type;
  using edge_t      = std::tuple<uint64_t, uint64_t, edge_data_t>;

 public:
  kronecker_edge_generator(ygm::comm& c, C1 graph1, C2 graph2,
                           uint64_t num_vertices_graph1,
                           uint64_t num_vertices_graph2)
      : m_graph1(graph1),
        m_graph2(graph2),
        m_comm(c),
        m_num_vertices_graph1(num_vertices_graph1),
        m_num_vertices_graph2(num_vertices_graph2) {
    m_vertex_scale =
        (uint64_t)ceil(log2(m_num_vertices_graph1 * m_num_vertices_graph2));
  }

  kronecker_edge_generator(ygm::comm& c, std::string filename1,
                           std::string filename2)
      : m_comm(c) {
    m_num_vertices_graph1 =
        kronecker::read_replicated_graph_file(m_comm, filename1, m_graph1);
    m_num_vertices_graph2 =
        kronecker::read_replicated_graph_file(m_comm, filename2, m_graph2);

    m_vertex_scale =
        (uint64_t)ceil(log2(m_num_vertices_graph1 * m_num_vertices_graph2));
    if (m_comm.rank0()) {
      std::cout << "Vertex Scale: " << m_vertex_scale << std::endl;
      std::cout << "sizeof(size_t) = " << sizeof(size_t) << std::endl;
    }
  }

  template <typename Function>
  void for_all(Function fn) {
    size_t graph1_pos = m_comm.rank();

    while (graph1_pos < m_graph1.size()) {
      const edge_t& graph1_edge = m_graph1.at(graph1_pos);

      std::for_each(m_graph2.begin(), m_graph2.end(),
                    [fn, &graph1_edge, this](const edge_t& graph2_edge) {
                      vertex_t    row1 = std::get<0>(graph1_edge);
                      vertex_t    col1 = std::get<1>(graph1_edge);
                      edge_data_t val1 = std::get<2>(graph1_edge);
                      vertex_t    row2 = std::get<0>(graph2_edge);
                      vertex_t    col2 = std::get<1>(graph2_edge);
                      edge_data_t val2 = std::get<2>(graph2_edge);

                      vertex_t row = row1 * this->m_num_vertices_graph2 + row2;
                      vertex_t col = col1 * this->m_num_vertices_graph2 + col2;
                      edge_data_t val = val1 * val2;

                      fn(row, col, val);
                    });

      graph1_pos += m_comm.size();
    }
  }

 private:
  ygm::comm m_comm;

  C1       m_graph1;
  C2       m_graph2;
  uint64_t m_num_vertices_graph1;
  uint64_t m_num_vertices_graph2;
  uint64_t m_vertex_scale;
};
