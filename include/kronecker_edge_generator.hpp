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

template <typename T>
void if_greater_set(T obs, T& counter) {
  if (obs > counter) {
    counter = obs;
  }
}

template <typename T, typename W>
std::tuple<T, T> read_graph_challenge_file(
    ygm::comm& world, std::string filename,
    std::vector<std::tuple<T, T, W>>& edge_list) {
  edge_list.clear();
  T    num_vertices(0);
  T    num_edges(0);
  auto num_vertices_ptr = world.make_ygm_ptr(num_vertices);
  auto num_edges_ptr    = world.make_ygm_ptr(num_edges);
  auto edge_list_ptr    = world.make_ygm_ptr(edge_list);

  if (world.rank0()) {
    std::ifstream filestream(filename);

    if (filestream.is_open()) {
      std::string line;
      if (!std::getline(filestream, line)) {
        std::cerr << "Empty file\n";
        exit(-1);
      }
      do {
        std::istringstream iss(line);
        T                  src, dest;
        W                  wgt;
        if (!(iss >> src >> dest >> wgt)) {
          std::cerr << "Malformed line in input\n";
          exit(-1);
        } else {
          if_greater_set(src, num_vertices);
          if_greater_set(dest, num_vertices);
          --src;
          --dest;
          edge_list.push_back(std::make_tuple(src, dest, wgt));
          ++num_edges;
          // Forcing to be symmetric, at least for now...
          edge_list.push_back(std::make_tuple(dest, src, wgt));
          ++num_edges;
        }
      } while (std::getline(filestream, line));
      filestream.close();
    } else {
      std::cerr << "Unable to open file " << filename << std::endl;
      exit(-1);
    }

    world.async_bcast(
        [](const auto num_vertices, const auto num_edges, const auto& edge_list,
           auto num_vertices_ptr, const auto num_edges_ptr,
           auto edge_list_ptr) {
          *num_vertices_ptr = num_vertices;
          *num_edges_ptr    = num_edges;
          *edge_list_ptr    = edge_list;
        },
        num_vertices, num_edges, edge_list, num_vertices_ptr, num_edges_ptr,
        edge_list_ptr);
  }

  world.barrier();

  return {num_vertices, num_edges};
}

template <typename T, typename C>
std::tuple<T, C> read_graph_challenge_truth_file(
    ygm::comm& world, std::string filename,
    std::vector<std::tuple<T, C>>& community_list) {
  community_list.clear();
  T    num_vertices(0);
  C    num_communities(0);
  auto num_vertices_ptr    = world.make_ygm_ptr(num_vertices);
  auto num_communities_ptr = world.make_ygm_ptr(num_communities);
  auto community_list_ptr  = world.make_ygm_ptr(community_list);

  if (world.rank0()) {
    std::ifstream filestream(filename);

    if (filestream.is_open()) {
      std::string line;
      if (!std::getline(filestream, line)) {
        std::cerr << "Empty file\n";
        exit(-1);
      }
      do {
        std::istringstream iss(line);
        T                  vtx;
        C                  cmty;
        if (!(iss >> vtx >> cmty)) {
          std::cerr << "Malformed line in input\n";
          exit(-1);
        } else {
          if_greater_set(vtx, num_vertices);
          if_greater_set(cmty, num_communities);
          --vtx;
          --cmty;
          community_list.push_back({vtx, cmty});
        }
      } while (std::getline(filestream, line));
      filestream.close();
    } else {
      std::cerr << "Unable to open file " << filename << std::endl;
      exit(-1);
    }

    world.async_bcast(
        [](const auto num_vertices, const auto num_communities,
           const auto& community_list, auto num_vertices_ptr,
           auto num_communities_ptr, auto community_list_ptr) {
          *num_vertices_ptr    = num_vertices;
          *num_communities_ptr = num_communities;
          *community_list_ptr  = community_list;
        },
        num_vertices, num_communities, community_list, num_vertices_ptr,
        num_communities_ptr, community_list_ptr);
  }

  world.barrier();

  return {num_vertices, num_communities};
}

template <typename V1, typename V2>
void check_agreement(ygm::comm& comm, V1 n1, V2 n2, std::string edge_file,
                     std::string truth_file) {
  if (n1 != n2) {
    comm.cerr0("edge stream file ", edge_file, " vertex count (", n1,
               ") disagress with community stream file ", truth_file,
               " vertex count (", n2, ")");
    exit(-1);
  }
}
}  // namespace kronecker

template <
    typename vertex_type = uint64_t, typename edge_data_type = uint32_t,
    typename GS1 =
        std::vector<std::tuple<vertex_type, vertex_type, edge_data_type>>,
    typename GS2 =
        std::vector<std::tuple<vertex_type, vertex_type, edge_data_type>>,
    typename community_data_type = uint32_t,
    typename TS1 = std::vector<std::tuple<vertex_type, community_data_type>>,
    typename TS2 = std::vector<std::tuple<vertex_type, community_data_type>>>
class kronecker_edge_generator {
 public:
  using vertex_t         = vertex_type;
  using edge_data_t      = edge_data_type;
  using community_data_t = community_data_type;
  using edge_t           = std::tuple<vertex_t, vertex_t, edge_data_t>;
  using truth_t          = typename TS2::value_type;

 public:
  kronecker_edge_generator(ygm::comm& c, GS1 graph1, GS2 graph2,
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
    m_comm.cout0("Vertex Scale: ", m_vertex_scale);
    m_comm.cout0("sizeof(size_t) = ", sizeof(size_t));
  }

  kronecker_edge_generator(ygm::comm& c, std::string graph_filename1,
                           std::string graph_filename2,
                           std::string truth_filename1,
                           std::string truth_filename2)
      : m_comm(c) {
    auto [_num_vertices_graph1, _num_edges_graph1] =
        kronecker::read_graph_challenge_file(m_comm, graph_filename1, m_graph1);
    auto [_num_vertices_graph2, _num_edges_graph2] =
        kronecker::read_graph_challenge_file(m_comm, graph_filename2, m_graph2);

    m_num_vertices_graph1 = _num_vertices_graph1;
    m_num_edges_graph1    = _num_edges_graph1;
    m_num_vertices_graph2 = _num_vertices_graph2;
    m_num_edges_graph2    = _num_edges_graph2;

    auto [_num_vertices_graph1_, _num_communities_graph1] =
        kronecker::read_graph_challenge_truth_file(m_comm, truth_filename1,
                                                   m_truth1);
    kronecker::check_agreement(m_comm, m_num_vertices_graph1,
                               _num_vertices_graph1_, graph_filename1,
                               truth_filename1);
    auto [_num_vertices_graph2_, _num_communities_graph2] =
        kronecker::read_graph_challenge_truth_file(m_comm, truth_filename2,
                                                   m_truth2);
    kronecker::check_agreement(m_comm, m_num_vertices_graph2,
                               _num_vertices_graph2_, graph_filename2,
                               truth_filename2);

    m_num_communities_graph1 = _num_communities_graph1;
    m_num_communities_graph2 = _num_communities_graph2;

    m_vertex_scale =
        (uint64_t)ceil(log2(m_num_vertices_graph1 * m_num_vertices_graph2));
    m_comm.cout0("Vertex Scale: ", m_vertex_scale);
    m_comm.cout0("Graph 1 Vertices: ", m_num_vertices_graph1);
    m_comm.cout0("Graph 1 Edges: ", m_num_edges_graph1);
    m_comm.cout0("Graph 1 Communities: ", m_num_communities_graph1);
    m_comm.cout0("Graph 2 Vertices: ", m_num_vertices_graph2);
    m_comm.cout0("Graph 2 Edges: ", m_num_edges_graph2);
    m_comm.cout0("Graph 2 Communities: ", m_num_communities_graph2);
    m_comm.cout0("sizeof(size_t) = ", sizeof(size_t));
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

  template <typename Function>
  void for_all_truth(Function fn) {
    size_t truth1_pos = m_comm.rank();
    while (truth1_pos < m_truth1.size()) {
      const truth_t& truth1 = m_truth1.at(truth1_pos);

      std::for_each(m_truth2.begin(), m_truth2.end(),
                    [fn, &truth1, this](const truth_t& truth2) {
                      vertex_t         vtx1  = std::get<0>(truth1);
                      community_data_t cmty1 = std::get<1>(truth1);
                      vertex_t         vtx2  = std::get<0>(truth2);
                      edge_data_t      cmty2 = std::get<1>(truth2);

                      vertex_t vtx = vtx1 * this->m_num_vertices_graph2 + vtx2;
                      community_data_t cmty =
                          cmty1 * this->m_num_communities_graph2 + cmty2;

                      fn(vtx, cmty);
                    });
      truth1_pos += m_comm.size();
    }
  }

 private:
  ygm::comm m_comm;

  GS1      m_graph1;
  GS2      m_graph2;
  uint64_t m_num_vertices_graph1;
  uint64_t m_num_vertices_graph2;
  uint64_t m_num_edges_graph1;
  uint64_t m_num_edges_graph2;
  TS1      m_truth1;
  TS2      m_truth2;
  uint32_t m_num_communities_graph1;
  uint32_t m_num_communities_graph2;
  uint64_t m_vertex_scale;
};
