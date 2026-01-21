#pragma once
#include <Eigen/Core>
#include <glog/logging.h>

#include <execution>
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>

constexpr int kno_match = std::numeric_limits<int>::lowest();

template <typename T, int N = 3> class KDTree {
public:
  using PointDist = std::pair<T, int>;
  using PointType = Eigen::Matrix<T, N, 1>;
  using PointCloud =
      std::vector<PointType, Eigen::aligned_allocator<PointType>>;
  struct Node {
    Node() = default;
    explicit Node(const size_t nid) : node_id(nid) {}
    int node_id = -1;
    int pt_idx = -1;
    Node *left = nullptr;
    Node *right = nullptr;
    int split_axis = -1;
    T split_val = 0.0;
    bool is_leaf() const { return !left && !right; }
  };
  void setInputCloud(const PointCloud &cloud) {
    reset();
    cloud_ = cloud;
    build();
  }
  void setInputCloud(PointCloud &&cloud) {
    reset();
    cloud_ = std::move(cloud);
    build();
  }

  bool nearest_neighbors(const PointType &query_pt, const size_t k,
                         std::vector<int> &nearest_idx,
                         std::vector<T> &nearest_dist) {
    if (nodes_.empty()) {
      LOG(ERROR) << "Failed to search. No points in the current kd-tree.";
      return false;
    }
    if (nearest_idx.size() != k) {
      nearest_idx = std::vector<int>(k, kno_match);
      nearest_dist = std::vector<T>(k, std::numeric_limits<T>::max());
    }
    std::priority_queue<PointDist> res;
    knn(root(), query_pt, k, res);
    for (int i = res.size() - 1; i >= 0; --i) {
      nearest_idx[i] = res.top().second;
      nearest_dist[i] = res.top().first;
      res.pop();
    }
    return true;
  }

  bool nearest_neighbors_kmt(const PointCloud &query_pc, const size_t k,
                             std::vector<std::vector<int>> &nearest_idx,
                             std::vector<std::vector<T>> &nearest_dist) {
    if (nodes_.empty()) {
      LOG(ERROR) << "Failed to search. No points in the current kd-tree.";
      return false;
    }
    nearest_idx.clear();
    nearest_idx = std::vector<std::vector<int>>(query_pc.size(),
                                                std::vector<int>(k, kno_match));
    nearest_dist.clear();
    nearest_dist = std::vector<std::vector<T>>(
        query_pc.size(), std::vector<T>(k, std::numeric_limits<T>::max()));
    std::vector<int> qids(query_pc.size());
    std::iota(qids.begin(), qids.end(), 0);
    std::for_each(
        std::execution::par_unseq, qids.begin(), qids.end(),
        [this, &query_pc, &k, &nearest_idx, &nearest_dist](const int qid) {
          nearest_neighbors(query_pc[qid], k, nearest_idx[qid],
                            nearest_dist[qid]);
        });
    return true;
  }

  void set_appx(const T app_rate) { appx_rate_ = app_rate; }
  void set_dist(
      std::function<T(const PointType &a, const PointType &b)> &&dist_fun) {
    dist_ = dist_fun;
  }

  Node *root() { return nodes_.empty() ? nullptr : nodes_.front().get(); }

  void print() const {
    for (const auto &node : nodes_) {
      if (node->is_leaf()) {
        LOG(INFO) << "node id: " << node->node_id
                  << ", pt idx: " << node->pt_idx << ", node: type: leaf.";
      } else {
        LOG(INFO) << "node id: " << node->node_id
                  << ", split axis: " << node->split_axis
                  << ", split value: " << node->split_val
                  << ", node type: split.";
      }
    }
  }

  PointType get_point(const size_t id) const { return cloud_[id]; }

  size_t node_size() const { return node_sz_; }
  size_t leaf_size() const { return leaf_sz_; }

private:
  void reset() {
    nodes_.clear();
    cloud_.clear();
    node_sz_ = 0;
    leaf_sz_ = 0;
  }
  bool build() {
    nodes_.emplace_back(std::make_unique<Node>(node_sz_++));
    std::vector<int> idx(cloud_.size());
    std::iota(idx.begin(), idx.end(), 0);
    insert(idx, root());

    return true;
  }

  bool insert(const std::vector<int> &pt_idx, Node *node) {
    if (pt_idx.size() == 1) {
      node->pt_idx = pt_idx.front();
      leaf_sz_++;
      return true;
    }

    std::vector<int> left_idx;
    std::vector<int> right_idx;
    if (!split(pt_idx, left_idx, right_idx, node)) {
      node->pt_idx = left_idx.empty() ? right_idx.front() : left_idx.front();
      leaf_sz_++;
      return true;
    }

    if (!left_idx.empty()) {
      nodes_.emplace_back(std::make_unique<Node>(node_sz_++));
      node->left = nodes_.back().get();
      insert(left_idx, nodes_.back().get());
    }
    if (!right_idx.empty()) {
      nodes_.emplace_back(std::make_unique<Node>(node_sz_++));
      node->right = nodes_.back().get();
      insert(right_idx, nodes_.back().get());
    }
    return true;
  }

  bool split(const std::vector<int> &idx, std::vector<int> &left_idx,
             std::vector<int> &right_idx, Node *node) {
    const PointType mean =
        std::accumulate(
            idx.begin(), idx.end(), PointType::Zero().eval(),
            [this](const PointType &sum, const int pt_idx) -> PointType {
              return sum + cloud_[pt_idx];
            }) /
        idx.size();
    const PointType variance =
        std::accumulate(
            idx.begin(), idx.end(), PointType::Zero().eval(),
            [this, mean](const PointType &sum, const int pt_idx) -> PointType {
              return sum + (cloud_[pt_idx] - mean).cwiseAbs2();
            }) /
        (idx.size() - 1);
    variance.maxCoeff(&node->split_axis);
    node->split_val = mean[node->split_axis];

    for (const auto pt_idx : idx) {
      if (cloud_[pt_idx][node->split_axis] < node->split_val) {
        left_idx.emplace_back(pt_idx);
      } else {
        right_idx.emplace_back(pt_idx);
      }
    }
    if (left_idx.empty() || right_idx.empty()) {
      return false;
    }
    return true;
  }

  void knn(Node *node, const PointType &pt, const size_t k,
           std::priority_queue<PointDist> &res) {
    if (!node) {
      return;
    }
    if (node->is_leaf()) {
      update_res(node, pt, k, res);
      return;
    }
    Node *nx = nullptr;
    Node *candidate = nullptr;
    if (pt[node->split_axis] < node->split_val) {
      nx = node->left;
      candidate = node->right;
    } else {
      nx = node->right;
      candidate = node->left;
    }
    knn(nx, pt, k, res);
    PointType split_pt = pt;
    split_pt[node->split_axis] = node->split_val;
    const T split_dist_sq = dist_(pt, split_pt);
    if (res.size() < k || split_dist_sq < appx_rate_ * current_max_dist(res)) {
      knn(candidate, pt, k, res);
    }
    return;
  }

  void update_res(Node *node, const PointType pt, const size_t k,
                  std::priority_queue<PointDist> &res) {
    const T dist_sq = dist_(pt, cloud_[node->pt_idx]);
    if (res.size() < k) {
      res.emplace(dist_sq, node->pt_idx);
      return;
    }
    if (dist_sq < current_max_dist(res)) {
      res.pop();
      res.emplace(dist_sq, node->pt_idx);
    }
    return;
  }

  T current_max_dist(const std::priority_queue<PointDist> &res) const {
    return res.top().first;
  }

  std::vector<std::unique_ptr<Node>> nodes_;
  PointCloud cloud_;
  size_t node_sz_ = 0;
  size_t leaf_sz_ = 0;
  T appx_rate_ = 1.0;
  std::function<T(const PointType &a, const PointType &b)> dist_ =
      [](const PointType &a, const PointType &b) {
        return (a - b).squaredNorm();
      };
};