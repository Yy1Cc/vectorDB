#include "index/faiss_index.h"
#include <faiss/IndexIDMap.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include "common/constants.h"
#include "logger/logger.h"
#include <faiss/index_io.h> // 更正头文件
#include <fstream>

namespace vectordb {
FaissIndex::FaissIndex(faiss::Index *index, bool normalize) : index_(index), normalize_(normalize) {}

auto RoaringBitmapIDSelector::is_member(int64_t id) const -> bool {
  return roaring_bitmap_contains(bitmap_, static_cast<uint32_t>(id));
}

void FaissIndex::InsertVectors(const std::vector<float> &data, int64_t label) {
  auto id = static_cast<int64_t>(label);
  std::vector<float> normalized = data;
  if (normalize_) { // ip2cos: L2 归一化后内积等价于余弦相似度
    InnerProductSpace::Normalize(normalized);
  }
  if (!index_->is_trained) { // 量化索引（SQ8/SQ4）需要先训练才能 add
    index_->train(1, normalized.data());
  }
  index_->add_with_ids(1, normalized.data(), &id);
}

void FaissIndex::BatchInsertVectors(const std::vector<float> &data, int n, const std::vector<int64_t> &labels) {
  std::vector<float> normalized = data;
  if (normalize_) { // ip2cos: 批量 L2 归一化
    InnerProductSpace::NormalizeBatch(normalized, n, index_->d);
  }
  if (!index_->is_trained) { // 量化索引（SQ8/SQ4）需要先训练才能 add
    index_->train(n, normalized.data());
  }
  index_->add_with_ids(n, normalized.data(), labels.data());
}

auto FaissIndex::SearchVectors(const std::vector<float> &query, int k, const roaring_bitmap_t *bitmap)
    -> std::pair<std::vector<int64_t>, std::vector<float>> {
  int dim = index_->d;
  std::vector<float> normalized_query = query;
  if (normalize_) { // ip2cos: 查询向量也需 L2 归一化
    int nq = static_cast<int>(normalized_query.size() / dim);
    InnerProductSpace::NormalizeBatch(normalized_query, nq, dim);
  }
  int num_queries = normalized_query.size() / dim;
  std::vector<int64_t> indices(num_queries * k);
  std::vector<float> distances(num_queries * k);

  // 如果传入了 bitmap 参数，则使用 RoaringBitmapIDSelector 初始化 faiss::SearchParameters 对象
  faiss::SearchParameters search_params;
  RoaringBitmapIDSelector selector(bitmap);
  if (bitmap != nullptr) {
    search_params.sel = &selector;
  }

  index_->search(num_queries, normalized_query.data(), k, distances.data(), indices.data(),&search_params);

  global_logger->debug("Retrieved values:");
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] != -1) {
      global_logger->debug("ID: {}, Distance: {}", indices[i], distances[i]);
    } else {
      global_logger->debug("No specific value found");
    }
  }
  return {indices, distances};
}

void FaissIndex::RemoveVectors(const std::vector<int64_t> &ids) {  // 添加remove_vectors函数实现
  auto *id_map = dynamic_cast<faiss::IndexIDMap *>(index_);
  if (index_ != nullptr) {
    // 初始化IDSelectorBatch对象
    faiss::IDSelectorBatch selector(ids.size(), ids.data());
    auto remove_size = id_map->remove_ids(selector);
    global_logger->debug("remove size = {}", remove_size);
  } else {
    throw std::runtime_error("Underlying Faiss index is not an IndexIDMap");
  }
}

void FaissIndex::SaveIndex(const std::string& file_path) { // 添加 saveIndex 方法实现
    faiss::write_index(index_, file_path.c_str());
}

void FaissIndex::LoadIndex(const std::string& file_path) { // 添加 loadIndex 方法实现
    std::ifstream file(file_path); // 尝试打开文件
    if (file.good()) { // 检查文件是否存在
        file.close();
        delete index_;
        index_ = faiss::read_index(file_path.c_str());
    } else {
        global_logger->warn("File not found: {}. Skipping loading index.", file_path);
    }
}

void FaissIndex::Train(int n, const std::vector<float>& data) { // 训练量化索引（SQ8/SQ4）
    index_->train(n, data.data());
}

auto FaissIndex::GetTotalCount() const -> int64_t {
    if (index_ == nullptr) {
        return 0;
    }
    return index_->ntotal;
}

}  // namespace vectordb