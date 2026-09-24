#include "index/hnswlib_index.h"
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include "logger/logger.h"
namespace vectordb {

HNSWLibIndex::HNSWLibIndex(int dim, int num_data, IndexFactory::MetricType metric, int M, int ef_construction):max_elements_(num_data)
{ // 将MetricType参数修改为第三个参数
    // bool normalize = false;
    if (metric == IndexFactory::MetricType::L2) {
        space_ = new hnswlib::L2Space(dim);
    } else {
        throw std::runtime_error("Invalid metric type.");
    }
    index_ = new hnswlib::HierarchicalNSW<float>(space_, num_data, M, ef_construction);
}

void HNSWLibIndex::InsertVectors(const std::vector<float>& data, int64_t label) {
    assert(index_ != nullptr);
    index_->addPoint(data.data(), label);
}

void HNSWLibIndex::BatchInsertVectors(const std::vector<float>& data, int n, const std::vector<int64_t>& labels) {
    assert(index_ != nullptr);
    int dim = space_->get_data_size() / sizeof(float);
    // 如果超出当前容量，扩容
    if (index_->getCurrentElementCount() + n > max_elements_) {
        max_elements_ = index_->getCurrentElementCount() + n;
        index_->resizeIndex(max_elements_);
    }
    for (int i = 0; i < n; ++i) {
        index_->addPoint(data.data() + i * dim, labels[i]);
    }
}

// 找到最多K个 可能不满K个 不满的都是label distance 为-1
auto HNSWLibIndex::SearchVectors(const std::vector<float>& query, int k,const roaring_bitmap_t* bitmap , int ef_search) -> std::pair<std::vector<int64_t>, std::vector<float>> { // 修改返回类型
    assert(index_ != nullptr);
    index_->setEf(ef_search);

    RoaringBitmapIDFilter* selector = nullptr;
    if (bitmap != nullptr) {
        selector = new RoaringBitmapIDFilter(bitmap);
    } 

    auto result = index_->searchKnn(query.data(), k,selector);

    std::vector<int64_t> indices(k,-1);
    std::vector<float> distances(k,-1);
    int j = 0;
    global_logger->debug("Retrieved values:");
    while(!result.empty()){
        auto item = result.top();
        indices[j] = item.second;
        distances[j] = item.first;
        result.pop();
        global_logger->debug("ID: {}, Distance: {}", indices[j], distances[j]);
        j++;
        if(j == k){
            break;
        }
    }
    global_logger->debug("HNSW index found {} vectors",j);

    if (bitmap != nullptr) {
        delete selector;
    } 
    return {indices, distances};
}

void HNSWLibIndex::RemoveVectors(const std::vector<int64_t>& ids) { // 添加RemoveVectors函数实现
    assert(index_ != nullptr);
    for(const auto &id:ids){
        index_->markDelete(id);
    }
}

void HNSWLibIndex::SaveIndex(const std::string& file_path) { // 添加 saveIndex 方法实现
    index_->saveIndex(file_path);
}

void HNSWLibIndex::LoadIndex(const std::string& file_path) { // 添加 loadIndex 方法实现
    std::ifstream file(file_path); // 尝试打开文件
    if (file.good()) { // 检查文件是否存在
        file.close();
        index_->loadIndex(file_path, space_, max_elements_);
    } else {
        global_logger->warn("File not found: {}. Skipping loading index.", file_path);
    }
}

namespace {
// 进程内单调递增，保证并发调用不会复用同一个临时文件名。
auto NextTempSerial() -> uint64_t {
    static std::atomic<uint64_t> serial{0};
    return serial.fetch_add(1);
}

auto MakeTempPath() -> std::filesystem::path {
    const auto base = std::filesystem::temp_directory_path();
    const auto pid = static_cast<unsigned long long>(
#ifdef _WIN32
        _getpid()
#else
        getpid()
#endif
    );
    return base / ("vectordb_hnsw_" + std::to_string(pid) + "_" + std::to_string(NextTempSerial()) + ".tmp");
}
}  // namespace

auto HNSWLibIndex::Serialize() -> std::vector<char> {
    if (index_ == nullptr) {
        return {};
    }
    const auto tmp = MakeTempPath();
    index_->saveIndex(tmp.string());

    std::ifstream in(tmp, std::ios::binary);
    if (!in.is_open()) {
        std::filesystem::remove(tmp);
        throw std::runtime_error("HNSWLibIndex::Serialize: cannot reopen temp file " + tmp.string());
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    std::filesystem::remove(tmp);
    return bytes;
}

void HNSWLibIndex::Deserialize(const char* data, size_t size) {
    if (index_ == nullptr) {
        return;
    }
    const auto tmp = MakeTempPath();
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("HNSWLibIndex::Deserialize: cannot create temp file " + tmp.string());
        }
        if (size > 0) {
            out.write(data, static_cast<std::streamsize>(size));
        }
        out.flush();
    }
    bool ok = false;
    std::string err;
    try {
        index_->loadIndex(tmp.string(), space_, max_elements_);
        ok = true;
    } catch (const std::exception& e) {
        err = e.what();
    }
    std::filesystem::remove(tmp);
    if (!ok) {
        throw std::runtime_error("HNSWLibIndex::Deserialize: loadIndex failed: " + err);
    }
}

auto HNSWLibIndex::GetTotalCount() const -> int64_t {
    if (index_ == nullptr) {
        return 0;
    }
    return static_cast<int64_t>(index_->getCurrentElementCount());
}

}  // namespace vectordb