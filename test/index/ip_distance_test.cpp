#include "index/inner_product_space.h"
#include "index/faiss_index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexScalarQuantizer.h"
#include <logger/logger.h>
#include <cstdint>
#include <cmath>
#include <random>
#include <set>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class IpDistanceEnvironment : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const ip_env =
    ::testing::AddGlobalTestEnvironment(new IpDistanceEnvironment);

static std::vector<float> GenerateRandomVectors(int n, int dim, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (auto& v : data) v = dist(rng);
    return data;
}

// ========== InnerProductSpace 单元测试 ==========

TEST(InnerProductSpaceTest, NormalizeSingleVector) {
    std::vector<float> vec = {3.0f, 4.0f}; // norm = 5
    InnerProductSpace::Normalize(vec);
    ASSERT_EQ(vec.size(), 2u);
    EXPECT_FLOAT_EQ(vec[0], 0.6f);  // 3/5
    EXPECT_FLOAT_EQ(vec[1], 0.8f);  // 4/5
    // 归一化后 L2 norm 应为 1
    float norm = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(InnerProductSpaceTest, NormalizeZeroVector) {
    std::vector<float> vec = {0.0f, 0.0f, 0.0f};
    InnerProductSpace::Normalize(vec);
    // 零向量归一化后应保持不变（不除以0）
    for (float v : vec) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(InnerProductSpaceTest, NormalizeBatch) {
    int dim = 4;
    std::vector<float> data = {
        3.0f, 0.0f, 0.0f, 0.0f,  // norm=3
        0.0f, 0.0f, 5.0f, 0.0f,  // norm=5
    };
    InnerProductSpace::NormalizeBatch(data, 2, dim);
    // 第一个向量归一化后
    EXPECT_NEAR(data[0], 1.0f, 1e-5f);
    // 第二个向量归一化后
    EXPECT_NEAR(data[6], 1.0f, 1e-5f);
}

// ========== IP_FLAT 索引测试 ==========

// ip2cos 核心验证：归一化后内积 = 余弦相似度
TEST(IPFlatTest, CosineSimilarityViaInnerProduct) {
    const int dim = 128;
    FaissIndex ip_index(new faiss::IndexIDMap(
        new faiss::IndexFlat(dim, faiss::METRIC_INNER_PRODUCT)), true);

    // 两个方向不同的向量
    std::vector<float> v1 = GenerateRandomVectors(1, dim, 1);
    std::vector<float> v2 = GenerateRandomVectors(1, dim, 2);

    ip_index.InsertVectors(v1, 1);
    ip_index.InsertVectors(v2, 2);

    // 用 v1 搜索，top-1 应该是 v1 自身（余弦相似度=1）
    auto [ids, dists] = ip_index.SearchVectors(v1, 2);
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 1); // 自身余弦相似度最高
    // distance 应接近 1.0（余弦相似度）
    EXPECT_NEAR(dists[0], 1.0f, 0.01f);
}

// IP_FLAT 批量插入 + 召回率：与手动归一化的 FLAT-L2 对比
TEST(IPFlatTest, BatchInsertAndRecall) {
    const int dim = 128;
    const int n = 200;
    const int k = 10;
    const int num_queries = 10;

    // IP_FLAT（自动归一化）
    FaissIndex ip_index(new faiss::IndexIDMap(
        new faiss::IndexFlat(dim, faiss::METRIC_INNER_PRODUCT)), true);

    // 手动归一化的 FLAT-L2（ground truth：L2 距离最小的就是余弦相似度最大的）
    FaissIndex flat_index(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));

    auto data = GenerateRandomVectors(n, dim);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    // 手动归一化后插入 FLAT-L2
    std::vector<float> normalized_data = data;
    InnerProductSpace::NormalizeBatch(normalized_data, n, dim);
    flat_index.BatchInsertVectors(normalized_data, n, labels);

    // IP_FLAT 自动归一化
    ip_index.BatchInsertVectors(data, n, labels);

    int correct = 0;
    int total = 0;
    for (int q = 0; q < num_queries; ++q) {
        std::vector<float> query(data.begin() + q * dim, data.begin() + (q + 1) * dim);
        auto [ip_ids, _1] = ip_index.SearchVectors(query, k);
        auto [flat_ids, _2] = flat_index.SearchVectors(
            [&]() { std::vector<float> nq = query; InnerProductSpace::Normalize(nq); return nq; }(), k);
        std::set<int64_t> flat_set(flat_ids.begin(), flat_ids.end());
        for (int i = 0; i < k; ++i) {
            if (flat_set.count(ip_ids[i])) ++correct;
            ++total;
        }
    }
    float recall = static_cast<float>(correct) / total;
    // ip2cos 召回率应接近 100%（IP_FLAT 是精确的，无量化损失）
    EXPECT_GT(recall, 0.95f) << "IP_FLAT recall too low: " << recall;
}

// ========== IP_SQ8 索引测试 ==========

TEST(IPSq8Test, BatchInsertAndRecall) {
    const int dim = 128;
    const int n = 200;
    const int k = 10;
    const int num_queries = 10;

    // IP_SQ8（自动归一化 + 8bit 量化）
    FaissIndex ip_sq8_index(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_INNER_PRODUCT)), true);

    // IP_FLAT（ground truth）
    FaissIndex ip_flat_index(new faiss::IndexIDMap(
        new faiss::IndexFlat(dim, faiss::METRIC_INNER_PRODUCT)), true);

    auto data = GenerateRandomVectors(n, dim);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    ip_flat_index.BatchInsertVectors(data, n, labels);
    ip_sq8_index.BatchInsertVectors(data, n, labels);

    int correct = 0;
    int total = 0;
    for (int q = 0; q < num_queries; ++q) {
        std::vector<float> query(data.begin() + q * dim, data.begin() + (q + 1) * dim);
        auto [flat_ids, _1] = ip_flat_index.SearchVectors(query, k);
        auto [sq8_ids, _2] = ip_sq8_index.SearchVectors(query, k);
        std::set<int64_t> flat_set(flat_ids.begin(), flat_ids.end());
        for (int i = 0; i < k; ++i) {
            if (flat_set.count(sq8_ids[i])) ++correct;
            ++total;
        }
    }
    float recall = static_cast<float>(correct) / total;
    EXPECT_GT(recall, 0.90f) << "IP_SQ8 recall too low: " << recall;
}

// ========== Save/Load 持久化 ==========

TEST(IPFlatTest, SaveAndLoad) {
    const int dim = 64;
    const int n = 50;

    FaissIndex ip_index(new faiss::IndexIDMap(
        new faiss::IndexFlat(dim, faiss::METRIC_INNER_PRODUCT)), true);

    auto data = GenerateRandomVectors(n, dim, 99);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i + 500;

    ip_index.BatchInsertVectors(data, n, labels);

    std::string file_path = "/tmp/ip_flat_test_index.index";
    ip_index.SaveIndex(file_path);

    FaissIndex loaded_index(new faiss::IndexIDMap(
        new faiss::IndexFlat(dim, faiss::METRIC_INNER_PRODUCT)), true);
    loaded_index.LoadIndex(file_path);

    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [orig_ids, _1] = ip_index.SearchVectors(query, 5);
    auto [loaded_ids, _2] = loaded_index.SearchVectors(query, 5);

    ASSERT_EQ(orig_ids.size(), loaded_ids.size());
    for (size_t i = 0; i < orig_ids.size(); ++i) {
        EXPECT_EQ(orig_ids[i], loaded_ids[i]);
    }
    std::remove(file_path.c_str());
}

}  // namespace vectordb
