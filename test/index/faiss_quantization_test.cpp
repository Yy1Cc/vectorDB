#include "index/faiss_index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexScalarQuantizer.h"
#include <logger/logger.h>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

// 全局测试环境：初始化 logger（SearchVectors 内部会调用 global_logger->debug）
class QuantizationEnvironment : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const quant_env =
    ::testing::AddGlobalTestEnvironment(new QuantizationEnvironment);

// 辅助函数：生成 n 个 dim 维随机向量（[-1,1] 均匀分布）
static std::vector<float> GenerateRandomVectors(int n, int dim, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(rng);
    }
    return data;
}

// ========== SQ8 测试 ==========

// SQ8：创建 + lazy train（首次插入自动训练）+ 单条插入 + 搜索
TEST(SQ8QuantizationTest, SingleInsertAndSearch) {
    const int dim = 128;
    FaissIndex sq8_index(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2)));

    auto data = GenerateRandomVectors(1, dim);
    sq8_index.InsertVectors(data, 100);

    auto [ids, dists] = sq8_index.SearchVectors(data, 1);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 100);
}

// SQ8：批量插入 + kNN 召回率（对比 FLAT ground truth）
TEST(SQ8QuantizationTest, BatchInsertAndRecall) {
    const int dim = 128;
    const int n = 200;
    const int k = 10;
    const int num_queries = 10;

    FaissIndex flat_index(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));
    FaissIndex sq8_index(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2)));

    auto data = GenerateRandomVectors(n, dim);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    flat_index.BatchInsertVectors(data, n, labels);
    sq8_index.BatchInsertVectors(data, n, labels);

    int correct = 0;
    int total = 0;
    for (int q = 0; q < num_queries; ++q) {
        std::vector<float> query(data.begin() + q * dim, data.begin() + (q + 1) * dim);
        auto [flat_ids, _1] = flat_index.SearchVectors(query, k);
        auto [sq8_ids, _2] = sq8_index.SearchVectors(query, k);
        std::set<int64_t> flat_set(flat_ids.begin(), flat_ids.end());
        for (int i = 0; i < k; ++i) {
            if (flat_set.count(sq8_ids[i])) ++correct;
            ++total;
        }
    }
    float recall = static_cast<float>(correct) / total;
    EXPECT_GT(recall, 0.95f) << "SQ8 recall too low: " << recall;
}

// ========== SQ4 测试 ==========

// SQ4：批量插入 + kNN 召回率（SQ4 压缩更高，召回率略低但应 >80%）
TEST(SQ4QuantizationTest, BatchInsertAndRecall) {
    const int dim = 128;
    const int n = 200;
    const int k = 10;
    const int num_queries = 10;

    FaissIndex flat_index(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));
    FaissIndex sq4_index(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_4bit, faiss::METRIC_L2)));

    auto data = GenerateRandomVectors(n, dim);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    flat_index.BatchInsertVectors(data, n, labels);
    sq4_index.BatchInsertVectors(data, n, labels);

    int correct = 0;
    int total = 0;
    for (int q = 0; q < num_queries; ++q) {
        std::vector<float> query(data.begin() + q * dim, data.begin() + (q + 1) * dim);
        auto [flat_ids, _1] = flat_index.SearchVectors(query, k);
        auto [sq4_ids, _2] = sq4_index.SearchVectors(query, k);
        std::set<int64_t> flat_set(flat_ids.begin(), flat_ids.end());
        for (int i = 0; i < k; ++i) {
            if (flat_set.count(sq4_ids[i])) ++correct;
            ++total;
        }
    }
    float recall = static_cast<float>(correct) / total;
    EXPECT_GT(recall, 0.80f) << "SQ4 recall too low: " << recall;
}

// ========== 压缩比验证 ==========

// 验证 code_size：FLAT=dim*4, SQ8=dim*1, SQ4=dim/2（向上取整）
TEST(QuantizationTest, CompressionRatio) {
    const int dim = 128;
    faiss::IndexFlat flat(dim, faiss::METRIC_L2);
    faiss::IndexScalarQuantizer sq8(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    faiss::IndexScalarQuantizer sq4(dim, faiss::ScalarQuantizer::QT_4bit, faiss::METRIC_L2);

    // FLAT: 每维 4 字节（float）
    EXPECT_EQ(flat.code_size, static_cast<size_t>(dim) * sizeof(float));
    // SQ8: 每维 1 字节（8bit）
    EXPECT_EQ(sq8.code_size, static_cast<size_t>(dim));
    // SQ4: 每维 0.5 字节（4bit），2 维打包成 1 字节
    EXPECT_EQ(sq4.code_size, static_cast<size_t>((dim + 1) / 2));
}

// ========== 持久化测试 ==========

// SQ8：Save/Load 后搜索结果一致
TEST(SQ8QuantizationTest, SaveAndLoad) {
    const int dim = 64;
    const int n = 50;

    FaissIndex sq8_index(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2)));

    auto data = GenerateRandomVectors(n, dim, 99);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i + 1000;

    sq8_index.BatchInsertVectors(data, n, labels);

    std::string file_path = "/tmp/sq8_test_index.index";
    sq8_index.SaveIndex(file_path);

    FaissIndex loaded_index(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2)));
    loaded_index.LoadIndex(file_path);

    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [orig_ids, _1] = sq8_index.SearchVectors(query, 5);
    auto [loaded_ids, _2] = loaded_index.SearchVectors(query, 5);

    ASSERT_EQ(orig_ids.size(), loaded_ids.size());
    for (size_t i = 0; i < orig_ids.size(); ++i) {
        EXPECT_EQ(orig_ids[i], loaded_ids[i]);
    }

    std::remove(file_path.c_str());
}

// ========== 删除测试 ==========

// SQ8：RemoveVectors 后搜索不应返回已删除的 ID
TEST(SQ8QuantizationTest, RemoveAndSearch) {
    const int dim = 64;
    const int n = 20;

    FaissIndex sq8_index(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2)));

    auto data = GenerateRandomVectors(n, dim, 77);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    sq8_index.BatchInsertVectors(data, n, labels);
    sq8_index.RemoveVectors({0});

    // 用 ID=0 的原始向量做 query，top-1 不应再是 0
    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, _] = sq8_index.SearchVectors(query, 1);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_NE(ids[0], 0);
}

}  // namespace vectordb
