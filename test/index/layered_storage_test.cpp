#include "index/layered_index.h"
#include "index/faiss_index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"
#include <logger/logger.h>
#include <cstdint>
#include <random>
#include <set>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class LayeredStorageEnvironment : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const layered_env =
    ::testing::AddGlobalTestEnvironment(new LayeredStorageEnvironment);

static std::vector<float> GenRandom(int n, int dim, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (auto& v : data) v = dist(rng);
    return data;
}

// ========== StreamingPart 测试 ==========

TEST(StreamingPartTest, InsertAndSearch) {
    const int dim = 64;
    StreamingPart sp(dim, false /*L2*/);

    // 插入 3 个向量
    auto data = GenRandom(3, dim);
    for (int i = 0; i < 3; ++i) {
        std::vector<float> vec(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        sp.Insert(vec, i + 100);
    }

    // 用第 0 个向量搜索，top-1 应该是自身
    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, dists] = sp.Search(query, 3);
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 100); // 自身 L2 距离=0
    EXPECT_NEAR(dists[0], 0.0f, 1e-5f);
    EXPECT_EQ(sp.Size(), 3u);
}

TEST(StreamingPartTest, NeedFlushThreshold) {
    StreamingPart sp(64, false, 5 /*flush_threshold*/);
    EXPECT_FALSE(sp.NeedFlush());

    for (int i = 0; i < 5; ++i) {
        sp.Insert(std::vector<float>(64, 0.1f * i), i);
    }
    EXPECT_TRUE(sp.NeedFlush());
}

// ========== LayeredIndex 测试 ==========

// 基本搜索：数据在 streaming part 中
TEST(LayeredIndexTest, SearchInStreamingPart) {
    const int dim = 64;
    FaissIndex* base = new FaissIndex(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));
    LayeredIndex layered(base, dim, false, 1000 /*大阈值，不 flush*/);

    auto data = GenRandom(5, dim);
    for (int i = 0; i < 5; ++i) {
        std::vector<float> vec(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        layered.Insert(vec, i + 1);
    }

    // 数据都在 streaming part 中
    EXPECT_EQ(layered.StreamingSize(), 5u);

    // 搜索 top-1
    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, dists] = layered.Search(query, 1);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 1); // 自身
    EXPECT_NEAR(dists[0], 0.0f, 1e-5f);

    delete base;
}

// Flush 后数据从 streaming part 转移到 base index
TEST(LayeredIndexTest, FlushTransfersData) {
    const int dim = 64;
    FaissIndex* base = new FaissIndex(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));
    LayeredIndex layered(base, dim, false, 1000);

    auto data = GenRandom(10, dim);
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        layered.Insert(vec, i + 1);
    }
    EXPECT_EQ(layered.StreamingSize(), 10u);

    layered.Flush();
    EXPECT_EQ(layered.StreamingSize(), 0u); // streaming part 已清空

    // 搜索仍然能找到数据（现在在 base index 中）
    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, dists] = layered.Search(query, 1);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 1);
    EXPECT_NEAR(dists[0], 0.0f, 1e-5f);

    delete base;
}

// 自动 flush：达到阈值后自动触发
TEST(LayeredIndexTest, AutoFlush) {
    const int dim = 64;
    const int flush_threshold = 5;
    FaissIndex* base = new FaissIndex(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));
    LayeredIndex layered(base, dim, false, flush_threshold);

    auto data = GenRandom(flush_threshold, dim);
    for (int i = 0; i < flush_threshold; ++i) {
        std::vector<float> vec(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        layered.Insert(vec, i + 1);
    }
    // 插入第 flush_threshold 个时触发自动 flush
    EXPECT_EQ(layered.StreamingSize(), 0u);

    // 搜索仍能找到数据
    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, _] = layered.Search(query, 1);
    EXPECT_EQ(ids[0], 1);

    delete base;
}

// 召回率：LayeredIndex vs 纯 FLAT（数据分布在 streaming + base）
TEST(LayeredIndexTest, RecallVsFlat) {
    const int dim = 64;
    const int n = 99;
    const int k = 10;
    const int flush_threshold = 50;

    // 纯 FLAT 索引（ground truth）
    FaissIndex flat_index(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));

    // LayeredIndex
    FaissIndex* base = new FaissIndex(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));
    LayeredIndex layered(base, dim, false, flush_threshold);

    auto data = GenRandom(n, dim);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    // 批量插入两个索引
    flat_index.BatchInsertVectors(data, n, labels);
    for (int i = 0; i < n; ++i) {
        std::vector<float> vec(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        layered.Insert(vec, labels[i]);
    }

    // 数据分布在 streaming part 和 base index 中
    // flush_threshold=50，前50个会自动 flush 到 base index，后49个在 streaming part
    EXPECT_EQ(layered.StreamingSize(), 49u);

    // 对比召回率
    int correct = 0;
    int total = 0;
    for (int q = 0; q < 10; ++q) {
        std::vector<float> query(data.begin() + q * dim, data.begin() + (q + 1) * dim);
        auto [flat_ids, _1] = flat_index.SearchVectors(query, k);
        auto [layered_ids, _2] = layered.Search(query, k);
        std::set<int64_t> flat_set(flat_ids.begin(), flat_ids.end());
        for (int i = 0; i < k; ++i) {
            if (layered_ids[i] != -1 && flat_set.count(layered_ids[i])) ++correct;
            ++total;
        }
    }
    float recall = static_cast<float>(correct) / total;
    // LayeredIndex 应该有 100% 召回率（暴力检索 + FLAT 都是精确的）
    EXPECT_GT(recall, 0.95f) << "LayeredIndex recall too low: " << recall;

    delete base;
}

// IP metric 的分层索引
TEST(LayeredIndexTest, IPSearch) {
    const int dim = 64;
    FaissIndex* base = new FaissIndex(
        new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_INNER_PRODUCT)), true);
    LayeredIndex layered(base, dim, true /*IP*/, 1000);

    auto data = GenRandom(10, dim);
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        layered.Insert(vec, i + 1);
    }

    // 搜索自身，IP 应该最大（接近1）
    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, dists] = layered.Search(query, 1);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 1); // 自身 IP 最大
    // IP = sum(q_i * v_i)，自身 IP = sum(v_i^2) > 0
    EXPECT_GT(dists[0], 0.0f);

    delete base;
}

}  // namespace vectordb
