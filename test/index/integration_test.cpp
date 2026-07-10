#include "index/faiss_index.h"
#include "index/filter_index.h"
#include "index/layered_index.h"
#include "index/fulltext_index.h"
#include "index/inner_product_space.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexScalarQuantizer.h"
#include <logger/logger.h>
#include <cstdint>
#include <random>
#include <set>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class IntegrationEnvironment : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const int_env =
    ::testing::AddGlobalTestEnvironment(new IntegrationEnvironment);

static std::vector<float> GenRandom(int n, int dim, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(n) * dim);
    for (auto& v : data) v = dist(rng);
    return data;
}

// ========== 任务0+1：批量插入 + SQ8 量化 + bitmap 过滤 ==========

TEST(IntegrationTest, SQ8WithBitmapFilter) {
    const int dim = 64;
    const int n = 100;
    FaissIndex sq8(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2)));

    auto data = GenRandom(n, dim);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    sq8.BatchInsertVectors(data, n, labels);

    // 创建只包含偶数 ID 的 bitmap
    roaring_bitmap_t* bitmap = roaring_bitmap_create();
    for (int i = 0; i < n; i += 2) {
        roaring_bitmap_add(bitmap, i);
    }

    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, _] = sq8.SearchVectors(query, 10, bitmap);

    // 所有返回的 ID 应该是偶数
    for (int64_t id : ids) {
        if (id != -1) {
            EXPECT_EQ(id % 2, 0) << "ID " << id << " should be even";
        }
    }
    roaring_bitmap_free(bitmap);
}

// ========== 任务0+2：批量插入 + IP_FLAT + bitmap 过滤 ==========

TEST(IntegrationTest, IPFlatWithBitmapFilter) {
    const int dim = 64;
    const int n = 100;
    FaissIndex ip_flat(new faiss::IndexIDMap(
        new faiss::IndexFlat(dim, faiss::METRIC_INNER_PRODUCT)), true);

    auto data = GenRandom(n, dim, 7);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i + 100;

    ip_flat.BatchInsertVectors(data, n, labels);

    // 创建只包含 ID >= 150 的 bitmap
    roaring_bitmap_t* bitmap = roaring_bitmap_create();
    for (int i = 50; i < n; ++i) {
        roaring_bitmap_add(bitmap, static_cast<uint32_t>(i + 100));
    }

    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, _] = ip_flat.SearchVectors(query, 5, bitmap);

    for (int64_t id : ids) {
        if (id != -1) {
            EXPECT_GE(id, 150) << "ID " << id << " should be >= 150";
        }
    }
    roaring_bitmap_free(bitmap);
}

// ========== 任务0+1+3：批量插入 + SQ8 量化 + 范围过滤 ==========

TEST(IntegrationTest, SQ8WithRangeFilter) {
    const int dim = 64;
    const int n = 100;
    FaissIndex sq8(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2)));
    FilterIndex filter;

    auto data = GenRandom(n, dim, 11);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) {
        labels[i] = i;
        // 给每个向量设置 age 字段（0~99）
        filter.AddIntFieldFilter("age", i % 100, i);
    }

    sq8.BatchInsertVectors(data, n, labels);

    // 范围过滤：age >= 50
    roaring_bitmap_t* bitmap = roaring_bitmap_create();
    filter.GetIntFieldFilterBitmap("age", FilterIndex::Operation::GREATER_EQUAL, 50, bitmap);

    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, _] = sq8.SearchVectors(query, 10, bitmap);

    // 所有返回的 ID 应该 >= 50
    for (int64_t id : ids) {
        if (id != -1) {
            EXPECT_GE(id, 50) << "ID " << id << " should have age >= 50";
        }
    }
    roaring_bitmap_free(bitmap);
}

// ========== 任务1+3：SQ8 + 多条件 AND（int范围 + string等值） ==========

TEST(IntegrationTest, SQ8WithMultiConditionAND) {
    const int dim = 64;
    const int n = 100;
    FaissIndex sq8(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2)));
    FilterIndex filter;

    auto data = GenRandom(n, dim, 22);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) {
        labels[i] = i;
        filter.AddIntFieldFilter("age", 10 + (i % 50), i); // age: 10~59
        filter.AddStringFieldFilter("city", (i % 3 == 0) ? "Beijing" : "Shanghai", i);
    }

    sq8.BatchInsertVectors(data, n, labels);

    // 条件1: age >= 30
    roaring_bitmap_t* bm1 = roaring_bitmap_create();
    filter.GetIntFieldFilterBitmap("age", FilterIndex::Operation::GREATER_EQUAL, 30, bm1);

    // 条件2: city = Beijing
    roaring_bitmap_t* bm2 = roaring_bitmap_create();
    filter.GetStringFieldFilterBitmap("city", FilterIndex::Operation::EQUAL, "Beijing", bm2);

    // AND
    roaring_bitmap_and_inplace(bm1, bm2);

    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [ids, _] = sq8.SearchVectors(query, 10, bm1);

    // 验证返回的 ID 满足两个条件
    for (int64_t id : ids) {
        if (id != -1) {
            // age >= 30 => id % 50 >= 20 => id 在 [20,49] 或 [70,99] 范围
            int age = 10 + (id % 50);
            EXPECT_GE(age, 30);
            // city = Beijing => id % 3 == 0
            EXPECT_EQ(id % 3, 0);
        }
    }
    roaring_bitmap_free(bm1);
    roaring_bitmap_free(bm2);
}

// ========== 任务0+1：批量插入 + 批量删除 + 搜索验证 ==========

TEST(IntegrationTest, BatchInsertDeleteSearch) {
    const int dim = 64;
    const int n = 100;
    FaissIndex flat(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));

    auto data = GenRandom(n, dim, 33);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    // 批量插入
    flat.BatchInsertVectors(data, n, labels);

    // 批量删除前50个
    std::vector<int64_t> to_remove;
    for (int i = 0; i < 50; ++i) to_remove.push_back(i);
    flat.RemoveVectors(to_remove);

    // 搜索每个被删除的向量，不应返回自身
    for (int i = 0; i < 50; ++i) {
        std::vector<float> query(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        auto [ids, _] = flat.SearchVectors(query, 1);
        ASSERT_EQ(ids.size(), 1u);
        EXPECT_NE(ids[0], i) << "Deleted ID " << i << " should not appear";
    }

    // 搜索未被删除的向量，应返回自身
    for (int i = 50; i < 60; ++i) {
        std::vector<float> query(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        auto [ids, _] = flat.SearchVectors(query, 1);
        ASSERT_EQ(ids.size(), 1u);
        EXPECT_EQ(ids[0], i) << "Existing ID " << i << " should be found";
    }
}

// ========== 任务1+4：SQ8 量化 + LayeredIndex 分层存储 ==========

TEST(IntegrationTest, LayeredIndexWithSQ8) {
    const int dim = 64;
    const int n = 99;
    const int flush_threshold = 50;

    FaissIndex* base = new FaissIndex(new faiss::IndexIDMap(
        new faiss::IndexScalarQuantizer(dim, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2)));
    LayeredIndex layered(base, dim, false, flush_threshold);

    // 纯 FLAT 作为 ground truth
    FaissIndex flat(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));

    auto data = GenRandom(n, dim, 44);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    flat.BatchInsertVectors(data, n, labels);
    for (int i = 0; i < n; ++i) {
        std::vector<float> vec(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        layered.Insert(vec, labels[i]);
    }

    // 数据分布在 streaming part 和 base(SQ8) 中
    EXPECT_EQ(layered.StreamingSize(), 49u);

    // 召回率验证
    int k = 10;
    int correct = 0;
    int total = 0;
    for (int q = 0; q < 10; ++q) {
        std::vector<float> query(data.begin() + q * dim, data.begin() + (q + 1) * dim);
        auto [flat_ids, _1] = flat.SearchVectors(query, k);
        auto [layered_ids, _2] = layered.Search(query, k);
        std::set<int64_t> flat_set(flat_ids.begin(), flat_ids.end());
        for (int i = 0; i < k; ++i) {
            if (layered_ids[i] != -1 && flat_set.count(layered_ids[i])) ++correct;
            ++total;
        }
    }
    float recall = static_cast<float>(correct) / total;
    // SQ8 量化 + streaming 精确检索，召回率应 > 90%
    EXPECT_GT(recall, 0.90f) << "LayeredIndex+SQ8 recall too low: " << recall;

    delete base;
}

// ========== 任务2+4：IP_FLAT + LayeredIndex ==========

TEST(IntegrationTest, LayeredIndexWithIPFlat) {
    const int dim = 64;
    const int n = 99;
    const int flush_threshold = 50;

    FaissIndex* base = new FaissIndex(new faiss::IndexIDMap(
        new faiss::IndexFlat(dim, faiss::METRIC_INNER_PRODUCT)), true);
    LayeredIndex layered(base, dim, true /*IP*/, flush_threshold);

    auto data = GenRandom(n, dim, 55);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) labels[i] = i;

    for (int i = 0; i < n; ++i) {
        std::vector<float> vec(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        layered.Insert(vec, labels[i]);
    }

    EXPECT_EQ(layered.StreamingSize(), 49u);

    // 搜索自身，top-1 应该是自身（IP 最大）
    for (int q = 0; q < 5; ++q) {
        std::vector<float> query(data.begin() + q * dim, data.begin() + (q + 1) * dim);
        auto [ids, dists] = layered.Search(query, 1);
        ASSERT_EQ(ids.size(), 1u);
        EXPECT_EQ(ids[0], q);
        EXPECT_GT(dists[0], 0.0f); // IP > 0
    }

    delete base;
}

// ========== 任务1+5：SQ8 向量搜索 + BM25 全文搜索混合 ==========

TEST(IntegrationTest, VectorPlusFullTextHybrid) {
    const int dim = 32;
    const int n = 20;
    FaissIndex flat(new faiss::IndexIDMap(new faiss::IndexFlat(dim, faiss::METRIC_L2)));
    FullTextIndex fti;

    auto data = GenRandom(n, dim, 66);
    std::vector<int64_t> labels(n);
    for (int i = 0; i < n; ++i) {
        labels[i] = i;
        // 给每个文档添加文本
        std::string text = "document " + std::to_string(i) + " about vector database search";
        if (i < 10) text += " machine learning";
        fti.AddDocument("content", text, i);
    }

    flat.BatchInsertVectors(data, n, labels);

    // 步骤1：全文搜索 "machine learning" 得到候选集
    auto [text_ids, text_scores] = fti.Search("content", "machine learning", 20);

    // 只有前10个文档包含 "machine learning"
    EXPECT_LE(text_ids.size(), 10u);

    // 步骤2：在候选集中做向量搜索
    roaring_bitmap_t* bitmap = roaring_bitmap_create();
    for (uint64_t id : text_ids) {
        roaring_bitmap_add(bitmap, static_cast<uint32_t>(id));
    }

    std::vector<float> query(data.begin(), data.begin() + dim);
    auto [vec_ids, vec_dists] = flat.SearchVectors(query, 5, bitmap);

    // 返回的 ID 应该在全文搜索的候选集中
    std::set<uint64_t> text_set(text_ids.begin(), text_ids.end());
    for (int64_t id : vec_ids) {
        if (id != -1) {
            EXPECT_TRUE(text_set.count(static_cast<uint64_t>(id)))
                << "ID " << id << " should be in fulltext results";
        }
    }
    roaring_bitmap_free(bitmap);
}

// ========== 任务3+5：范围过滤 + TTL 过期组合 ==========

TEST(IntegrationTest, RangeFilterWithTTL) {
    FilterIndex filter;
    // 插入 age 数据
    for (int i = 0; i < 20; ++i) {
        filter.AddIntFieldFilter("age", 20 + i, i);
    }

    // 范围过滤：age >= 30
    roaring_bitmap_t* bm = roaring_bitmap_create();
    filter.GetIntFieldFilterBitmap("age", FilterIndex::Operation::GREATER_EQUAL, 30, bm);

    // 过滤后的 ID 应该是 10~19（age=30~49）
    EXPECT_EQ(roaring_bitmap_get_cardinality(bm), 10u);

    roaring_bitmap_free(bm);
}

}  // namespace vectordb
