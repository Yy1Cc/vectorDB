#include "index/garden_index.h"
#include "index/hnswlib_index.h"
#include "roaring/roaring.h"
#include "common/vector_init.h"
#include <gtest/gtest.h>
#include <random>
#include <map>
#include <string>
#include <vector>
#include <chrono>

namespace vectordb {

// 初始化 logger / config（GardenIndex 内部用到 global_logger）
class GardenTestEnv : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const garden_env =
    ::testing::AddGlobalTestEnvironment(new GardenTestEnv);

static std::vector<float> GenVec(int dim, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// 辅助：创建 bitmap
static roaring_bitmap_t* MakeBitmap(const std::vector<uint32_t>& ids) {
    roaring_bitmap_t* bm = roaring_bitmap_create();
    for (auto id : ids) roaring_bitmap_add(bm, id);
    return bm;
}

// ==================== 模块1：多子图 ====================

// 测试1：离散子图构建
TEST(GardenIndex, DiscreteSubgraphBuild) {
    GardenIndex garden(8, 10000);
    garden.RegisterDiscreteField("brand");

    for (int i = 0; i < 30; ++i) {
        auto vec = GenVec(8, i);
        std::string brand = (i < 10) ? "apple" : (i < 20) ? "xiaomi" : "huawei";
        garden.Insert(vec, i, {}, {{"brand", brand}});
    }

    EXPECT_EQ(garden.GetTotalCount(), 30);
    EXPECT_TRUE(garden.HasDiscreteSubgraph("brand"));
    auto values = garden.GetDiscreteValues("brand");
    EXPECT_EQ(values.size(), 3);
}

// 测试2：离散子图搜索隔离
TEST(GardenIndex, DiscreteSubgraphSearchIsolation) {
    GardenIndex garden(8, 10000);
    garden.RegisterDiscreteField("brand");

    // apple 向量集中在 [0.5, 0.5, ...]，xiaomi 集中在 [2.5, 2.5, ...]
    for (int i = 0; i < 20; ++i) {
        std::vector<float> vec(8, 0.5f + i * 0.001f);
        garden.Insert(vec, i, {}, {{"brand", "apple"}});
    }
    for (int i = 0; i < 20; ++i) {
        std::vector<float> vec(8, 2.5f + i * 0.001f);
        garden.Insert(vec, 100 + i, {}, {{"brand", "xiaomi"}});
    }

    // 搜索 apple 区域，指定 brand=apple
    std::vector<float> query(8, 0.5f);
    auto* apple_bm = MakeBitmap({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19});

    GardenFilter gf;
    gf.field = "brand";
    gf.discrete_values = {"apple"};
    gf.bitmap = apple_bm;
    gf.cardinality = 20;

    auto [ids, dists] = garden.Search(query, 5, {gf});

    // 验证返回的全是 apple 的 ID (0-19)
    int valid = 0;
    for (int i = 0; i < 5; ++i) {
        if (ids[i] >= 0) {
            valid++;
            EXPECT_LT(ids[i], 20) << "ID " << ids[i] << " should be apple (0-19)";
        }
    }
    EXPECT_GE(valid, 1);

    roaring_bitmap_free(apple_bm);
}

// 测试3：连续分桶构建
TEST(GardenIndex, ContinuousBucketBuild) {
    GardenIndex garden(8, 10000);
    garden.RegisterContinuousField("price", 0, 1000, 100);  // 10 个桶

    for (int i = 0; i < 50; ++i) {
        auto vec = GenVec(8, i);
        int64_t price = i * 20;  // 0, 20, 40, ..., 980
        garden.Insert(vec, i, {{"price", price}}, {});
    }

    EXPECT_EQ(garden.GetTotalCount(), 50);
    EXPECT_TRUE(garden.HasContinuousSubgraph("price"));
}

// 测试4：连续分桶搜索
TEST(GardenIndex, ContinuousBucketSearch) {
    GardenIndex garden(8, 10000);
    garden.RegisterContinuousField("price", 0, 1000, 100);

    // price 0-99 的向量在 [0.1, ...] 附近，price 200-299 的在 [0.9, ...]
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(8, 0.1f + i * 0.001f);
        garden.Insert(vec, i, {{"price", static_cast<int64_t>(i * 10)}}, {});
    }
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec(8, 0.9f + i * 0.001f);
        garden.Insert(vec, 100 + i, {{"price", static_cast<int64_t>(200 + i * 10)}}, {});
    }

    // 搜索 price 0-99 范围
    auto* range_bm = MakeBitmap({0,1,2,3,4,5,6,7,8,9});

    GardenFilter gf;
    gf.field = "price";
    gf.has_range = true;
    gf.range_min = 0;
    gf.range_max = 99;
    gf.bitmap = range_bm;
    gf.cardinality = 10;

    std::vector<float> query(8, 0.1f);
    auto [ids, dists] = garden.Search(query, 3, {gf});

    // 验证返回的 ID 在 0-9 范围
    for (int i = 0; i < 3; ++i) {
        if (ids[i] >= 0) {
            EXPECT_LT(ids[i], 10) << "ID " << ids[i] << " should be in price 0-99";
        }
    }

    roaring_bitmap_free(range_bm);
}

// 测试5：自动注册离散字段
TEST(GardenIndex, AutoRegisterDiscreteField) {
    GardenIndex garden(8, 10000);

    EXPECT_FALSE(garden.HasDiscreteSubgraph("color"));

    auto vec = GenVec(8, 1);
    garden.Insert(vec, 1, {}, {{"color", "red"}});

    EXPECT_TRUE(garden.HasDiscreteSubgraph("color"));
    auto colors = garden.GetDiscreteValues("color");
    EXPECT_EQ(colors.size(), 1);
}

// ==================== 模块2：多策略路由 ====================

// 测试6：暴力检索路由（候选集 < kBruteBound）
TEST(GardenIndex, BruteForceRoute) {
    GardenIndex garden(8, 10000);

    for (int i = 0; i < 100; ++i) {
        auto vec = GenVec(8, i);
        garden.Insert(vec, i, {}, {});
    }

    // 小候选集 → 暴力检索
    std::vector<uint32_t> cand_ids;
    for (int i = 0; i < 50; ++i) cand_ids.push_back(i);
    auto* small_bm = MakeBitmap(cand_ids);

    GardenFilter gf;
    gf.field = "test";
    gf.bitmap = small_bm;
    gf.cardinality = 50;

    auto query = GenVec(8, 10);
    auto [ids, dists] = garden.Search(query, 5, {gf});

    int valid = 0;
    for (int i = 0; i < 5; ++i) {
        if (ids[i] >= 0) {
            valid++;
            // 暴力检索的 ID 应该在候选集内
            bool in_set = false;
            for (auto cid : cand_ids) {
                if (ids[i] == static_cast<int64_t>(cid)) { in_set = true; break; }
            }
            EXPECT_TRUE(in_set) << "ID " << ids[i] << " should be in candidate set";
        }
    }
    EXPECT_GE(valid, 1);

    roaring_bitmap_free(small_bm);
}

// 测试7：无过滤走全量图
TEST(GardenIndex, NoFilterFullGraph) {
    GardenIndex garden(8, 10000);

    for (int i = 0; i < 50; ++i) {
        auto vec = GenVec(8, i);
        garden.Insert(vec, i, {}, {});
    }

    auto query = GenVec(8, 5);
    auto [ids, dists] = garden.Search(query, 5);

    int valid = 0;
    for (int i = 0; i < 5; ++i) {
        if (ids[i] >= 0) valid++;
    }
    EXPECT_GE(valid, 1);
}

// 测试8：大数据集触发子图搜索路径（candidate > kBruteBound）
TEST(GardenIndex, LargeScaleSubgraphRoute) {
    // 12000 apple + 12000 xiaomi = 24000，dim=4 加速构建
    GardenIndex garden(4, 30000);
    garden.RegisterDiscreteField("brand");

    for (int i = 0; i < 12000; ++i) {
        std::vector<float> vec(4, 0.1f + i * 0.00001f);
        garden.Insert(vec, i, {}, {{"brand", "apple"}});
    }
    for (int i = 0; i < 12000; ++i) {
        std::vector<float> vec(4, 0.9f + i * 0.00001f);
        garden.Insert(vec, 20000 + i, {}, {{"brand", "xiaomi"}});
    }

    EXPECT_EQ(garden.GetTotalCount(), 24000);

    // brand=apple 候选集 12000 > kBruteBound(10000)，selectivity=50% < kFullSearchRate(80%)
    // → 走子图搜索路径
    std::vector<uint32_t> apple_ids;
    for (int i = 0; i < 12000; ++i) apple_ids.push_back(i);
    auto* apple_bm = MakeBitmap(apple_ids);

    GardenFilter gf;
    gf.field = "brand";
    gf.discrete_values = {"apple"};
    gf.bitmap = apple_bm;
    gf.cardinality = 12000;

    std::vector<float> query(4, 0.1f);
    auto [ids, dists] = garden.Search(query, 10, {gf});

    // 验证返回的全是 apple 的 ID (0-11999)
    int valid = 0;
    for (int i = 0; i < 10; ++i) {
        if (ids[i] >= 0) {
            valid++;
            EXPECT_LT(ids[i], 12000) << "ID " << ids[i] << " should be apple";
        }
    }
    EXPECT_GE(valid, 1);

    roaring_bitmap_free(apple_bm);
}

// ==================== 模块3：Pruner/Grafter/Selector 分级 ====================

// 测试9：Pruner + Grafter 多条件 AND
TEST(GardenIndex, PrunerGrafterAND) {
    GardenIndex garden(8, 10000);
    garden.RegisterDiscreteField("brand");

    // apple 30 条，其中 category=0 的有 10 条 (id 0,3,6,...,27)
    for (int i = 0; i < 30; ++i) {
        auto vec = GenVec(8, i);
        garden.Insert(vec, i, {{"category", static_cast<int64_t>(i % 3)}}, {{"brand", "apple"}});
    }
    for (int i = 0; i < 30; ++i) {
        auto vec = GenVec(8, 100 + i);
        garden.Insert(vec, 100 + i, {{"category", static_cast<int64_t>(i % 3)}}, {{"brand", "xiaomi"}});
    }

    // 条件1: brand=apple (有子图, cardinality=30) → Pruner
    auto* brand_bm = MakeBitmap({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29});

    // 条件2: category=0 (无子图, cardinality=20) → Grafter
    std::vector<uint32_t> cat_ids;
    for (int i = 0; i < 30; ++i) { if (i % 3 == 0) cat_ids.push_back(i); }
    for (int i = 0; i < 30; ++i) { if (i % 3 == 0) cat_ids.push_back(100 + i); }
    auto* cat_bm = MakeBitmap(cat_ids);

    GardenFilter brand_filter;
    brand_filter.field = "brand";
    brand_filter.discrete_values = {"apple"};
    brand_filter.bitmap = brand_bm;
    brand_filter.cardinality = 30;

    GardenFilter cat_filter;
    cat_filter.field = "category";
    cat_filter.bitmap = cat_bm;
    cat_filter.cardinality = 20;

    auto query = GenVec(8, 1);
    auto [ids, dists] = garden.Search(query, 5, {brand_filter, cat_filter});

    // 验证返回的 ID 都是 apple (0-29) 且 category=0 (id % 3 == 0)
    for (int i = 0; i < 5; ++i) {
        if (ids[i] >= 0) {
            EXPECT_LT(ids[i], 30) << "ID " << ids[i] << " should be apple";
            EXPECT_EQ(ids[i] % 3, 0) << "ID " << ids[i] << " should have category=0";
        }
    }

    roaring_bitmap_free(brand_bm);
    roaring_bitmap_free(cat_bm);
}

// 测试10：Selector 黑名单排除
TEST(GardenIndex, SelectorBlacklist) {
    GardenIndex garden(8, 10000);
    garden.RegisterDiscreteField("brand");

    for (int i = 0; i < 20; ++i) {
        auto vec = GenVec(8, i);
        garden.Insert(vec, i, {}, {{"brand", "apple"}});
    }

    // brand=apple 但排除 id 0,1,2
    auto* brand_bm = MakeBitmap({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19});
    auto* selector_bm = MakeBitmap({0, 1, 2});

    GardenFilter gf;
    gf.field = "brand";
    gf.discrete_values = {"apple"};
    gf.bitmap = brand_bm;
    gf.cardinality = 20;

    auto query = GenVec(8, 0);
    auto [ids, dists] = garden.Search(query, 10, {gf}, selector_bm);

    // 验证 0,1,2 不在结果中
    for (int i = 0; i < 10; ++i) {
        if (ids[i] >= 0) {
            EXPECT_TRUE(ids[i] != 0 && ids[i] != 1 && ids[i] != 2)
                << "ID " << ids[i] << " should be excluded by selector";
        }
    }

    roaring_bitmap_free(brand_bm);
    roaring_bitmap_free(selector_bm);
}

// 测试11：高选择性走全量图 + bitmap（选择性 > kFullSearchRate）
TEST(GardenIndex, HighSelectivityFullGraph) {
    GardenIndex garden(8, 10000);

    for (int i = 0; i < 100; ++i) {
        auto vec = GenVec(8, i);
        garden.Insert(vec, i, {}, {});
    }

    // 候选集 90/100 = 90% > kFullSearchRate(80%) → 全量图 + bitmap
    std::vector<uint32_t> cand_ids;
    for (int i = 0; i < 90; ++i) cand_ids.push_back(i);
    auto* high_bm = MakeBitmap(cand_ids);

    GardenFilter gf;
    gf.field = "test";
    gf.bitmap = high_bm;
    gf.cardinality = 90;

    auto query = GenVec(8, 10);
    auto [ids, dists] = garden.Search(query, 5, {gf});

    int valid = 0;
    for (int i = 0; i < 5; ++i) {
        if (ids[i] >= 0) {
            valid++;
            // ID 应该在候选集内
            bool in_set = (ids[i] < 90);
            EXPECT_TRUE(in_set) << "ID " << ids[i] << " should be in candidate set";
        }
    }
    EXPECT_GE(valid, 1);

    roaring_bitmap_free(high_bm);
}

// 测试12：多离散值 Pruner（brand IN [apple, xiaomi]）
TEST(GardenIndex, MultiValuePruner) {
    GardenIndex garden(8, 10000);
    garden.RegisterDiscreteField("brand");

    // apple 向量在 0.1 附近，xiaomi 在 0.5 附近，huawei 在 0.9 附近
    for (int i = 0; i < 20; ++i) {
        garden.Insert(std::vector<float>(8, 0.1f + i * 0.001f), i, {}, {{"brand", "apple"}});
    }
    for (int i = 0; i < 20; ++i) {
        garden.Insert(std::vector<float>(8, 0.5f + i * 0.001f), 100 + i, {}, {{"brand", "xiaomi"}});
    }
    for (int i = 0; i < 20; ++i) {
        garden.Insert(std::vector<float>(8, 0.9f + i * 0.001f), 200 + i, {}, {{"brand", "huawei"}});
    }

    // brand IN [apple, xiaomi]，排除 huawei
    std::vector<uint32_t> cand_ids;
    for (int i = 0; i < 20; ++i) cand_ids.push_back(i);
    for (int i = 0; i < 20; ++i) cand_ids.push_back(100 + i);
    auto* bm = MakeBitmap(cand_ids);

    GardenFilter gf;
    gf.field = "brand";
    gf.discrete_values = {"apple", "xiaomi"};
    gf.bitmap = bm;
    gf.cardinality = 40;

    // 搜索 apple 区域
    std::vector<float> query(8, 0.1f);
    auto [ids, dists] = garden.Search(query, 5, {gf});

    // 验证返回的不含 huawei (200+)
    for (int i = 0; i < 5; ++i) {
        if (ids[i] >= 0) {
            EXPECT_LT(ids[i], 200) << "ID " << ids[i] << " should not be huawei";
        }
    }

    roaring_bitmap_free(bm);
}

// ==================== 模块4：RemoveVectors 删除 ====================

// 测试13：删除离散子图中的向量，全量图与子图均不再返回
TEST(GardenIndex, RemoveVectorsFromDiscrete) {
    GardenIndex garden(8, 10000);
    garden.RegisterDiscreteField("brand");

    for (int i = 0; i < 30; ++i) {
        garden.Insert(std::vector<float>(8, 0.5f + i * 0.001f), i, {}, {{"brand", "apple"}});
    }
    ASSERT_EQ(garden.GetTotalCount(), 30);

    garden.RemoveVectors({5});
    EXPECT_EQ(garden.GetTotalCount(), 29);

    // 无过滤全量图搜索：id 5 不应出现
    std::vector<float> query(8, 0.5f);
    auto [ids1, dists1] = garden.Search(query, 30);
    for (int i = 0; i < 30; ++i) {
        if (ids1[i] >= 0) {
            EXPECT_NE(ids1[i], 5) << "deleted id 5 should not appear in full graph search";
        }
    }

    // 离散子图搜索（bitmap 含 0-29，含已删除的 5）：5 不应出现
    std::vector<uint32_t> all_apple;
    for (int i = 0; i < 30; ++i) all_apple.push_back(i);
    auto* apple_bm = MakeBitmap(all_apple);
    GardenFilter gf;
    gf.field = "brand";
    gf.discrete_values = {"apple"};
    gf.bitmap = apple_bm;
    gf.cardinality = 30;
    auto [ids2, dists2] = garden.Search(query, 30, {gf});
    for (int i = 0; i < 30; ++i) {
        if (ids2[i] >= 0) {
            EXPECT_NE(ids2[i], 5) << "deleted id 5 should not appear in subgraph search";
        }
    }
    roaring_bitmap_free(apple_bm);
}

// 测试14：删除后重新插入同 id（模拟 upsert 覆盖）不应产生重复
TEST(GardenIndex, RemoveThenReinsertNoDuplicate) {
    GardenIndex garden(8, 10000);
    garden.RegisterDiscreteField("brand");

    garden.Insert(std::vector<float>(8, 0.5f), 1, {}, {{"brand", "apple"}});
    // upsert 语义：先删旧 id 1，再插新向量
    garden.RemoveVectors({1});
    garden.Insert(std::vector<float>(8, 0.9f), 1, {}, {{"brand", "apple"}});

    EXPECT_EQ(garden.GetTotalCount(), 1);

    std::vector<float> q(8, 0.9f);
    auto [ids, dists] = garden.Search(q, 5);
    int count_id1 = 0;
    for (int i = 0; i < 5; ++i) {
        if (ids[i] == 1) count_id1++;
    }
    EXPECT_EQ(count_id1, 1) << "id 1 should appear exactly once after remove+reinsert";
}

// 测试15：删除连续分桶子图中的向量
TEST(GardenIndex, RemoveVectorsFromContinuous) {
    GardenIndex garden(8, 10000);
    garden.RegisterContinuousField("price", 0, 1000, 100);

    for (int i = 0; i < 20; ++i) {
        garden.Insert(std::vector<float>(8, 0.1f + i * 0.001f), i,
                      {{"price", static_cast<int64_t>(i * 10)}}, {});
    }
    ASSERT_EQ(garden.GetTotalCount(), 20);

    garden.RemoveVectors({3});  // price=30，桶0
    EXPECT_EQ(garden.GetTotalCount(), 19);

    std::vector<float> query(8, 0.1f);
    auto [ids, dists] = garden.Search(query, 20);
    for (int i = 0; i < 20; ++i) {
        if (ids[i] >= 0) {
            EXPECT_NE(ids[i], 3) << "deleted id 3 should not appear";
        }
    }
}

// 测试16：Save/Load 后反向映射持久化，删除仍生效
TEST(GardenIndex, SaveLoadPreservesReverseMap) {
    GardenIndex garden(8, 10000);
    garden.RegisterDiscreteField("brand");

    for (int i = 0; i < 20; ++i) {
        garden.Insert(std::vector<float>(8, 0.5f + i * 0.001f), i, {}, {{"brand", "apple"}});
    }

    std::string path = "/tmp/garden_saveload_test";
    garden.SaveIndex(path);

    GardenIndex garden2(8, 10000);
    garden2.RegisterDiscreteField("brand");
    garden2.LoadIndex(path);

    EXPECT_EQ(garden2.GetTotalCount(), 20);

    // 加载后删除 id 7（依赖反向映射已持久化恢复，全量图 mark delete 生效）
    garden2.RemoveVectors({7});
    EXPECT_EQ(garden2.GetTotalCount(), 19);

    std::vector<float> query(8, 0.5f);
    auto [ids, dists] = garden2.Search(query, 20);
    for (int i = 0; i < 20; ++i) {
        if (ids[i] >= 0) {
            EXPECT_NE(ids[i], 7) << "deleted id 7 should not appear after load+remove";
        }
    }
}

// 测试17：Save/Load 自洽重建子图结构（不依赖外部 Register）
// 验证 FIELDS 段持久化 + LoadIndex 重建 discrete/continuous 字段 + value 反推 + count 恢复
TEST(GardenIndex, SaveLoadRebuildsFieldsAndCounts) {
    GardenIndex garden(8, 10000);
    garden.RegisterDiscreteField("brand");
    garden.RegisterContinuousField("price", 0, 1000, 100);

    for (int i = 0; i < 10; ++i) {
        garden.Insert(std::vector<float>(8, 0.1f + i * 0.001f), i,
                      {{"price", static_cast<int64_t>(i * 10)}}, {{"brand", "apple"}});
    }
    for (int i = 0; i < 10; ++i) {
        garden.Insert(std::vector<float>(8, 0.9f + i * 0.001f), 100 + i,
                      {{"price", static_cast<int64_t>(200 + i * 10)}}, {{"brand", "xiaomi"}});
    }
    ASSERT_EQ(garden.GetTotalCount(), 20);

    std::string path = "/tmp/garden_rebuild_test";
    garden.SaveIndex(path);

    // 新实例不调任何 Register，验证 LoadIndex 自洽重建子图结构
    GardenIndex garden2(8, 10000);
    garden2.LoadIndex(path);

    EXPECT_EQ(garden2.GetTotalCount(), 20);
    // 离散字段 + value 集合重建
    EXPECT_TRUE(garden2.HasDiscreteSubgraph("brand"));
    auto values = garden2.GetDiscreteValues("brand");
    EXPECT_EQ(values.size(), 2);  // apple + xiaomi
    // 连续字段重建
    EXPECT_TRUE(garden2.HasContinuousSubgraph("price"));

    // 暴力检索路径（候选集小）验证 id_to_vector + 反向映射加载后仍正确过滤
    std::vector<uint32_t> apple_ids;
    for (int i = 0; i < 10; ++i) apple_ids.push_back(i);
    auto* apple_bm = MakeBitmap(apple_ids);
    GardenFilter gf;
    gf.field = "brand";
    gf.discrete_values = {"apple"};
    gf.bitmap = apple_bm;
    gf.cardinality = 10;
    std::vector<float> query(8, 0.1f);
    auto [ids2, dists2] = garden2.Search(query, 5, {gf});
    int valid = 0;
    for (int i = 0; i < 5; ++i) {
        if (ids2[i] >= 0) {
            valid++;
            EXPECT_LT(ids2[i], 10) << "should be apple after load";
        }
    }
    EXPECT_GE(valid, 1);
    roaring_bitmap_free(apple_bm);
}

}  // namespace vectordb
