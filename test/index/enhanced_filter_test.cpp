#include "index/filter_index.h"
#include <logger/logger.h>
#include <cstdint>
#include <string>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class EnhancedFilterEnvironment : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const filter_env =
    ::testing::AddGlobalTestEnvironment(new EnhancedFilterEnvironment);

// ========== 范围查询测试 ==========

TEST(EnhancedFilterTest, IntRangeGreaterThan) {
    FilterIndex fi;
    // 插入 age=10(id=1), age=20(id=2), age=30(id=3)
    fi.AddIntFieldFilter("age", 10, 1);
    fi.AddIntFieldFilter("age", 20, 2);
    fi.AddIntFieldFilter("age", 30, 3);

    roaring_bitmap_t* bm = roaring_bitmap_create();
    fi.GetIntFieldFilterBitmap("age", FilterIndex::Operation::GREATER_THAN, 15, bm);

    EXPECT_TRUE(roaring_bitmap_contains(bm, 2));  // age=20 > 15
    EXPECT_TRUE(roaring_bitmap_contains(bm, 3));  // age=30 > 15
    EXPECT_FALSE(roaring_bitmap_contains(bm, 1)); // age=10 < 15
    EXPECT_EQ(roaring_bitmap_get_cardinality(bm), 2u);
    roaring_bitmap_free(bm);
}

TEST(EnhancedFilterTest, IntRangeLessEqual) {
    FilterIndex fi;
    fi.AddIntFieldFilter("age", 10, 1);
    fi.AddIntFieldFilter("age", 20, 2);
    fi.AddIntFieldFilter("age", 30, 3);

    roaring_bitmap_t* bm = roaring_bitmap_create();
    fi.GetIntFieldFilterBitmap("age", FilterIndex::Operation::LESS_EQUAL, 20, bm);

    EXPECT_TRUE(roaring_bitmap_contains(bm, 1));  // age=10 <= 20
    EXPECT_TRUE(roaring_bitmap_contains(bm, 2));  // age=20 <= 20
    EXPECT_FALSE(roaring_bitmap_contains(bm, 3)); // age=30 > 20
    EXPECT_EQ(roaring_bitmap_get_cardinality(bm), 2u);
    roaring_bitmap_free(bm);
}

TEST(EnhancedFilterTest, IntRangeGreaterEqual) {
    FilterIndex fi;
    fi.AddIntFieldFilter("score", 50, 10);
    fi.AddIntFieldFilter("score", 70, 20);
    fi.AddIntFieldFilter("score", 90, 30);

    roaring_bitmap_t* bm = roaring_bitmap_create();
    fi.GetIntFieldFilterBitmap("score", FilterIndex::Operation::GREATER_EQUAL, 70, bm);

    EXPECT_FALSE(roaring_bitmap_contains(bm, 10)); // 50 < 70
    EXPECT_TRUE(roaring_bitmap_contains(bm, 20));  // 70 >= 70
    EXPECT_TRUE(roaring_bitmap_contains(bm, 30));  // 90 >= 70
    roaring_bitmap_free(bm);
}

TEST(EnhancedFilterTest, IntRangeLessThan) {
    FilterIndex fi;
    fi.AddIntFieldFilter("score", 50, 10);
    fi.AddIntFieldFilter("score", 70, 20);
    fi.AddIntFieldFilter("score", 90, 30);

    roaring_bitmap_t* bm = roaring_bitmap_create();
    fi.GetIntFieldFilterBitmap("score", FilterIndex::Operation::LESS_THAN, 70, bm);

    EXPECT_TRUE(roaring_bitmap_contains(bm, 10));  // 50 < 70
    EXPECT_FALSE(roaring_bitmap_contains(bm, 20)); // 70 not < 70
    EXPECT_FALSE(roaring_bitmap_contains(bm, 30)); // 90 > 70
    roaring_bitmap_free(bm);
}

TEST(EnhancedFilterTest, NotEqualCorrectness) {
    FilterIndex fi;
    fi.AddIntFieldFilter("cat", 1, 100);
    fi.AddIntFieldFilter("cat", 2, 200);
    fi.AddIntFieldFilter("cat", 3, 300);

    roaring_bitmap_t* bm = roaring_bitmap_create();
    fi.GetIntFieldFilterBitmap("cat", FilterIndex::Operation::NOT_EQUAL, 2, bm);

    // NOT_EQUAL 2 应包含 cat=1 和 cat=3（修正前 bug 只保留最后一个）
    EXPECT_TRUE(roaring_bitmap_contains(bm, 100));
    EXPECT_TRUE(roaring_bitmap_contains(bm, 300));
    EXPECT_FALSE(roaring_bitmap_contains(bm, 200));
    EXPECT_EQ(roaring_bitmap_get_cardinality(bm), 2u);
    roaring_bitmap_free(bm);
}

// ========== 字符串字段过滤测试 ==========

TEST(EnhancedFilterTest, StringEqual) {
    FilterIndex fi;
    fi.AddStringFieldFilter("city", "Beijing", 1);
    fi.AddStringFieldFilter("city", "Shanghai", 2);
    fi.AddStringFieldFilter("city", "Beijing", 3);

    roaring_bitmap_t* bm = roaring_bitmap_create();
    fi.GetStringFieldFilterBitmap("city", FilterIndex::Operation::EQUAL, "Beijing", bm);

    EXPECT_TRUE(roaring_bitmap_contains(bm, 1));
    EXPECT_TRUE(roaring_bitmap_contains(bm, 3));
    EXPECT_FALSE(roaring_bitmap_contains(bm, 2));
    EXPECT_EQ(roaring_bitmap_get_cardinality(bm), 2u);
    roaring_bitmap_free(bm);
}

TEST(EnhancedFilterTest, StringNotEqual) {
    FilterIndex fi;
    fi.AddStringFieldFilter("city", "Beijing", 1);
    fi.AddStringFieldFilter("city", "Shanghai", 2);
    fi.AddStringFieldFilter("city", "Shenzhen", 3);

    roaring_bitmap_t* bm = roaring_bitmap_create();
    fi.GetStringFieldFilterBitmap("city", FilterIndex::Operation::NOT_EQUAL, "Beijing", bm);

    EXPECT_FALSE(roaring_bitmap_contains(bm, 1));
    EXPECT_TRUE(roaring_bitmap_contains(bm, 2));
    EXPECT_TRUE(roaring_bitmap_contains(bm, 3));
    EXPECT_EQ(roaring_bitmap_get_cardinality(bm), 2u);
    roaring_bitmap_free(bm);
}

TEST(EnhancedFilterTest, StringUpdate) {
    FilterIndex fi;
    fi.AddStringFieldFilter("city", "Beijing", 10);

    // 更新 city 从 Beijing 到 Shanghai
    std::string old_val = "Beijing";
    fi.UpdateStringFieldFilter("city", &old_val, "Shanghai", 10);

    roaring_bitmap_t* bm_bj = roaring_bitmap_create();
    fi.GetStringFieldFilterBitmap("city", FilterIndex::Operation::EQUAL, "Beijing", bm_bj);
    EXPECT_FALSE(roaring_bitmap_contains(bm_bj, 10));
    roaring_bitmap_free(bm_bj);

    roaring_bitmap_t* bm_sh = roaring_bitmap_create();
    fi.GetStringFieldFilterBitmap("city", FilterIndex::Operation::EQUAL, "Shanghai", bm_sh);
    EXPECT_TRUE(roaring_bitmap_contains(bm_sh, 10));
    roaring_bitmap_free(bm_sh);
}

// ========== 多条件 AND 测试 ==========

TEST(EnhancedFilterTest, MultiConditionAND) {
    FilterIndex fi;
    // id=1: age=20, city=Beijing
    // id=2: age=30, city=Beijing
    // id=3: age=30, city=Shanghai
    fi.AddIntFieldFilter("age", 20, 1);
    fi.AddIntFieldFilter("age", 30, 2);
    fi.AddIntFieldFilter("age", 30, 3);
    fi.AddStringFieldFilter("city", "Beijing", 1);
    fi.AddStringFieldFilter("city", "Beijing", 2);
    fi.AddStringFieldFilter("city", "Shanghai", 3);

    // 条件1: age >= 30
    roaring_bitmap_t* bm1 = roaring_bitmap_create();
    fi.GetIntFieldFilterBitmap("age", FilterIndex::Operation::GREATER_EQUAL, 30, bm1);
    // bm1 = {2, 3}

    // 条件2: city = Beijing
    roaring_bitmap_t* bm2 = roaring_bitmap_create();
    fi.GetStringFieldFilterBitmap("city", FilterIndex::Operation::EQUAL, "Beijing", bm2);
    // bm2 = {1, 2}

    // AND
    roaring_bitmap_and_inplace(bm1, bm2);
    // 结果 = {2}

    EXPECT_TRUE(roaring_bitmap_contains(bm1, 2));
    EXPECT_FALSE(roaring_bitmap_contains(bm1, 1));
    EXPECT_FALSE(roaring_bitmap_contains(bm1, 3));
    EXPECT_EQ(roaring_bitmap_get_cardinality(bm1), 1u);

    roaring_bitmap_free(bm1);
    roaring_bitmap_free(bm2);
}

TEST(EnhancedFilterTest, ThreeConditionAND) {
    FilterIndex fi;
    // id=1: age=25, city=Beijing,  cat=1
    // id=2: age=25, city=Beijing,  cat=2
    // id=3: age=35, city=Beijing,  cat=1
    // id=4: age=25, city=Shanghai, cat=1
    for (int id = 1; id <= 4; ++id) {
        // 用 AddIntFieldFilter 和 AddStringFieldFilter 直接建索引
    }
    fi.AddIntFieldFilter("age", 25, 1);
    fi.AddIntFieldFilter("age", 25, 2);
    fi.AddIntFieldFilter("age", 35, 3);
    fi.AddIntFieldFilter("age", 25, 4);
    fi.AddStringFieldFilter("city", "Beijing", 1);
    fi.AddStringFieldFilter("city", "Beijing", 2);
    fi.AddStringFieldFilter("city", "Beijing", 3);
    fi.AddStringFieldFilter("city", "Shanghai", 4);
    fi.AddIntFieldFilter("cat", 1, 1);
    fi.AddIntFieldFilter("cat", 2, 2);
    fi.AddIntFieldFilter("cat", 1, 3);
    fi.AddIntFieldFilter("cat", 1, 4);

    // age = 25 AND city = Beijing AND cat = 1 => 只剩 id=1
    roaring_bitmap_t* bm = roaring_bitmap_create();
    fi.GetIntFieldFilterBitmap("age", FilterIndex::Operation::EQUAL, 25, bm);

    roaring_bitmap_t* bm_city = roaring_bitmap_create();
    fi.GetStringFieldFilterBitmap("city", FilterIndex::Operation::EQUAL, "Beijing", bm_city);
    roaring_bitmap_and_inplace(bm, bm_city);
    roaring_bitmap_free(bm_city);

    roaring_bitmap_t* bm_cat = roaring_bitmap_create();
    fi.GetIntFieldFilterBitmap("cat", FilterIndex::Operation::EQUAL, 1, bm_cat);
    roaring_bitmap_and_inplace(bm, bm_cat);
    roaring_bitmap_free(bm_cat);

    EXPECT_TRUE(roaring_bitmap_contains(bm, 1));
    EXPECT_FALSE(roaring_bitmap_contains(bm, 2)); // cat=2
    EXPECT_FALSE(roaring_bitmap_contains(bm, 3)); // age=35
    EXPECT_FALSE(roaring_bitmap_contains(bm, 4)); // city=Shanghai
    EXPECT_EQ(roaring_bitmap_get_cardinality(bm), 1u);
    roaring_bitmap_free(bm);
}

}  // namespace vectordb
