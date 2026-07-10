#include "index/fulltext_index.h"
#include "database/ttl_manager.h"
#include <logger/logger.h>
#include <cstdint>
#include <thread>
#include <vector>
#include "common/vector_init.h"
#include "gtest/gtest.h"

namespace vectordb {

class FullTextTTLEnvironment : public ::testing::Environment {
public:
    void SetUp() override { VdbServerInit(1); }
};
::testing::Environment* const ft_env =
    ::testing::AddGlobalTestEnvironment(new FullTextTTLEnvironment);

// ========== BM25 全文检索测试 ==========

TEST(FullTextIndexTest, BasicSearch) {
    FullTextIndex fti;

    fti.AddDocument("content", "the quick brown fox jumps over the lazy dog", 1);
    fti.AddDocument("content", "a quick brown dog runs in the park", 2);
    fti.AddDocument("content", "the lazy cat sleeps on the sofa", 3);

    // 搜索 "quick brown" 应该返回文档 1 和 2
    auto [ids, scores] = fti.Search("content", "quick brown", 3);
    ASSERT_GE(ids.size(), 2u);
    // 文档1 "quick" 出现1次, "brown" 出现1次，文档2 也是
    // 文档1 和 2 都包含 "quick" 和 "brown"
    EXPECT_TRUE(std::find(ids.begin(), ids.end(), 1u) != ids.end());
    EXPECT_TRUE(std::find(ids.begin(), ids.end(), 2u) != ids.end());
    // 文档3 不包含 "quick" 或 "brown"
    EXPECT_FALSE(std::find(ids.begin(), ids.end(), 3u) != ids.end());
}

TEST(FullTextIndexTest, BM25Ranking) {
    FullTextIndex fti;

    // 文档1 包含 "vector" 3次
    fti.AddDocument("content", "vector vector vector database system", 1);
    // 文档2 包含 "vector" 1次
    fti.AddDocument("content", "vector search engine design", 2);
    // 文档3 不包含 "vector"
    fti.AddDocument("content", "machine learning models", 3);

    auto [ids, scores] = fti.Search("content", "vector", 3);
    ASSERT_EQ(ids.size(), 2u); // 只有文档1和2包含"vector"
    // 文档1 的 BM25 分数应高于文档2（因为词频更高）
    EXPECT_EQ(ids[0], 1u);
    EXPECT_GT(scores[0], scores[1]);
}

TEST(FullTextIndexTest, RemoveDocument) {
    FullTextIndex fti;
    fti.AddDocument("content", "hello world search engine", 1);
    fti.AddDocument("content", "hello world again", 2);

    auto [ids1, _1] = fti.Search("content", "hello", 5);
    EXPECT_EQ(ids1.size(), 2u);

    fti.RemoveDocument("content", 1);

    auto [ids2, _2] = fti.Search("content", "hello", 5);
    EXPECT_EQ(ids2.size(), 1u);
    EXPECT_EQ(ids2[0], 2u);
}

TEST(FullTextIndexTest, GetScore) {
    FullTextIndex fti;
    fti.AddDocument("content", "the quick brown fox", 1);
    fti.AddDocument("content", "the slow turtle", 2);

    // 文档1 包含 "quick"
    float score1 = fti.GetScore("content", "quick", 1);
    EXPECT_GT(score1, 0.0f);

    // 文档2 不包含 "quick"
    float score2 = fti.GetScore("content", "quick", 2);
    EXPECT_EQ(score2, 0.0f);
}

TEST(FullTextIndexTest, SaveAndLoad) {
    FullTextIndex fti;
    fti.AddDocument("content", "vector database search engine", 1);
    fti.AddDocument("content", "machine learning vector space", 2);

    std::string path = "/tmp/fulltext_test_index.idx";
    fti.SaveIndex(path);

    FullTextIndex loaded;
    loaded.LoadIndex(path);

    auto [orig_ids, orig_scores] = fti.Search("content", "vector", 5);
    auto [loaded_ids, loaded_scores] = loaded.Search("content", "vector", 5);

    ASSERT_EQ(orig_ids.size(), loaded_ids.size());
    for (size_t i = 0; i < orig_ids.size(); ++i) {
        EXPECT_EQ(orig_ids[i], loaded_ids[i]);
        EXPECT_NEAR(orig_scores[i], loaded_scores[i], 0.001f);
    }

    std::remove(path.c_str());
}

TEST(FullTextIndexTest, MultipleFields) {
    FullTextIndex fti;
    fti.AddDocument("title", "vector database introduction", 1);
    fti.AddDocument("body", "this article introduces vector databases and search", 1);
    fti.AddDocument("title", "machine learning basics", 2);
    fti.AddDocument("body", "machine learning models and training", 2);

    auto [title_ids, _1] = fti.Search("title", "vector", 5);
    EXPECT_EQ(title_ids.size(), 1u);
    EXPECT_EQ(title_ids[0], 1u);

    auto [body_ids, _2] = fti.Search("body", "machine", 5);
    EXPECT_EQ(body_ids.size(), 1u);
    EXPECT_EQ(body_ids[0], 2u);
}

// ========== TTL 过期管理测试 ==========

TEST(TTLManagerTest, SetTTLAndCheckExpired) {
    TTLManager ttl;

    // 设置1秒TTL
    ttl.SetTTL(100, 1);
    ttl.SetTTL(200, 3600); // 1小时后过期

    EXPECT_FALSE(ttl.IsExpired(100)); // 刚设置，未过期
    EXPECT_FALSE(ttl.IsExpired(200));
    EXPECT_FALSE(ttl.IsExpired(999)); // 没有TTL记录

    // 等待2秒让 id=100 过期
    std::this_thread::sleep_for(std::chrono::seconds(2));

    EXPECT_TRUE(ttl.IsExpired(100));  // 已过期
    EXPECT_FALSE(ttl.IsExpired(200)); // 仍未过期
}

TEST(TTLManagerTest, GetExpiredIDs) {
    TTLManager ttl;

    // 设置两个马上过期的和一个不过期的
    ttl.SetExpireAt(1, 0);           // 已过期（时间戳0 = 1970年）
    ttl.SetExpireAt(2, 0);           // 已过期
    ttl.SetExpireAt(3, 9999999999);  // 远未来，不过期

    auto expired = ttl.GetExpiredIDs();
    ASSERT_EQ(expired.size(), 2u);
    // 应该包含 1 和 2
    EXPECT_TRUE(std::find(expired.begin(), expired.end(), 1ull) != expired.end());
    EXPECT_TRUE(std::find(expired.begin(), expired.end(), 2ull) != expired.end());
}

TEST(TTLManagerTest, Remove) {
    TTLManager ttl;
    ttl.SetTTL(42, 3600);
    EXPECT_TRUE(ttl.HasTTL(42));
    EXPECT_EQ(ttl.Size(), 1u);

    ttl.Remove(42);
    EXPECT_FALSE(ttl.HasTTL(42));
    EXPECT_EQ(ttl.Size(), 0u);
    EXPECT_FALSE(ttl.IsExpired(42)); // 没有记录，不算过期
}

TEST(TTLManagerTest, SetExpireAt) {
    TTLManager ttl;

    // 设置绝对过期时间
    ttl.SetExpireAt(100, 9999999999); // 远未来
    EXPECT_FALSE(ttl.IsExpired(100));

    ttl.SetExpireAt(200, 1); // 1970年，已过期
    EXPECT_TRUE(ttl.IsExpired(200));

    EXPECT_EQ(ttl.GetExpireTime(100), 9999999999);
    EXPECT_EQ(ttl.GetExpireTime(200), 1);
    EXPECT_EQ(ttl.GetExpireTime(999), -1); // 不存在
}

TEST(TTLManagerTest, UpdateTTL) {
    TTLManager ttl;
    ttl.SetExpireAt(1, 9999999999); // 远未来
    EXPECT_FALSE(ttl.IsExpired(1));

    // 更新为已过期
    ttl.SetExpireAt(1, 0);
    EXPECT_TRUE(ttl.IsExpired(1));
}

}  // namespace vectordb
