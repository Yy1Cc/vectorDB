#pragma once

#include <vector>

namespace vectordb {

// InnerProductSpace: ip2cos 向量预处理
// 将向量 L2 归一化后，内积等价于余弦相似度
// 用于 IP_FLAT / IP_SQ8 索引类型，在插入和查询前自动预处理
class InnerProductSpace {
public:
    // 对单个向量做 L2 归一化（原地修改）
    static void Normalize(std::vector<float>& vec);

    // 对 n 个 dim 维向量（连续存储）做 L2 归一化（原地修改）
    static void NormalizeBatch(std::vector<float>& data, int n, int dim);
};

}  // namespace vectordb
