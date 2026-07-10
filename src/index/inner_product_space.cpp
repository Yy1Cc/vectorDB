#include "index/inner_product_space.h"
#include <cmath>

namespace vectordb {

void InnerProductSpace::Normalize(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float v : vec) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& v : vec) {
            v /= norm;
        }
    }
}

void InnerProductSpace::NormalizeBatch(std::vector<float>& data, int n, int dim) {
    for (int i = 0; i < n; ++i) {
        float norm = 0.0f;
        int offset = i * dim;
        for (int j = 0; j < dim; ++j) {
            float v = data[offset + j];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (int j = 0; j < dim; ++j) {
                data[offset + j] /= norm;
            }
        }
    }
}

}  // namespace vectordb
