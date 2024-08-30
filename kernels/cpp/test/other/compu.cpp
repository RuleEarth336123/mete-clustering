#include <emmintrin.h>  // SSE2: _mm_set_pd, _mm_sub_pd, _mm_mul_pd, _mm_sqrt_pd, _mm_add_pd

extern "C" {
    struct Point {
        double x, y;
    };

    double distance_sse(const Point& p1, const Point& p2) {
        __m128d a = _mm_set_pd(p1.x, p1.y);
        __m128d b = _mm_set_pd(p2.x, p2.y);
        __m128d result = _mm_sub_pd(a, b);
        result = _mm_mul_pd(result, result);
        result = _mm_sqrt_pd(result);
        // Manual horizontal add for SSE2
        result = _mm_add_pd(result, _mm_shuffle_pd(result, result, 1));
        return ((double*)&result)[0];
    }
}
