#include <immintrin.h>
#include <stddef.h> 
#include <stdint.h> 

size_t despace_avx512(const char *src, char *dst, size_t len) {
    size_t i = 0, j = 0;

    for (; i + 63 < len; i += 64) {
        __m512i data = _mm512_loadu_si512((const __m512i *) (src + i));
        __m512i space = _mm512_set1_epi8(' ');
        __m512i tab = _mm512_set1_epi8('\t');
        __m512i newline = _mm512_set1_epi8('\n');
        uint64_t mask = _mm512_cmpneq_epi8_mask(data, space) & _mm512_cmpneq_epi8_mask(data, tab) & _mm512_cmpneq_epi8_mask(data, newline);
        _mm512_mask_compressstoreu_epi8(dst + j, mask, data);
        j += _mm_popcnt_u64(mask);
    }

    for (; i < len; ++i) {
        dst[j] = src[i];
        j += src[i] != ' ' && src[i] != '\t' && src[i] != '\n';
    }

    return j;
}
