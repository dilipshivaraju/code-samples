#include <immintrin.h>
#include <stdio.h>
#include <limits>
#include <float.h>
#include <type_traits>
#include <bit>
#include <cstring>
#include <iostream>

#define BFLOAT16_MANT_DIG         (FLT_MANT_DIG - 16)

struct BFloat16
{
    uint16_t data;
    inline BFloat16() = default;
    inline BFloat16(float f);
    static constexpr int digits = BFLOAT16_MANT_DIG;
};
typedef struct BFloat16 BFloat16;

static inline uint16_t to_bfloat16(float f)
{
    using Fp32Limits = std::numeric_limits<float>;
    static constexpr int MantissaBits = Fp32Limits::digits - 1;    // because of the implicit bit
    static constexpr uint64_t MantissaMask = (uint64_t(1) << MantissaBits) - 1;
    uint32_t v = std::bit_cast<uint32_t>(f);
    int mant = (v & MantissaMask) >> (Fp32Limits::digits - BFloat16::digits); // keep only the most significant bits of the mantissa


    /* Since random values used are between 0.0 and 2.0, no de-normals or infinitiy or NaNs*/
    int rounding_bias = 0x7fff + (mant & 1);
    v += rounding_bias;
    return v >> 8 * (sizeof(float) - sizeof(BFloat16));
}

BFloat16 tobf16_emulated(float f)
{
    BFloat16 r;
    r.data = to_bfloat16(f);
    return r;
}

inline BFloat16::BFloat16(float f)
    : BFloat16(tobf16_emulated(f))
{
}

static inline float frombf16_emulated(BFloat16 r)
{
    // we zero-extend, shamelessly
    float f;
    uint32_t x = r.data;
    x <<= 16;
    memcpy(&f, &x, sizeof(f));
    return f;
}