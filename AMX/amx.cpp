#include <immintrin.h>
#include <stdio.h>
#include <limits>
#include <stdbool.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include <bit>
#include <sys/syscall.h>
#include <unistd.h>
#include <asm/prctl.h>
#include <sys/prctl.h>

#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

#define FLOAT16_EXPONENT_MASK  0x1fu
#define BFLOAT16_EXPONENT_MASK 0xffu
#define FLOAT32_EXPONENT_MASK  0xffu
#define FLOAT64_EXPONENT_MASK  0x7ffu
#define FLOAT80_EXPONENT_MASK  0x7fffu

#define FLOAT16_INFINITY_EXPONENT  0x1fu
#define BFLOAT16_INFINITY_EXPONENT 0xffu
#define FLOAT32_INFINITY_EXPONENT  0xffu
#define FLOAT64_INFINITY_EXPONENT  0x7ffu
#define FLOAT80_INFINITY_EXPONENT  0x7fffu

#define FLOAT16_NAN_EXPONENT  0x1fu
#define BFLOAT16_NAN_EXPONENT 0xffu
#define FLOAT32_NAN_EXPONENT  0xffu
#define FLOAT64_NAN_EXPONENT  0x7ffu
#define FLOAT80_NAN_EXPONENT  0x7fffu

#define FLOAT16_DENORM_EXPONENT  0x00
#define BFLOAT16_DENORM_EXPONENT 0x00
#define FLOAT32_DENORM_EXPONENT  0x00
#define FLOAT64_DENORM_EXPONENT  0x00
#define FLOAT80_DENORM_EXPONENT  0x00

#define FLOAT16_EXPONENT_BIAS  0x0fu
#define BFLOAT16_EXPONENT_BIAS 0x7fu
#define FLOAT32_EXPONENT_BIAS  0x7fu
#define FLOAT64_EXPONENT_BIAS  0x3ffu
#define FLOAT80_EXPONENT_BIAS  0x3fffu

#define FLOAT16_MANTISSA_MASK  0x3ffu
#define BFLOAT16_MANTISSA_MASK 0x7fu
#define FLOAT32_MANTISSA_MASK  0x7fffffu
#define FLOAT64_MANTISSA_MASK  0xfffffffffffffu
#define FLOAT80_MANTISSA_MASK  0x7fffffffffffffffu

#define FLOAT16_MANTISSA_QUIET_NAN_MASK  0x200u
#define BFLOAT16_MANTISSA_QUIET_NAN_MASK 0x40u
#define FLOAT32_MANTISSA_QUIET_NAN_MASK  0x400000u
#define FLOAT64_MANTISSA_QUIET_NAN_MASK  0x8000000000000u
#define FLOAT80_MANTISSA_QUIET_NAN_MASK  0x4000000000000000u

#define FLOAT16_SIGN_BITS        1
#define FLOAT16_EXPONENT_BITS    5
#define FLOAT16_MANTISSA_BITS    10
#define FLOAT16_QUIET_BITS       1

#define BFLOAT16_SIGN_BITS      1
#define BFLOAT16_EXPONENT_BITS  8
#define BFLOAT16_MANTISSA_BITS  7
#define BFLOAT16_QUIET_BITS     1

#define FLOAT32_SIGN_BITS     1
#define FLOAT32_EXPONENT_BITS 8
#define FLOAT32_MANTISSA_BITS 23
#define FLOAT32_QUIET_BITS    1

#define FLOAT64_SIGN_BITS     1
#define FLOAT64_EXPONENT_BITS 11
#define FLOAT64_MANTISSA_BITS 52
#define FLOAT64_QUIET_BITS    1

#define FLOAT80_SIGN_BITS     1
#define FLOAT80_EXPONENT_BITS 15
#define FLOAT80_JBIT_BITS     1
#define FLOAT80_MANTISSA_BITS 63
#define FLOAT80_QUIET_BITS    1

#define FLOAT16_DECIMAL_DIG        5
#define FLOAT16_DENORM_MIN         5.96046447753906250000000000000000000e-8
#define FLOAT16_DIG                3
#define FLOAT16_EPSILON            9.76562500000000000000000000000000000e-4
#define FLOAT16_HAS_DENORM         1
#define FLOAT16_HAS_INFINITY       1
#define FLOAT16_HAS_QUIET_NAN      1
#define FLOAT16_MANT_DIG           11
#define FLOAT16_MAX_10_EXP         4
#define FLOAT16_MAX                6.55040000000000000000000000000000000e+4
#define FLOAT16_MAX_EXP            16
#define FLOAT16_MIN_10_EXP         (-4)
#define FLOAT16_MIN                6.10351562500000000000000000000000000e-5
#define FLOAT16_MIN_EXP            (-13)
#define FLOAT16_NORM_MAX           6.55040000000000000000000000000000000e+4

#define BFLOAT16_DECIMAL_DIG      3
#define BFLOAT16_DENORM_MIN       (0x1p-133)
#define BFLOAT16_DIG              2
#define BFLOAT16_EPSILON          (FLT_EPSILON * 65536)
#define BFLOAT16_HAS_DENORM       1
#define BFLOAT16_HAS_INFINITY     1
#define BFLOAT16_HAS_QUIET_NAN    1
#define BFLOAT16_MANT_DIG         (FLT_MANT_DIG - 16)
#define BFLOAT16_MAX_10_EXP       FLT_MAX_10_EXP
#define BFLOAT16_MAX_EXP          FLT_MAX_EXP
#define BFLOAT16_MAX              (0x1.fep+127f)
#define BFLOAT16_MIN_10_EXP       FLT_MIN_10_EXP
#define BFLOAT16_MIN_EXP          FLT_MIN_EXP
#define BFLOAT16_MIN              (0x1p-126f)
#define BFLOAT16_NORM_MAX         BFLOAT16_MAX

struct BFloat16
{
    union {
        struct __attribute__((packed)) {
            uint16_t mantissa : BFLOAT16_MANTISSA_BITS;
            uint16_t exponent : BFLOAT16_EXPONENT_BITS;
            uint16_t sign     : BFLOAT16_SIGN_BITS;
        };
        struct __attribute__((packed)) {
            uint16_t payload  : BFLOAT16_MANTISSA_BITS - BFLOAT16_QUIET_BITS;
            uint16_t quiet    : BFLOAT16_QUIET_BITS;
            uint16_t exponent : BFLOAT16_EXPONENT_BITS;
            uint16_t sign     : BFLOAT16_SIGN_BITS;
        } as_nan;
        uint16_t as_hex;
        uint16_t payload;
    };

#ifdef __cplusplus
    inline BFloat16() = default;
    inline BFloat16(float f);
    constexpr inline BFloat16(uint16_t s, uint16_t e, uint16_t m): mantissa(m), exponent(e), sign(s) { }

    // same API as std::numeric_limits:
    static constexpr int digits = BFLOAT16_MANT_DIG;
    static constexpr int digits10 = BFLOAT16_DIG;
    static constexpr int max_digits10 = 3;  // log2(digits)
    static constexpr int min_exponent = std::numeric_limits<float>::min_exponent;
    static constexpr int min_exponent10 = std::numeric_limits<float>::min_exponent10;
    static constexpr int max_exponent = std::numeric_limits<float>::max_exponent;
    static constexpr int max_exponent10 = std::numeric_limits<float>::max_exponent10;

    static constexpr bool radix = 2;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = std::numeric_limits<float>::has_infinity;
    static constexpr bool has_quiet_NaN = std::numeric_limits<float>::has_quiet_NaN;
    static constexpr bool has_signaling_NaN = has_quiet_NaN;
    static constexpr std::float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr std::float_round_style round_style =
            std::round_toward_zero;   // unlike std::numeric_limits<float>::round_style

    static constexpr BFloat16 max()           { return BFloat16(Holder{0x7f7f}); }
    static constexpr BFloat16 min()           { return BFloat16(Holder{0x0080}); }
    static constexpr BFloat16 lowest()        { return BFloat16(Holder{0xff7f}); }
    static constexpr BFloat16 denorm_min()    { return BFloat16(Holder{0x0001}); }
    static constexpr BFloat16 epsilon()       { return BFloat16(Holder{0x3c00}); }
    static constexpr BFloat16 round_error()   { return BFloat16(Holder{0x3f00}); }
    static constexpr BFloat16 infinity()      { return BFloat16(Holder{0x7f80}); }
    static constexpr BFloat16 neg_infinity()  { return BFloat16(Holder{0xff80}); }
    static constexpr BFloat16 quiet_NaN()     { return BFloat16(Holder{0x7fc0}); }
    static constexpr BFloat16 signaling_NaN() { return BFloat16(Holder{0x7fa0}); }

    // extra
    static constexpr float epsilon_v()        { return std::numeric_limits<float>::epsilon() * 65536; }

    constexpr inline bool     is_negative() const       { return sign != 0; }
    constexpr inline bool     is_zero() const           { return (exponent == BFLOAT16_DENORM_EXPONENT) && (mantissa == 0); }
    constexpr inline bool     is_denormal() const       { return (exponent == BFLOAT16_DENORM_EXPONENT) && (mantissa != 0); }
    constexpr inline bool     is_inf() const            { return (exponent == BFLOAT16_INFINITY_EXPONENT) && (mantissa == 0); }

    // NaNs
    constexpr inline bool     is_nan() const            { return  (exponent == BFLOAT16_NAN_EXPONENT) && (mantissa != 0); }
    constexpr inline bool     is_snan() const           { return is_nan() && ((mantissa & BFLOAT16_MANTISSA_QUIET_NAN_MASK) == 0); }
    constexpr inline bool     is_qnan() const           { return is_nan() && ((mantissa & BFLOAT16_MANTISSA_QUIET_NAN_MASK) != 0); }

    constexpr inline uint16_t get_nan_payload() const   { return mantissa & (~BFLOAT16_MANTISSA_QUIET_NAN_MASK); }

private:
    struct Holder { uint16_t payload; };
    explicit constexpr BFloat16(Holder h) : as_hex(h.payload) {}
#endif
};
typedef struct BFloat16 BFloat16;

template <typename T> static inline uint16_t to_bfloat16(T f)
{
    using OutputLimits = BFloat16;
    using InputLimits = std::numeric_limits<T>;
    using UInt = std::conditional_t<sizeof(f) == sizeof(uint32_t), uint32_t, uint64_t>;
    static constexpr int TotalBits = sizeof(f) * 8;
    static constexpr int MantissaBits = InputLimits::digits - 1;    // because of the implicit bit
    static constexpr uint64_t MantissaMask = (uint64_t(1) << MantissaBits) - 1;
    static constexpr int ExponentBits = __builtin_ctz(InputLimits::max_exponent) + 1;
    static constexpr uint64_t ExponentMask = ((uint64_t(1) << ExponentBits) - 1) << MantissaBits;
    static constexpr uint64_t SignMask = uint64_t(1) << (MantissaBits + ExponentBits);
    static_assert(MantissaBits + ExponentBits + 1 == TotalBits);

    UInt v = std::bit_cast<UInt>(f);

    int sign = (v & SignMask) >> (MantissaBits + ExponentBits);
    int exp = (v & ExponentMask) >> MantissaBits;
    int mant = (v & MantissaMask) >> (InputLimits::digits - OutputLimits::digits); // keep only the most significant bits of the mantissa

    // move sign bit to the right bit position
    sign <<= sizeof(BFloat16) * 8 - 1;

    if (exp == 0) {
        // zero or denormal
        return sign;
    } else if (exp == 2 * InputLimits::max_exponent - 1) {
        uint16_t r = v >> 8 * (sizeof(T) - sizeof(BFloat16));
#if defined(__i386__) || defined(__x86_64__)
        /* x86 always quiets any SNaN, so do the same */
        if (mant)
            r |= 1 << (OutputLimits::digits - 2);
#endif
        return r;
    }

    /* normal number */
    int rounding_bias = 0x7fff + (mant & 1);
    v += rounding_bias;
    return v >> 8 * (sizeof(T) - sizeof(BFloat16));
}

BFloat16 tobf16_emulated(float f)
{
    BFloat16 r;
    r.as_hex = to_bfloat16(f);
    return r;
}

inline BFloat16::BFloat16(float f)
    : BFloat16(tobf16_emulated(f))
{
}

struct amx_tileconfig
{
    uint8_t palette;
    uint8_t start_row;
    uint8_t reserved[14];

    // Note: documentation lists 8 tile registers, but the layout of this
    // structure (Table 3-1 in the Instruction Set Extension manual revision
    // 040) adds reserved space that matches exactly what it would look like if
    // there were 16 tile registers (which they could access using the REX R, W
    // or B bits, or VEX.vvvv).
    uint16_t colsb[16];
    uint8_t rows[16];
};

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
{
   printf("\n Failed to enable XFEATURE_XTILEDATA \n\n");
   return false;
}
else
{
   printf("\n TILE DATA USE SET - OK \n\n");
   return true;
}
return true;
}

int main()
{
       // Request permission to linux kernel to run AMX
   if (!set_tiledata_use())
      exit(-1);
    BFloat16 *A = new BFloat16[1048576];
    for(int idx = 0; idx < 1048576; idx++) {
        if (idx % 2 == 0)
            A[idx] = 0.810791015625;
        else
            A[idx] = 2.0;
    }

    BFloat16 *B = new BFloat16[1048576];
    for(int idx = 0; idx < 1048576; idx++) {
        if (idx % 2 == 0)
            B[idx] = 1.0;
        else
            B[idx] = 2.0;
        //B[idx] = BFloat16(frandomf_scale(10.0f));
    }

    BFloat16 *B_new = new BFloat16[1048576];
    //std::fill(std::begin(B_new), std::end(B_new), 2.0);

    for(int idx = 0; idx < 1048576; idx++) {
        B_new[idx] = 2.0;
    }
    //float res_test[1048576] = {0.0};
    float *res_test = new float[1048576];
    //std::fill(std::begin(res_test), std::end(res_test), 0.0);
    for(int idx = 0; idx < 1048576; idx++) {
        res_test[idx] = 0.0;
    }
    int r_org = 1024, c_org = 1024;
    int r_new = 512, c_new = 2048;

    int r_o = 0, c_o = 0;

    for (int r = 0; r < r_new; r++) {
        for (int c = 0; c < c_new; c += 2) {
            B_new[r * c_new + c] = B[(r_o * c_org) + c_o];
            B_new[r * c_new + c + 1] = B[(r_o * c_org) + c_o + c_org];
            c_o++;
        }
        r_o += 2;
        c_o = 0;
    }

    struct amx_tileconfig tc = {};
    tc.palette = 1;
    tc.start_row = 0;
    //int num_rows = 0x1 << (random32() % 5); // one of 1, 2, 4, 8 and 16
    for (int jj = 0; jj < 3; ++jj) {
        tc.rows[jj] = 16;
        tc.colsb[jj] = 64;
    }

    _tile_loadconfig(&tc);

    int TILESIZE_A = 16 * 64;
    int TILESIZE_B = 16 * 64;
    int TILESIZE_C = 16 * 64;

    int A_stride_in_bytes = 2048;
    int B_stride_in_bytes = 4096;
    int C_stride_in_bytes = 4096;

    int T_R = 64, T_K = 32, T_C = 64;

    char *a = (char *)A;
    char *b = (char *)B_new;
    char *c = (char *)res_test;

    for(int t_r = 0; t_r < T_R; t_r++) {
        for(int t_c = 0; t_c < T_C; t_c++) {
            _tile_loadd(0, c + (t_r * T_C * TILESIZE_C) + (t_c * 64) ,C_stride_in_bytes);
            for(int t_k = 0; t_k < T_K; t_k++) {
                _tile_loadd(1, a + (t_r * T_K * TILESIZE_A) + (t_k * 64) ,A_stride_in_bytes);
                _tile_loadd(2, b + (t_k * T_C * TILESIZE_B) + (t_c * 64) ,B_stride_in_bytes);
                _tile_dpbf16ps(0, 1, 2);
            }
            _tile_stored(0, c + (t_r * T_C * TILESIZE_C) + (t_c * 64), C_stride_in_bytes);
        }
    }

    _tile_release();
    delete [] A;
    delete [] B;
    delete [] B_new;
    delete [] res_test;
    return 0;
}