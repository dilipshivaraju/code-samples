#include <immintrin.h>
#include <stdio.h>
#include <limits>
//#include <stdbool.h>
//#include <stdint.h>
#include <float.h>
#include <type_traits>
#include <bit>
#include <sys/syscall.h>
#include <unistd.h>
#include <asm/prctl.h>
#include <sys/prctl.h>
#include <cstring>
#include <iostream>

#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

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


    /* normal number */
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
    printf("here");
       // Request permission to linux kernel to run AMX
   if (!set_tiledata_use())
      exit(-1);
    
    BFloat16 *A = new BFloat16[1048576];
    for(int idx = 0; idx < 1048576; idx++) {
        if (idx % 2 == 0)
            A[idx] = 1.0;
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

    float *res = new float[1048576];
    for(int idx = 0; idx < 1048576; idx++) {
        res[idx] = 0.0;
    }
    //std::fill(std::begin(*res), std::end(*res), 0.0);

    for(int r = 0; r < 1024; r++) {
        for(int c = 0; c < 1024; c++) {
            float acc = 0.0;
            for(int k = 0; k < 1024; k++) {
                acc += frombf16_emulated(A[r * 1024 + k]) * frombf16_emulated(B[k * 1024 + c]);
            }
            res[r * 1024 + c] = acc;
        }
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

    for(int idx = 0; idx < 1024 * 1024; idx++) {
        if (res[idx] != res_test[idx]) {
            printf("not fine\n");
            exit(-1);
        }
    }

    printf("fine\n");

    delete [] A;
    delete [] B;
    delete [] B_new;
    delete [] res;
    delete [] res_test;
    return 0;
}