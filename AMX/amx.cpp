#include <bfloat16.h>
#include <amx.h>
#include <memory>

void check_for_correctness()
{
    // matrix multiply 16x32 and 32x16 matrices
    BFloat16 A[512], B[512], transformed_B[512];
    float C_standard[256] = {}, C_amx[256] = {}; // 16x16 matrix

    for(int idx = 0; idx < 16 * 32; idx++) {
        if (idx % 2 == 0) {
            A[idx] = 1.0;
            B[idx] = 1.0;
        }
        else {
            A[idx] = 2.0;
            B[idx] = 2.0;
        }       
    }

    for(int r = 0; r < 16; r++) {
        for(int c = 0; c < 16; c++) {
            float acc = 0.0;
            for(int k = 0; k < 32; k++) {
                acc += frombf16_emulated(A[r * 32 + k]) * frombf16_emulated(B[k * 16 + c]);
            }
            C_standard[r * 16 + c] = acc;
        }
    }

    int r_org = 32, c_org = 16;
    int r_new = 16, c_new = 32;

    int r_o = 0, c_o = 0;

    for(int r = 0; r < r_new; r++) {
        for (int c = 0; c < c_new; c += 2) {
            transformed_B[r * c_new + c] = B[(r_o * c_org) + c_o];
            transformed_B[r * c_new + c + 1] = B[(r_o * c_org) + c_o + c_org];
            c_o++;
        }
        r_o += 2;
        c_o = 0;
    }

    struct amx_tileconfig tc = {};
    tc.palette = 1;
    tc.start_row = 0;
    for (int jj = 0; jj < 3; ++jj) {
        tc.rows[jj] = 16;
        tc.colsb[jj] = 64;
    }
    _tile_loadconfig(&tc);

    int TILESIZE_A_BYTES = 16 * 64;
    int TILESIZE_B_BYTES = 16 * 64;
    int TILESIZE_C_BYTES = 16 * 64;

    int A_stride_in_bytes = 64;
    int B_stride_in_bytes = 64;
    int C_stride_in_bytes = 64;

    int T_R = 1, T_K = 1, T_C = 1;

    char *a = (char *)A;
    char *b = (char *)transformed_B;
    char *c = (char *)C_amx;

    for(int t_r = 0; t_r < T_R; t_r++) {
        for(int t_c = 0; t_c < T_C; t_c++) {
            _tile_loadd(0, c + (t_r * T_C * TILESIZE_C_BYTES) + (t_c * 64) ,C_stride_in_bytes);
            for(int t_k = 0; t_k < T_K; t_k++) {
                _tile_loadd(1, a + (t_r * T_K * TILESIZE_A_BYTES) + (t_k * 64) ,A_stride_in_bytes);
                _tile_loadd(2, b + (t_k * T_C * TILESIZE_B_BYTES) + (t_c * 64) ,B_stride_in_bytes);
                _tile_dpbf16ps(0, 1, 2);
            }
            _tile_stored(0, c + (t_r * T_C * TILESIZE_C_BYTES) + (t_c * 64), C_stride_in_bytes);
        }
    }

    _tile_release();

    for(int idx = 0; idx < 16 * 16; idx++) {
        if (C_standard[idx] != C_amx[idx]) {
            printf("%f %f %d not fine\n", C_standard[idx], C_amx[idx], idx);
            exit(-1);
        }
    }

}

int main()
{

    // check if current processor supports
    if(!check_amx_support()) {
        printf("AMX not supported on this platform\n");
        exit(-1);
    }

    // Request linux kernel's permission to run AMX
    if (!set_tiledata_use())
        exit(-1);

    // how will I know if amx is doing the right thing?
    check_for_correctness();

    //BFloat16 *A = new BFloat16[1048576];
    std::unique_ptr<BFloat16[]> A = std::make_unique<BFloat16[]>(1048576);
    std::unique_ptr<BFloat16[]> B = std::make_unique<BFloat16[]>(1048576);
    std::unique_ptr<BFloat16[]> transformed_B = std::make_unique<BFloat16[]>(1048576);
    std::unique_ptr<float[]> C = std::make_unique<float[]>(1048576);

    for(int idx = 0; idx < 1048576; idx++) {
        if (idx % 2 == 0) {
            A[idx] = 1.0;
            B[idx] = 1.0;
        }
        else {
            A[idx] = 2.0;
            B[idx] = 2.0;
        }       
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
    
    //float res_test[1048576] = {0.0};
    //float *res_test = new float[1048576];
    //std::fill(std::begin(res_test), std::end(res_test), 0.0);
    /*for(int idx = 0; idx < 1048576; idx++) {
        res_test[idx] = 0.0;
    }*/
    int r_org = 1024, c_org = 1024;
    int r_new = 512, c_new = 2048;

    int r_o = 0, c_o = 0;

    for (int r = 0; r < r_new; r++) {
        for (int c = 0; c < c_new; c += 2) {
            transformed_B[r * c_new + c] = B[(r_o * c_org) + c_o];
            transformed_B[r * c_new + c + 1] = B[(r_o * c_org) + c_o + c_org];
            c_o++;
        }
        r_o += 2;
        c_o = 0;
    }
    
    struct amx_tileconfig tc = {};
    tc.palette = 1;
    tc.start_row = 0;
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

    char *a = (char *)A.get();
    char *b = (char *)transformed_B.get();
    char *c = (char *)C.get();

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
        if (res[idx] != C[idx]) {
            printf("not fine\n");
            exit(-1);
        }
    }

    printf("fine\n");

    //delete [] A;
    //delete [] B;
    //delete [] B_new;
    delete [] res;
    //delete [] res_test;
    return 0;
}