#include <memory>
#include <random>
#include <bfloat16.hpp>
#include <amx.hpp>

#define A_ROWS 1024
#define A_COLS 1024
#define B_ROWS 1024
#define B_COLS 1024
#define C_ROWS 1024
#define C_COLS 1024

void standard_matrix_multiplication(BFloat16 *A, BFloat16 *B, float *C_standard)
{
    for(int r = 0; r < C_ROWS; r++) {
        for(int c = 0; c < C_COLS; c++) {
            float acc = 0.0;
            for(int k = 0; k < B_COLS; k++) {
                acc += bf16_to_fp32(A[r * 1024 + k]) * bf16_to_fp32(B[k * 1024 + c]);
            }
            C_standard[r * 1024 + c] = acc;
        }
    }
}

void populate_A_B_with_random_values(BFloat16 *A, BFloat16 *B)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    // No de-normal numbers or infinities or NaNs
    std::uniform_real_distribution<float> dis(1.0f, 2.0f);

    for(int idx = 0; idx < 1024 * 1024; idx++) {
        A[idx] = dis(gen);
        B[idx] = dis(gen);
    }
}

// transform B's layout to be compatible with AMX instructions
void transform_B_matrix(BFloat16 *B, BFloat16 *transformed_B)
{
    int INITIAL_COLS = 1024;
    int TRANSFORMED_ROWS = 512, TRANSFORMED_COLS = 2048;

    int initial_r = 0, initial_c = 0;

    for (int r = 0; r < TRANSFORMED_ROWS; r++) {
        for (int c = 0; c < TRANSFORMED_COLS; c += 2) {
            transformed_B[r * TRANSFORMED_COLS + c] = B[(initial_r * INITIAL_COLS) + initial_c];
            transformed_B[r * TRANSFORMED_COLS + c + 1] = B[(initial_r * INITIAL_COLS) + initial_c + INITIAL_COLS];
            initial_c++;
        }
        initial_r += 2;
        initial_c = 0;
    }
}

void amx_matrix_multiplication(BFloat16 *A, BFloat16 *transformed_B, float *C_amx)
{
    // configure tile registers
    struct amx_tileconfig tc = {};
    tc.palette = 1;
    tc.start_row = 0;
    for (int jj = 0; jj < 3; ++jj) {
        tc.rows[jj] = 16;
        tc.colsb[jj] = 64;
    }

    _tile_loadconfig(&tc);

    int As_TILESIZE_BYTES = 16 * 64;
    int Bs_TILESIZE_BYTES = 16 * 64;
    int Cs_TILESIZE_BYTES = 16 * 64;

    int As_STRIDE_IN_BYTES = 2048;
    int Bs_STRIDE_IN_BYTES = 4096;
    int Cs_STRIDE_IN_BYTES = 4096;

    int TILES_R = 64, TILES_K = 32, TILES_C = 64;

    char *a = (char *)A;
    char *b = (char *)transformed_B;
    char *c = (char *)C_amx;

    for(int t_r = 0; t_r < TILES_R; t_r++) {
        for(int t_c = 0; t_c < TILES_C; t_c++) {
            
            // load tile in C to tmm0
            int c_offset_bytes = (t_r * TILES_C * Cs_TILESIZE_BYTES) + (t_c * 64);
            _tile_loadd(0, c + c_offset_bytes, Cs_STRIDE_IN_BYTES);

            for(int t_k = 0; t_k < TILES_K; t_k++) {
                
                //load tile in A to tmm1
                int a_offset_bytes = (t_r * TILES_K * As_TILESIZE_BYTES) + (t_k * 64);
                _tile_loadd(1, a + a_offset_bytes, As_STRIDE_IN_BYTES);
                
                // load tile in B to tmm2
                int b_offset_bytes = (t_k * TILES_C * Bs_TILESIZE_BYTES) + (t_c * 64);
                _tile_loadd(2, b + b_offset_bytes, Bs_STRIDE_IN_BYTES);
                
                // tmm0 += matmul(tmm1, tmm2)
                _tile_dpbf16ps(0, 1, 2);
            }
            //store tmm0 back to C
            _tile_stored(0, c + c_offset_bytes, Cs_STRIDE_IN_BYTES);
        }
    }

    _tile_release();
}

void compare_standard_and_amx_results(float *C_standard, float *C_amx)
{
    /* Convert the original FP32 result to bfloat16 because FP32 results for C_standard and C_amx will be different
    due to re-ordering of operations. When converted to bfloat16 the max difference uint16_t representation of bfloat16 
    for both cases will be 1 which we will take into account when comparing */

    for(int idx = 0; idx < 1024 * 1024; idx++) {
        BFloat16 a = fp32_to_bf16(C_standard[idx]);
        BFloat16 b = fp32_to_bf16(C_amx[idx]);
        if (a.data != b.data && abs(a.data - b.data) > 1) {
            printf("Mismatch in computation at idx:%d. a.data: %u b.data: %u\n", idx, a.data, b.data);
            exit(-1);
        }
    }
}

int main()
{

    // check if current processor supports
    if(!check_amx_support()) {
        printf("AMX not supported on the current processor\n");
        exit(-1);
    }

    // Request linux kernel's permission to run AMX
    if (!set_tiledata_use())
        exit(-1);

    // Allocate necessary data
    std::unique_ptr<BFloat16[]> A = std::make_unique<BFloat16[]>(A_ROWS * A_COLS);
    std::unique_ptr<BFloat16[]> B = std::make_unique<BFloat16[]>(B_ROWS * B_COLS);
    std::unique_ptr<BFloat16[]> transformed_B = std::make_unique<BFloat16[]>((B_ROWS / 2) * (B_COLS * 2));
    std::unique_ptr<float[]> C_amx = std::make_unique<float[]>(C_ROWS * C_COLS);
    std::unique_ptr<float[]> C_standard = std::make_unique<float[]>(C_ROWS * C_COLS);

    populate_A_B_with_random_values(A.get(), B.get());
    
    standard_matrix_multiplication(A.get(), B.get(), C_standard.get());

    transform_B_matrix(B.get(), transformed_B.get());

    amx_matrix_multiplication(A.get(), transformed_B.get(), C_amx.get());

    // compare results from standard multiplication. How will i verify if i am right or not?
    compare_standard_and_amx_results(C_standard.get(), C_amx.get());

    printf("Both standard matrix multiplication values and amx matrix multiplication values match\n");

    return 0;
}