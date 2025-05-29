#include <sys/syscall.h>
#include <unistd.h>
#include <asm/prctl.h>
#include <sys/prctl.h>
#include <cpuid.h>

#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

struct amx_tileconfig
{
    uint8_t palette;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

bool check_amx_support() {
    unsigned int eax, ebx, ecx, edx;
    
    // Check if CPUID leaf 7 is supported
    __cpuid(0, eax, ebx, ecx, edx);
    if (eax < 7) return false;
    
    // Get CPUID leaf 7, subleaf 0
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    
    bool amx_tile = (edx >> 24) & 1;  // Bit 24
    bool amx_bf16 = (edx >> 22) & 1;  // Bit 22  
    
    printf("AMX-TILE: %s\n", amx_tile ? "Yes" : "No");
    printf("AMX-BF16: %s\n", amx_bf16 ? "Yes" : "No");
    //printf("AMX-INT8: %s\n", amx_int8 ? "Yes" : "No");
    
    return amx_tile && amx_bf16;
}

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