#include <sys/syscall.h>
#include <unistd.h>
#include <asm/prctl.h>
#include <sys/prctl.h>

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