#ifndef GEMMINI_PARAMS_H
#define GEMMINI_PARAMS_H

#include <stdint.h>
#include <limits.h>
#include "softfloat/source/include/softfloat.h" 

#define DIM 8
#define ADDR_LEN 32
#define BANK_NUM 4
#define BANK_ROWS 2048
#define ACC_ROWS 2048
#define MAX_BYTES 64
#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*4))
#define MAX_BLOCK_LEN_ACC (MAX_BYTES/(DIM*4))

typedef float elem_t;
typedef float acc_t;
typedef double full_t;

elem_t elem_t_max = 3.4028235E38;
elem_t elem_t_min = -3.4028235E38;
#define ELEM_T_IS_FLOAT
#define ELEM_T_EXP_BITS 8
#define ELEM_T_SIG_BITS 24
#define ACC_T_EXP_BITS 8
#define ACC_T_SIG_BITS 24
typedef uint32_t elem_t_bits;
typedef uint32_t acc_t_bits;

#define HAS_MVIN_SCALE
typedef float scale_t;
typedef uint32_t scale_t_bits;

#define HAS_MVIN_ACC_SCALE
typedef float scale_acc_t;
typedef uint32_t scale_acc_t_bits;

#define row_align(blocks) __attribute__((aligned(blocks*DIM*sizeof(elem_t))))
#define row_align_acc(blocks) __attribute__((aligned(blocks*DIM*sizeof(acc_t))))

#define MVIN_SCALE_ONE 1.0

#endif // GEMMINI_PARAMS_H