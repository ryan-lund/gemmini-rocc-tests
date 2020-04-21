// See LICENSE for license details.
// The main point of this test is just to check whether we can switch between output- and weight-stationary dataflows

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include <time.h>
#include "include/gemmini.h"

#define N (2)

#define ABS(x)    (((x) < 0) ? -(x) : (x))
#define SIGN(x)   (((x) < 0) ? (1) : (0))

#define NUM_TESTS (10)

int failed = 0;

uint32_t float_to_int(float f) {
  return *((uint32_t*) &f);
}

uint16_t bf_to_int(bfloat16_t bf) {
  return *((uint16_t*) &bf);
}

void check_result(float a, float b) {
  printf("Result: 0x%x \nExpected: 0x%x \n", float_to_int(a), float_to_int(b));
  if (ABS(a) - ABS(b) < 0.25 && SIGN(a) == SIGN(b)) {
    printf("Test passes \n");
  } else {
    printf("Test fails \n");
    failed++;
  }
  printf("\n");
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

#ifndef ELEM_T_IS_BFLOAT
  exit(0);
#else

  gemmini_flush(0);
  
  for (int i = 0; i < NUM_TESTS; i++) {
    printf("Starting test round %d \n", i);
    printf("Starting round trip test \n");
    float f = rand_double();
    bfloat16_t bf = float_to_bf16(f);
    float f2 = bf16_to_float(bf);
    check_result(f, f2);  
  
    printf("Starting addition test \n");
    f = rand_double();
    f2 = rand_double();
    bf = float_to_bf16(f);
    bfloat16_t bf2 = float_to_bf16(f2);
    bfloat16_t bf3 = bf16_add(bf, bf2);
    float f3 = bf16_to_float(bf3);
    check_result(f3, f+f2);

    printf("Starting multiplication test \n");
    bf3 = bf16_mul(bf, bf2);
    f3 = bf16_to_float(bf3);
    check_result(f3, f*f2);
    
    printf("Starting mulAdd test \n");
    float f4 = rand_double();
    bfloat16_t bf4 = float_to_bf16(f4);
    float f5 = bf16_to_float(bf16_mulAdd(bf, bf2, bf4));
    check_result(f5, f*f2+f4);

    printf("Starting equality tests \n");
    check_result(bf16_eq(bf, bf), f == f);
    check_result(bf16_eq(bf, bf2), f == f2);
    check_result(bf16_lt(bf, bf2), f < f2);
    check_result(bf16_le(bf, bf), f <= f);
    check_result(bf16_le(bf, bf2), f <= f2); 
  }
 
  if (failed > 0) {
    printf("%d tests failed \n"); 
  } else {
    printf("All tests passed \n");
  }
 
  exit(0);
#endif
}

