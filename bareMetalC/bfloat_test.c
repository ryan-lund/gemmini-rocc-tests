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

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

#ifndef ELEM_T_IS_BFLOAT
  exit(0);
#endif

  gemmini_flush(0);

  rand_bfloat();

  exit(0);
}

