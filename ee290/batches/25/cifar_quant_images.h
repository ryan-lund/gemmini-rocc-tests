
  #ifndef MOBILENET_IMAGES_H
  #define MOBILENET_IMAGES_H

  #ifdef ELEM_T_IS_BFLOAT
  static bfloat16_t images[100][32][32][3] row_align(1);
  #else
  #endif

  static const int labels[100] = {3,8,8,0,6,6,1,6,3,1,0,9,5,7,9,8,5,7,8,6,7,0,4,9,5,2,4,0,9,6,6,5,4,5,9,2,4,1,9,5,4,6,5,6,0,9,3,9,7,6,9,8,0,3,8,8,7,7,4,6,7,3,6,3,6,2,1,2,3,7,2,6,8,8,0,2,9,3,3,8,8,1,1,7,2,5,2,7,8,9,0,3,8,6,4,6,6,0,0,7};

  static const int num_batches = 25;

  #endif
  