#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "cifar_quant_params3_og.h"
#include "cifar_quant_images.h"

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check;
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    #ifdef ELEM_T_IS_BFLOAT
    images_to_bfloat(4*num_batches, images, images_float);
    weights_to_bfloat(128, 64, conv_1_w, conv_1_w_float);
    weights_to_bfloat(192, 64, conv_2_w, conv_2_w_float);
    weights_to_bfloat(128, 448, fc_3_w, fc_3_w_float);
    weights_to_bfloat(128, 128, fc_4_w, fc_4_w_float);
    weights_to_bfloat(64, 128, fc_5_w, fc_5_w_float);
    #endif

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;

    int correct = 0;
    int match = 0;
    int mismatch = 0;
    int mismatch_tf_correct = 0;
    int mismatch_gemmini_correct = 0;
    int mismatch_neither = 0;

    elem_t batch_images[4][32][32][3];
    int batch_labels[4];
    int batch_tf_labels[4];

    for (int global_batch = 0; global_batch < num_batches; global_batch++) {
        gemmini_flush(0);
        // conv_1
        printf("Batch %d/%d\n", global_batch+1, num_batches);

        for (int a = 0; a < 4; a++) {
            for (int b = 0; b < 32; b++) {
                for (int c = 0; c < 32; c++) {
                    for (int d = 0; d < 3; d++) {
                        batch_images[a][b][c][d] = images[(4*global_batch) + a][b][c][d];
                    }   
                }
            } 
            batch_labels[a] = labels[(4*global_batch) + a];
            batch_tf_labels[a] = tf_labels[(4*global_batch) + a];           
        }


        start = read_cycles();
        
        im2col(conv_1_params.batch_size, conv_1_params.in_channels, conv_1_params.in_dim,
            conv_1_params.I, conv_1_params.K,
            batch_images, conv_1_in, &conv_1_params);
        
        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_1_params.I, conv_1_params.J, conv_1_params.K,
            conv_1_in, conv_1_w, NULL, conv_1_out,
            RELU, conv_1_params.output_scale, true, conv_1_params.output_scale,
            tiled_matmul_type, check, "conv_1");

        end = read_cycles();
        matmul_cycles += end - start;

        start = read_cycles();

        pool_with_col2im(conv_1_params.I, conv_1_params.J,
            conv_1_params.batch_size, conv_1_params.out_channels, conv_1_params.out_dim_pooled,
            conv_1_out, conv_1_out_pooled, &conv_1_params);

        end = read_cycles();
        pool_cycles += end - start;

        // conv_2
        start = read_cycles();
        
        im2col(conv_2_params.batch_size, conv_2_params.in_channels, conv_2_params.in_dim,
            conv_2_params.I, conv_2_params.K,
            conv_1_out_pooled, conv_2_in, &conv_2_params);
        
        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_2_in, conv_2_w, NULL, conv_2_out,
            RELU, conv_2_params.output_scale, true, conv_2_params.output_scale,
            tiled_matmul_type, check, "conv_2");

        end = read_cycles();
        matmul_cycles += end - start;

        start = read_cycles();

        pool_with_col2im(conv_2_params.I, conv_2_params.J,
            conv_2_params.batch_size, conv_2_params.out_channels, conv_2_params.out_dim_pooled,
            conv_2_out, conv_2_out_pooled, &conv_2_params);

        end = read_cycles();
        pool_cycles += end - start;

        // Convert conv output to fc input
        static elem_t fc_3_in[448][64] row_align(1);

        start = read_cycles();

        for (size_t batch = 0; batch < conv_2_params.batch_size; batch++) {
            size_t pixel = 0;
            for (size_t channel = 0; channel < conv_2_params.out_channels; channel++) {
                for (size_t row = 0; row < conv_2_params.out_dim_pooled; row++) {
                    for (size_t col = 0; col < conv_2_params.out_dim_pooled; col++) {
                        fc_3_in[pixel][batch] = conv_2_out_pooled[batch][row][col][channel];
                        pixel++;
                    }
                }
            }
        }

        end = read_cycles();
        other_cycles += end - start;

        // fc_3
        start = read_cycles();

        tiled_matmul_nn_auto(fc_3_params.I, fc_3_params.J, fc_3_params.K,
            fc_3_w, fc_3_in, NULL, fc_3_out,
            RELU, fc_3_params.output_scale, false, fc_3_params.output_scale,
            tiled_matmul_type, check, "fc_3");

        end = read_cycles();
        matmul_cycles += end - start;

        // fc_4
        start = read_cycles();

        tiled_matmul_nn_auto(fc_4_params.I, fc_4_params.J, fc_4_params.K,
            fc_4_w, fc_3_out, NULL, fc_4_out,
            RELU, fc_4_params.output_scale, false, fc_4_params.output_scale,
            tiled_matmul_type, check, "fc_4");

        end = read_cycles();
        matmul_cycles += end - start;

        // fc_5
        start = read_cycles();

        tiled_matmul_nn_auto(fc_5_params.I, fc_5_params.J, fc_5_params.K,
            fc_5_w, fc_4_out, NULL, fc_5_out,
            NO_ACTIVATION, fc_5_params.output_scale, false, fc_5_params.output_scale,
            tiled_matmul_type, check, "fc_5");

        end = read_cycles();
        matmul_cycles += end - start;

        // Find highest probs
        char * classes[] = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
        int preds[fc_5_params.batch_size];

        for (int batch = 0; batch < fc_5_params.batch_size; batch++) {
            elem_t max_prob = fc_5_out[0][batch];
            size_t max_idx = 0;

            for (int i = 1; i < fc_5_params.out_features; i++) {
                #ifndef ELEM_T_IS_BFLOAT
                if (fc_5_out[i][batch] > max_prob) {
                #else 
                if (!bf16_le(fc_5_out[i][batch], max_prob)) {
                #endif
                    max_prob = fc_5_out[i][batch];
                    max_idx = i;
                }
            }
            preds[batch] = max_idx;
        }

        for (int i = 0; i < fc_5_params.batch_size; i++) {
            if (preds[i] == batch_labels[i]) {
                correct += 1;
            }
            if (preds[i] == batch_tf_labels[i]) {
                match += 1;
            } else {
                mismatch += 1;
                if (preds[i] == batch_labels[i]) {
                    mismatch_gemmini_correct += 1;
                } else if (batch_tf_labels[i] == batch_labels[i]) {
                    mismatch_tf_correct += 1;
                } else {
                    mismatch_neither += 1;
                }
            }
        }
    }
    
    uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_dw_cycles + res_add_cycles + other_cycles;

    printf("\nTotal cycles: %llu\n", total_cycles);
    printf("Matmul cycles: %llu\n", matmul_cycles);
    printf("Im2col cycles: %llu\n", im2col_cycles);
    printf("Pooling cycles: %llu\n", pool_cycles);
    printf("Depthwise convolution cycles: %llu\n", conv_dw_cycles);
    printf("Other cycles: %llu\n", other_cycles);


    printf("Test Finished \n");
    printf("Correct: %d / %d \n", correct, 4*num_batches);
    printf("Matched: %d / %d \n", match, 4*num_batches);
    printf("Mismatched, TF Correct: %d / %d \n", mismatch_tf_correct, mismatch);
    printf("Mismatched, Gemmini Correct: %d / %d \n", mismatch_gemmini_correct, mismatch);
    printf("Mismatched, Neither Correct: %d / %d \n", mismatch_neither, mismatch);
    exit(0);
}

