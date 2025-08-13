#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

static inline uint8_t transpose_bits(
    uint8_t *src, int x, int y, int k, int stride, int step, int bits) {
    uint8_t result = 0;
    uint8_t mask = (1 << bits) - 1;
    int count = 8 / bits;
    for (int i = 0; i < count; i++) {
        result |= ((src[(x + i*step) * stride + y] >> (k * bits)) & mask) << (bits * i);
    }
    return result;
}


void do_my_transpose(uint8_t *dst, uint8_t *src, int nx, int ny, int stride, bool full_trans) {
    int offset = 0;
    for (int ix = 0; ix < nx; ix += stride) {

        // loop for d (128*2)
        for (int i = 0; i < stride*2; i++) {
            dst[offset++] = src[(ix + i/2) * ny + i%2];
        }

        // loop for supergroup
        for (int base = 2; base < ny; base += 70) {
            for (int k = 0; k < 8; k++) {

                // extra (1-bit)
                for (int i = 0; i < stride; i += 8) {
                    dst[offset++] = transpose_bits(src, ix+i, base, k, ny, 1, 1);
                }

                // scale_h (1-bit)
                for (int i = 0; i < stride; i += 8) {
                    dst[offset++] = transpose_bits(src, ix+i, base+1, k, ny, 1, 1);
                }

                // scale_l (4-bit)
                for (int i = 0; i < stride; i += 8) {
                    for (int j = 0; j < 4; j++) {
                        dst[offset++] = transpose_bits(src, ix+i+j, base+2+k/2, k%2, ny, 4, 4);
                    }
                }

                if (full_trans) continue;
                // qs (2-bit)
                for (int i = 0; i < 32; i++) {
                    for (int ii = 0; ii < stride/128; ii++) {
                        for (int jjj = 0; jjj < 2; jjj++) {
                            for (int jj = jjj; jj < 8; jj += 2) {
                                for (int j = 0; j < 4; j++) {
                                    dst[offset++] = transpose_bits(src, ix+128*ii+4*jj+j, base+6+i+32*(k/4), k%4, ny, 32, 2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (full_trans) {
        for (int base = 2; base < ny; base += 70) {
            for (int k = 0; k < 8; k++) {
                // qs (2-bit)
                for (int i = 0; i < 32; i++) {
                    for (int ii = 0; ii < nx/128; ii++) {
                        for (int jjj = 0; jjj < 2; jjj++) {
                            for (int jj = jjj; jj < 8; jj += 2) {
                                for (int j = 0; j < 4; j++) {
                                    dst[offset++] = transpose_bits(src, 128*ii+4*jj+j, base+6+i+32*(k/4), k%4, ny, 32, 2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert(offset == nx*ny);
}
