#include <stdint.h>

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

void do_my_transpose(uint8_t *dst, uint8_t *src, int nx, int ny) {
    for (int ix = 0; ix < nx; ix += 128) {
        int offset = 0;

        // loop for d (128*2)
        for (int i = 0; i < 256; i++) {
            dst[ix * ny + offset++] = src[(ix + i/2) * ny + i%2];
        }

        // loop for supergroup
        for (int base = 2; base < ny; base += 70) {
            for (int k = 0; k < 8; k++) {

                // extra (1-bit)
                for (int i = 0; i < 128; i += 8) {
                    dst[ix * ny + offset++] = transpose_bits(src, ix+i, base, k, ny, 1, 1);
                }

                // scale_h (1-bit)
                for (int i = 0; i < 128; i += 8) {
                    dst[ix * ny + offset++] = transpose_bits(src, ix+i, base+1, k, ny, 1, 1);
                }

                // scale_l (4-bit)
                for (int i = 0; i < 128; i += 8) {
                    for (int j = 0; j < 4; j++) {
                        dst[ix * ny + offset++] = transpose_bits(src, ix+i+j, base+2+k/2, k%2, ny, 4, 4);
                    }
                }

                // qs (2-bit)
                for (int i = 0; i < 32; i++) {
                    for (int jjj = 0; jjj < 2; jjj++) {
                        for (int jj = jjj; jj < 8; jj += 2) {
                            for (int j = 0; j < 4; j++) {
                                dst[ix * ny + offset++] = transpose_bits(src, ix+4*jj+j, base+6+i+32*(k/4), k%4, ny, 32, 2);
                            }
                        }
                    }
                }
            }
        }
    }
}
