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


void do_my_transpose(uint8_t *dst, uint8_t *src, int nx, int ny, int stride, int avxlen, bool full_trans) {
    int offset = 0;
    for (int ix = 0; ix < nx; ix += stride) {

        // loop for d (stride*2)
        for (int i = 0; i < stride*2; i++) {
            dst[offset++] = src[(ix + i/2) * ny + i%2];
        }

        // loop for supergroup
        for (int base = 2; base < ny; base += 70) {
            for (int k = 0; k < 8; k++) {
                // extra (1-bit)
                // AVX256: unpack 8*1bit b0b1b2b3b4b5b6b7
                // AVX512: unpack 16*1bit b0b2b4b6b8b10b12b14|b1b3b5b7b9b11b13b15
                for (int i = 0; i < stride; i += avxlen/32) {
                    for (int j = 0; j < avxlen/256; ++j) {
                        dst[offset++] = transpose_bits(src, ix+i+j, base, k, ny, avxlen/256, 1);
                    }
                }

                // scale_h (1-bit)
                for (int i = 0; i < stride; i += avxlen/32) {
                    for (int j = 0; j < avxlen/256; ++j) {
                        dst[offset++] = transpose_bits(src, ix+i+j, base+1, k, ny, avxlen/256, 1);
                    }
                }

                // scale_l (4-bit)
                // AVX256: unpack 8*4bit  B0B4|B1B5|B2B6|B3B7
                // AVX512: unpack 16*4bit B0B8|B1B9|B2B10|B3B11||B4B12|B5B13|B6B14|B7B15
                for (int i = 0; i < stride; i += avxlen/32) {
                    for (int j = 0; j < avxlen/64; ++j) {
                        dst[offset++] = transpose_bits(src, ix+i+j, base+2+k/2, k%2, ny, avxlen/64, 4);
                    }
                }
            }
            if (full_trans) continue;
            for (int k = 0; k < 8; k++) {
                // qs (2-bit)
                /* avx256:
                    b0b32b64b96 |b1b33b65b97 |b2b34b66b98 |b3b35b67b99 |<3*4B>|| //128bit
                    b4b36b68b100|b5b37b69b101|b6b38b70b102|b7b39b71b103|<3*4B>|| //128bit
                   avx512:
                    b0b64b128b192|b1...|b2...|b3...|<3*4B>|| //128bit
                    b4b68b132b196|b5...|b6...|b7...|<3*4B>|| //128bit
                    b8b72b136b200|b9...|b10..|b11..|<3*4B>|| //128bit
                    b12b76b140b204|b13.|b14..|b15..|<3*4B>|| //128bit
                */
                for (int i = 0; i < 32; ++i) {
                    for (int ii = 0; ii < stride/(avxlen/2); ++ii) { // avxlen/2 2b-weights in a avx register
                        for (int basej = 0; basej < avxlen/128; ++basej) { // loop for 128b lanes
                            for (int stepj = 0; stepj < 4; ++stepj) { // 4x4B
                                for (int j = 0; j < 4; ++j) {
                                    dst[offset++] = transpose_bits(src,
                                        /*col_id=*/ ix+(avxlen/2)*ii+4*basej+(avxlen/32)*stepj+j,
                                        /*row_id*/ base+6+i+32*(k/4),
                                        k%4, ny, avxlen/8, 2);
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
                for (int i = 0; i < 32; ++i) {
                    for (int ii = 0; ii < nx/(avxlen/2); ++ii) { // avxlen/2 2b-weights in a avx register
                        for (int basej = 0; basej < avxlen/128; ++basej) { // loop for 128b lanes
                            for (int stepj = 0; stepj < 4; ++stepj) { // 4x4B
                                for (int j = 0; j < 4; ++j) {
                                    dst[offset++] = transpose_bits(src,
                                        /*col_id=*/ (avxlen/2)*ii+4*basej+(avxlen/32)*stepj+j,
                                        /*row_id*/ base+6+i+32*(k/4),
                                        k%4, ny, avxlen/8, 2);
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
