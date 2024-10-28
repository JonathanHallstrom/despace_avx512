#!/bin/bash
zig cc -c src/despace_avx512.c -march=native -mavx512bw -mevex512 -mavx512vbmi2 -O3 && zig run src/main.zig despace_avx512.o -OReleaseFast