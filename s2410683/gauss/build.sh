#!/bin/bash
set -e
echo "=== Cleaning old binaries ==="
rm -f ./main ./d ./debug *.o
echo "=== Verifying source ==="
head -1 main.cc
echo "=== Compiling ==="
g++ -O2 -march=armv8-a+simd -fopenmp -o main main.cc
echo "=== Binary info ==="
ls -la ./main
file ./main
echo "=== Testing n=64 (scalar, 1 run) ==="
./main 64 114514 1 1
echo "=== Testing n=64 (SIMD, 5 runs) ==="
./main 64 114514 5 0
