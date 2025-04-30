// sample.cu

#include <hip/hip_runtime.h>
#include <iostream>

__global__ void add_kernel(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    const int N = 256;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    size_t size = N * sizeof(int);
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    hipMemcpy(d_a, a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, b, size, hipMemcpyHostToDevice);

    add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    hipMemcpy(c, d_c, size, hipMemcpyDeviceToHost);

    std::cout << "c[0] = " << c[0] << ", c[1] = " << c[1] << std::endl;

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}