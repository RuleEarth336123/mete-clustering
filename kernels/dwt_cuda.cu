#include <iostream>
#include <cmath>

__global__ void dtw_kernel(float *dist, float *seq1, float *seq2, int m, int n, int window) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= m) return;
    for (int j = 0; j < n; j++) {
        if (abs(i - j) > window) continue; // Apply window constraint

        float cost = pow((seq1[i] - seq2[j]), 2);
        
        if (i == 0 && j == 0) {
            dist[i * n + j] = cost;
        } else if (i == 0) {
            dist[i * n + j] = cost + dist[j - 1];
        } else if (j == 0) {
            dist[i * n + j] = cost + dist[(i - 1) * n];
        } else {
            dist[i * n + j] = cost + fminf(dist[(i - 1) * n + j], fminf(dist[i * n + j - 1], dist[(i - 1) * n + j - 1]));
        }
    }
}

int main() {
    const int m = 100; // Length of sequence 1
    const int n = 100; // Length of sequence 2
    const int window = 10; // Window constraint

    // Allocate memory on host
    float *seq1 = new float[m];
    float *seq2 = new float[n];
    float *dist = new float[m * n];

    // Initialize sequences with data

    // Allocate memory on device
    float *d_seq1, *d_seq2, *d_dist;
    cudaMalloc(&d_seq1, m * sizeof(float));
    cudaMalloc(&d_seq2, n * sizeof(float));
    cudaMalloc(&d_dist, m * n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_seq1, seq1, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq2, seq2, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int num_blocks = (m + block_size - 1) / block_size;
    dtw_kernel<<<num_blocks, block_size>>>(d_dist, d_seq1, d_seq2, m, n, window);

    // Copy result back to host
    cudaMemcpy(dist, d_dist, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_seq1);
    cudaFree(d_seq2);
    cudaFree(d_dist);

    // Free host memory
    delete[] seq1;
    delete[] seq2;
    delete[] dist;

    return 0;
}
