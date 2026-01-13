#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

void testGPU(int device) {
    cudaSetDevice(device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("\n=== Testing GPU %d ===\n", device);
    printf("GPU Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multi-processors: %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    
    // Test vector addition
    int numElements = 1000000;
    size_t size = numElements * sizeof(float);
    
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // Initialize host arrays
    for (int i = 0; i < numElements; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Vector addition completed in: %.3f ms\n", milliseconds);
    printf("Throughput: %.2f GB/s\n", 
           (3 * size / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0));
    
    // Verify result
    int errors = 0;
    for (int i = 0; i < numElements; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            errors++;
            if (errors > 10) break;
        }
    }
    
    if (errors == 0) {
        printf("Test PASSED!\n");
    } else {
        printf("Test FAILED with %d errors\n", errors);
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    
    printf("Total CUDA devices found: %d\n", numDevices);
    
    for (int i = 0; i < numDevices; i++) {
        testGPU(i);
    }
    
    // Test multi-GPU (optional)
    printf("\n=== Testing Multi-GPU Bandwidth ===\n");
    for (int src = 0; src < numDevices; src++) {
        for (int dst = 0; dst < numDevices; dst++) {
            if (src != dst) {
                cudaSetDevice(src);
                float *d_src, *d_dst;
                size_t size = 100 * 1024 * 1024; // 100 MB
                
                cudaMalloc(&d_src, size);
                cudaSetDevice(dst);
                cudaMalloc(&d_dst, size);
                
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                
                cudaEventRecord(start);
                cudaMemcpyPeer(d_dst, dst, d_src, src, size);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                float bandwidth = (size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
                
                printf("GPU %d -> GPU %d: %.2f GB/s\n", src, dst, bandwidth);
                
                cudaFree(d_src);
                cudaFree(d_dst);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
        }
    }
    
    return 0;
}
