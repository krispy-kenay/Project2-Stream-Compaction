#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScanPass(int n, int offset, const int* in, int* out) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= n) return;

            if (i >= offset) {
                out[i] = in[i] + in[i - offset];
            }
            else {
                out[i] = in[i];
            }
        }

        __global__ void kernInclusiveToExclusive(int n, const int* in, int* out) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= n) return;

            out[i] = (i == 0) ? 0 : in[i - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            if (n <= 0) {
                return;
            }
            int* dev_bufA = nullptr;
            int* dev_bufB = nullptr;

            cudaMalloc(&dev_bufA, n * sizeof(int));
            cudaMalloc(&dev_bufB, n * sizeof(int));

            cudaMemcpy(dev_bufA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 blockSize(256);
            dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

            int passes = ilog2ceil(n);

            timer().startGpuTimer();
            
            int* in = dev_bufA;
            int* out = dev_bufB;

            for (int i = 0; i < passes; i++) {
                int offset = 1 << i;
                kernNaiveScanPass<<<gridSize, blockSize >>>(n, offset, in, out);
                cudaDeviceSynchronize();
                std::swap(in, out);
            }
            kernInclusiveToExclusive<<<gridSize, blockSize>>>(n, in, out);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_bufA);
            cudaFree(dev_bufB);
        }
    }
}
