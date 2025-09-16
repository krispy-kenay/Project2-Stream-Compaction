#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 256

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

        void scanDevice(int n, int* dev_out, const int* dev_in) {
            int* dev_bufA = nullptr;
            int* dev_bufB = nullptr;

            cudaMalloc(&dev_bufA, n * sizeof(int));
            cudaMalloc(&dev_bufB, n * sizeof(int));
            cudaMemcpy(dev_bufA, dev_in, n * sizeof(int), cudaMemcpyHostToDevice);

            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            int passes = ilog2ceil(n);

            int* in = dev_bufA;
            int* out = dev_bufB;

            for (int i = 0; i < passes; i++) {
                int offset = 1 << i;
                kernNaiveScanPass<<<gridSize, BLOCK_SIZE >>>(n, offset, in, out);
                cudaDeviceSynchronize();
                std::swap(in, out);
            }
            kernInclusiveToExclusive<<<gridSize, BLOCK_SIZE >>>(n, in, out);
            cudaDeviceSynchronize();

            cudaMemcpy(dev_out, out, n * sizeof(int), cudaMemcpyDeviceToDevice);

            cudaFree(dev_bufA);
            cudaFree(dev_bufB);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }
            int* dev_in, * dev_out;
            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            scanDevice(n, dev_out, dev_in);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            int* dev_in, * dev_flags, * dev_indices, * dev_out;
            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_flags, n * sizeof(int));
            cudaMalloc(&dev_indices, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            timer().startGpuTimer();

            timer().startGpuSubTimer("map");
            StreamCompaction::Common::kernMapToBoolean<<<gridSize, BLOCK_SIZE >>>(n, dev_flags, dev_in);
            cudaDeviceSynchronize();
            timer().endGpuSubTimer();

            timer().startGpuSubTimer("scan");
            scanDevice(n, dev_indices, dev_flags);
            cudaDeviceSynchronize();
            timer().endGpuSubTimer();

            int lastScan, lastFlag;
            cudaMemcpy(&lastScan, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastFlag, dev_flags + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int validCount = lastScan + lastFlag;

            timer().startGpuSubTimer("scatter");
            StreamCompaction::Common::kernScatter<<<gridSize, BLOCK_SIZE >>>(n, dev_out, dev_in, dev_flags, dev_indices);
            cudaDeviceSynchronize();
            timer().endGpuSubTimer();

            timer().endGpuTimer();

            if (validCount > 0) {
                cudaMemcpy(odata, dev_out, validCount * sizeof(int), cudaMemcpyDeviceToHost);
            }

            cudaFree(dev_in);
            cudaFree(dev_flags);
            cudaFree(dev_indices);
            cudaFree(dev_out);

            return validCount;
        }
    }
}
