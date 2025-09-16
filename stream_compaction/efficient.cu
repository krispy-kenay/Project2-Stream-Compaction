#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpsweep(int d, int n, int* data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int stride = 1 << (d + 1);
            int numWorkItems = n / stride;

            if (k >= numWorkItems) return;

            int right = (k + 1) * stride - 1;
            int left = right - (1 << d);

            data[right] += data[left];
        }

        __global__ void kernDownsweep(int d, int n, int* data) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int stride = 1 << (d + 1);
            int numWorkItems = n / stride;

            if (k >= numWorkItems) return;

            int right = (k + 1) * stride - 1;
            int left = right - (1 << d);

            int t = data[left];
            data[left] = data[right];
            data[right] = t + data[right];
        }

        void scanDevice(int n, int* dev_out, const int* dev_in) {
            int log2n = ilog2ceil(n);
            int m = 1 << log2n;

            int* dev_buf;
            cudaMalloc(&dev_buf, m * sizeof(int));

            cudaMemcpy(dev_buf, dev_in, n * sizeof(int), cudaMemcpyDeviceToDevice);

            if (m > n) {
                cudaMemset(dev_buf + n, 0, (m - n) * sizeof(int));
            }

            for (int d = 0; d < log2n; d++) {
                int numWorkItems = m >> (d + 1);
                int blocks = (numWorkItems + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernUpsweep<<<blocks, BLOCK_SIZE >>>(d, m, dev_buf);
            }

            cudaMemset(dev_buf + (m - 1), 0, sizeof(int));

            for (int d = log2n - 1; d >= 0; d--) {
                int numWorkItems = m >> (d + 1);
                int blocks = (numWorkItems + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownsweep<<<blocks, BLOCK_SIZE >>>(d, m, dev_buf);
            }

            cudaMemcpy(dev_out, dev_buf, n * sizeof(int), cudaMemcpyDeviceToDevice);

            cudaFree(dev_buf);
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
        int compact(int n, int *odata, const int *idata) {
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
