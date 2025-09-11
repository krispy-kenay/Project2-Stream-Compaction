#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int write = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[write] = idata[i];
                    write += 1;
                }
            }
            timer().endCpuTimer();
            return write;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n == 0) return 0;
            int* flags = new int[n];
            int* indices = new int[n];

            for (int i = 0; i < n; i++) {
                flags[i] = (idata[i] != 0) ? 1 : 0;
            }

            { //Inline scan
                int sum = 0;
                for (int i = 0; i < n; i++) {
                    indices[i] = sum;
                    sum += flags[i];
                }
            }

            for (int i = 0; i < n; i++) {
                if (flags[i] == 1) {
                    odata[indices[i]] = idata[i];
                }
            }
            int count = (n == 0) ? 0 : indices[n - 1] + flags[n - 1];

            delete[] flags;
            delete[] indices;
            timer().endCpuTimer();
            return count;
        }
    }
}
