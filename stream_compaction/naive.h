#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();
        void scanDevice(int n, int* dev_out, const int* dev_in);
        void scan(int n, int *odata, const int *idata);

        int compact(int n, int* odata, const int* idata);
    }
}
