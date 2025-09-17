/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"
#include <iostream>
#include <vector>
#include <functional>
#include <array>
#include <chrono>

const int SIZE = 1 << 26; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
const int SIZE_SORT = 1 << 22;
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int* aa = new int[SIZE_SORT];
int* bb = new int[SIZE_SORT];
int* cc = new int[SIZE_SORT];

const bool PERFTEST = false;


struct Impl {
    const char* name;
    std::function<void(int, int*, const int*)> scan;
    std::function<int(int, int*, const int*)> compact;
    StreamCompaction::Common::PerformanceTimer& timer;
};

std::vector<Impl> implementations = {
    { "cpu_naive",
      StreamCompaction::CPU::scan,
      StreamCompaction::CPU::compactWithScan,
      StreamCompaction::CPU::timer() },

    { "gpu_naive",
      StreamCompaction::Naive::scan,
      StreamCompaction::Naive::compact,
      StreamCompaction::Naive::timer() },

    { "gpu_efficient",
      StreamCompaction::Efficient::scan,
      StreamCompaction::Efficient::compact,
      StreamCompaction::Efficient::timer() },

    { "gpu_thrust",
      StreamCompaction::Thrust::scan,
      nullptr,
      StreamCompaction::Thrust::timer() },
};

struct PerfData {
    std::vector<float> compactTotal;
    std::vector<float> map;
    std::vector<float> scan;
    std::vector<float> scatter;
    std::vector<float> scanTotal;
};

std::map<std::string, PerfData> allResults;


void test() {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);


    printf("\n");
    printf("*******************\n");
    printf("** SORTING TESTS **\n");
    printf("*******************\n");

    genArray(SIZE_SORT, aa, 50);
    printArray(SIZE_SORT, aa, true);

    memcpy(bb, aa, SIZE_SORT * sizeof(int));
    printDesc("cpu std::sort");
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(bb, bb + SIZE_SORT);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    float ms = duration.count();
    printElapsedTime(ms, "(std::chrono Measured)");
    printArray(SIZE_SORT, bb, true);

    zeroArray(SIZE_SORT, cc);
    printDesc("work-efficient radix sort");
    StreamCompaction::Efficient::radixSort(SIZE_SORT, cc, aa);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE_SORT, cc, true);


    // Compare results
    printCmpResult(SIZE_SORT, bb, cc);

    delete[] a;
    delete[] b;
    delete[] c;

    delete[] aa;
    delete[] bb;
    delete[] cc;
}

template<typename T>
void printVector(const std::vector<T>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i + 1 < vec.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

template <typename ScanFunc, typename CompactFunc>
std::array<float, 5> runTest(ScanFunc scanFunc, CompactFunc compactFunc, StreamCompaction::Common::PerformanceTimer& timer, int size, int* input, int* output) {
    float compactTimeTotal = 0.f, mapTime = 0.f, scanTime = 0.f, scatterTime = 0.f, scanTimeTotal = 0.f;
    zeroArray(size, output);
    if (compactFunc) {
        for (int i = 0; i < 3; i++) {
            compactFunc(size, output, input);
            compactTimeTotal += timer.getElapsedTimeForPreviousOperation();
            mapTime += timer.getSubTimer("map");
            scanTime += timer.getSubTimer("scan");
            scatterTime += timer.getSubTimer("scatter");
        }
        compactTimeTotal /= 3;
        mapTime /= 3;
        scanTime /= 3;
        scatterTime /= 3;
    }
    zeroArray(size, output);
    if (scanFunc) {
        for (int i = 0; i < 3; i++) {
            scanFunc(size, output, input);
            scanTimeTotal += timer.getElapsedTimeForPreviousOperation();
        }
        scanTimeTotal /= 3;
    }
    return { compactTimeTotal, mapTime, scanTime, scatterTime, scanTimeTotal };
}

void perftest(const std::vector<int>& sizes) {
    allResults.clear();
    for (size_t i = 0; i < sizes.size(); i++) {
        int size = sizes[i];
        int* input = new int[size];
        int* output = new int[size];
        genArray(size, input, 50);

        for (auto& impl : implementations) {
            auto timings = runTest(impl.scan, impl.compact, impl.timer, size, input, output);

            PerfData& data = allResults[impl.name];
            data.compactTotal.push_back(timings[0]);
            data.map.push_back(timings[1]);
            data.scan.push_back(timings[2]);
            data.scatter.push_back(timings[3]);
            data.scanTotal.push_back(timings[4]);
        }
        delete[] input;
        delete[] output;
    }

    printf("Array length: ");
    printVector(sizes);
    printf("\n");

    for (auto& [name, data] : allResults) {

        std::cout << name << "_com_tot = "; printVector(data.compactTotal);
        std::cout << name << "_com_map = "; printVector(data.map);
        std::cout << name << "_com_scan = "; printVector(data.scan);
        std::cout << name << "_com_scat = "; printVector(data.scatter);
        std::cout << name << "_scan_tot = "; printVector(data.scanTotal);
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (PERFTEST) {
        // Powers of two test + half sizes
        printf("===============================\n");
        printf("Powers of Two Performance Sweep\n");
        printf("===============================\n");
        int lowestPot = 3;
        int highestPot = 28;
        std::vector<int> potHpot;
        for (int i = 0; i < (highestPot - lowestPot); i++) {
            int pot = 1 << (i + lowestPot);
            potHpot.push_back(5*pot/4 - pot/2);
            potHpot.push_back(pot);
        }
        perftest(potHpot);
        return 0;
    }
    else {
        test();
    }

    system("pause"); // stop Win32 console from closing on exit   
}
