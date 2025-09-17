**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2**

* Yannick Gachnang
  * [LinkedIn](https://www.linkedin.com/in/yannickga/)
* Tested on: Windows 10, EPYC 9354 @ 3.25GHz, 16GB RAM, RTX 2000 Ada 23GB VRAM

Project 2 Stream Compaction
====================

This project implements stream compaction in CUDA using different versions of prefix sum (scan). The implementations include a CPU baseline, a naive GPU scan, a work-efficient GPU scan with compaction, and Thrust’s built-in scan. Additionally, I implemented radix sort as an extra credit feature.

## Implementation

As per the project instruction, I implemented all the required scan and compaction variants, plus radix sort as an extension. Below is a breakdown of the different implementations and how they work.

### CPU Scan & Compaction

For the CPU side, I wrote three different functions. The scan is implemented as a simple exclusive prefix sum using a for loop. Then I added two versions of compaction: one that simply loops over the input and copies nonzero values directly, and one that mirrors the GPU algorithm by performing map → scan → scatter on the CPU. These CPU versions serve as the correctness reference for all the GPU tests.  

### Naive GPU Scan & Compaction

The naive GPU scan works by running `ilog2ceil(n)` passes where each pass adds values with a given offset. I used two buffers and swapped between them each iteration, and finally converted the inclusive result into an exclusive scan. This approach is simple but requires many kernel launches and synchronizations. I implemented stream compaction on top of this by first mapping the input into 0/1 flags, then scanning these flags, and finally scattering the surviving elements into the output array. This method avoids race conditions and works in place, but at the upper levels of the tree only a few threads remain active which means utilization goes down.  

### Work-Efficient GPU Scan & Compaction

The work-efficient GPU scan is based on Blelloch’s upsweep/downsweep algorithm. In the upsweep phase, partial sums are built in a binary tree structure, and then in the downsweep the values are propagated back down to get an exclusive scan. The compaction was implemented the same way as in the naive version, except with the scan portion swapped out.

### Thrust

Thrust’s scan is just a wrapper around `thrust::exclusive_scan` on a device vector and serves to compare my implementation against a highly optimized version of these algorithms.

### Radix Sort (Extra Credit)

Finally, for the extra credit I implemented radix sort on top of the work-efficient scan. Each digit is processed by splitting the array into buckets using map → scan → scatter, and then recombining them. Repeating this for each bit group results in a complete integer sort on the GPU.

### CMake modifications

I had to modify the CMakeLists.txt and add `target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include")` after `add_executable(...)` because otherwise CMake somehow cannot find CUDA. But aside from that I didn't add any other files or changed the file in any other way.

---

## Performance Testing

### CPU vs GPU Scaling

For this test I tested power-of-two inputs ranging from $2^3$ (8) elements up to $2^{28}$ (268435456) elements on the various scan implementations.

![GPU speedup factor](images/gpu_vs_cpu.png)

Naive GPU scan stays below zero everywhere. Starting at a factor somewhere around 0.0001x for small arrays and growing towards a factor of 0.2x (still 5 times slower) at $2^{20}$ elements. It levels off after that and stays relatively constant. The work-efficient scan is better than naive but it also sits below zero across the whole sweep, though it does better with a 0.51x factor towards the end of the range. Thrust starts negative at the very small sizes, crosses to positive as we scale up, and then keeps climbing. Around the mid-range its ahead of the CPU by about a factor of 1.5x and at the very end, thrust is 8.06 ms, which is roughly 5.9x faster than CPU.

<details>
  <summary>Raw Data</summary>
  
| Array Length | Speedup Naive  | Speedup Eff.   | Speedup Thrust |
|----------|----------------|----------------|----------------|
| 6        | 1.39499560e-04 | 8.52893803e-04 | 6.05208116e-04 |
| 12       | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 |
| 24       | 6.72432784e-05 | 2.44560121e-04 | 3.96271635e-04 |
| 48       | 6.39869543e-05 | 3.38973519e-04 | 4.60574549e-04 |
| 96       | 1.25914279e-04 | 7.41664071e-04 | 8.34669221e-04 |
| 192      | 1.57087719e-04 | 9.26655238e-04 | 7.70968413e-04 |
| 384      | 2.40813286e-04 | 2.30415698e-03 | 3.22331372e-03 |
| 768      | 2.78109457e-04 | 2.03450230e-03 | 3.65863334e-03 |
| 1536     | 4.79805007e-04 | 4.34431286e-03 | 7.70970791e-03 |
| 3072     | 1.08284094e-03 | 5.47925190e-03 | 1.38115824e-02 |
| 6144     | 3.47204942e-03 | 9.57380607e-03 | 3.46537570e-02 |
| 12288    | 3.97507627e-03 | 2.43559212e-02 | 3.94921745e-02 |
| 24576    | 6.89812596e-03 | 3.57076590e-02 | 1.03301862e-01 |
| 49152    | 2.03881838e-02 | 1.21899484e-01 | 3.45082873e-01 |
| 98304    | 3.16683276e-02 | 1.66221804e-01 | 5.16874836e-01 |
| 196608   | 5.61533161e-02 | 1.21991266e-01 | 8.53266300e-01 |
| 393216   | 9.64783260e-02 | 2.52851750e-01 | 1.84085557e+00 |
| 786432   | 1.78413329e-01 | 5.35759963e-01 | 1.12184691e+00 |
| 1572864  | 2.06761002e-01 | 7.19364838e-01 | 1.91361592e+00 |
| 3145728  | 2.53704806e-01 | 8.97012405e-01 | 2.78276683e+00 |
| 6291456  | 2.20772225e-01 | 4.42567622e-01 | 4.36865965e+00 |
| 12582912 | 2.24075881e-01 | 4.16621147e-01 | 5.22550209e+00 |
| 25165824 | 2.18828879e-01 | 3.66681124e-01 | 6.04026327e+00 |
| 50331648 | 2.10224991e-01 | 3.63676478e-01 | 5.41886640e+00 |
| 100663296| 1.94963261e-01 | 4.10170692e-01 | 6.85003693e+00 |

  
</details>

### Block Size Optimization

Here I tested block sizes from 16 up to 1024 threads for both the naive and efficient GPU scans at $2^{20}$ ($\approx$ 1M elements) averaged over three runs.

![Block sizes vs. runtime](images/block_size_runtimes.png)

There is a clear optimum for the naive scan at 128 threads per block with 1.873 ms at 128 and a U-shape around it. This looks like the typical balance between getting enough work per block to hide latency and not sacrificing occupancy. For the work-efficient scan the curve is flatter but still has a clear best region around 64 threads per block with 0.602 ms at 64 and 0.739 ms at 128. It climbs for larger block sizes. At 1024 threads it recovers a bit (0.734 ms), which suggests that the resource mix at that size on this GPU happens to be in a decent occupancy regime for this kernel, but it still doesn’t beat the 64-thread sweet spot. Overall, 128 is the right choice for naive and 64–128 is the safe choice for efficient on this hardware.

<details>
  <summary>Raw Data</summary>
  
| Block Size | GPU Naive Scan (ms) | GPU Efficient Scan (ms) |
| ---------- | ------------------- | ----------------------- |
| 16         | 4.70141             | 0.786624                |
| 32         | 3.67190             | 0.769472                |
| 64         | 3.03306             | 0.602016                |
| 128        | 1.87341             | 0.739008                |
| 256        | 1.95811             | 0.917312                |
| 512        | 2.37363             | 1.11926                 |
| 1024       | 3.07606             | 0.73424                 |
  
</details>

### Scan Runtimes

For this test I tested both power-of-two inputs and arbitrary non-power-of-two inputs ranging from $2^3$ (8) elements up to $2^{28}$ (268435456) elements on the various scan implementations.

![Scan vs. runtime](images/scan_runtimes.png)

If we look at the small end first, the CPU wins easily against all GPU implementations. By 100k elements Thrust manages to beat the CPU version and stays in the lead after that. The work-efficient scan stays well ahead of the naive scan across the entire range but never manages to outright beat either the CPU or Thrust versions. I didn't expect the CPU version to be doing so well across the board, but my AMD EPYC 9354 with 32 cores is a server grade CPU, while my 2000 series GPU is several generations old and not the most modern card anymore. In my mind this is a reasonable explanation why the CPU is doing so well.

<details>
  <summary>Raw Data</summary>
  
|   Array Length |   CPU Naive Runtime (ms) |   GPU Naive Runtime (ms) |   GPU Efficient Runtime (ms) |   GPU Thrust Runtime (ms) |
|---------------:|-------------------------:|-------------------------:|-----------------------------:|--------------------------:|
|              6 |              6.66667e-05 |                 0.477899 |                    0.0781653 |                 0.110155  |
|              8 |              3.33333e-05 |                 0.433152 |                    0.0722133 |                 0.0977387 |
|             12 |              0           |                 0.443392 |                    0.089504  |                 0.09664   |
|             16 |              3.33333e-05 |                 0.394048 |                    0.099712  |                 0.0703467 |
|             24 |              3.33333e-05 |                 0.495712 |                    0.136299  |                 0.0841173 |
|             32 |              3.33333e-05 |                 0.60416  |                    0.088064  |                 0.159488  |
|             48 |              3.33333e-05 |                 0.520939 |                    0.098336  |                 0.0723733 |
|             64 |              0.0001      |                 0.452405 |                    0.136629  |                 0.0670187 |
|             96 |              6.66667e-05 |                 0.529461 |                    0.089888  |                 0.079872  |
|            128 |              3.33333e-05 |                 0.935029 |                    0.094016  |                 0.0688747 |
|            192 |              0.0001      |                 0.636587 |                    0.107915  |                 0.129707  |
|            256 |              0.000166667 |                 0.606208 |                    0.0987627 |                 0.0928427 |
|            384 |              0.000266667 |                 1.10736  |                    0.115733  |                 0.0827307 |
|            512 |              0.0002      |                 0.59984  |                    0.159499  |                 0.0702613 |
|            768 |              0.000233333 |                 0.838997 |                    0.114688  |                 0.063776  |
|           1024 |              0.0003      |                 0.744448 |                    0.121291  |                 0.0635627 |
|           1536 |              0.0005      |                 1.04209  |                    0.115093  |                 0.0648533 |
|           2048 |              0.000633333 |                 0.817152 |                    0.114592  |                 0.170987  |
|           3072 |              0.0009      |                 0.831147 |                    0.164256  |                 0.0651627 |
|           4096 |              0.0012      |                 1.4413   |                    0.157472  |                 0.068032  |
|           6144 |              0.003       |                 0.864043 |                    0.313355  |                 0.0865707 |
|           8192 |              0.00243333  |                 0.887893 |                    0.149707  |                 0.0671573 |
|          12288 |              0.0036      |                 0.905643 |                    0.147808  |                 0.0911573 |
|          16384 |              0.0049      |                 1.37355  |                    0.147872  |                 0.0822507 |
|          24576 |              0.00716667  |                 1.03893  |                    0.200704  |                 0.069376  |
|          32768 |              0.00976667  |                 1.03372  |                    0.179776  |                 0.136875  |
|          49152 |              0.0227      |                 1.11339  |                    0.186219  |                 0.0657813 |
|          65536 |              0.0197333   |                 1.0289   |                    0.184949  |                 0.0702827 |
|          98304 |              0.0333667   |                 1.05363  |                    0.200736  |                 0.0645547 |
|         131072 |              0.0397333   |                 0.971445 |                    0.194539  |                 0.0657387 |
|         196608 |              0.0676333   |                 1.20444  |                    0.554411  |                 0.079264  |
|         262144 |              0.0811333   |                 1.55318  |                    0.336096  |                 0.195477  |
|         393216 |              0.136567    |                 1.41552  |                    0.540107  |                 0.0741867 |
|         524288 |              0.165767    |                 1.40222  |                    0.545141  |                 0.266101  |
|         786432 |              0.266       |                 1.49092  |                    0.496491  |                 0.237109  |
|        1048576 |              0.342       |                 1.70588  |                    0.584992  |                 0.225952  |
|        1572864 |              0.521933    |                 2.52433  |                    0.725547  |                 0.272747  |
|        2097152 |              0.671533    |                 2.61609  |                    0.714731  |                 0.30416   |
|        3145728 |              1.01453     |                 3.99886  |                    1.13101   |                 0.364576  |
|        4194304 |              1.46617     |                 7.07876  |                    1.37707   |                 0.384971  |
|        6291456 |              2.21737     |                10.0437   |                    5.01024   |                 0.507563  |
|        8388608 |              3.94937     |                12.8617   |                    5.49797   |                 0.575488  |
|       12582912 |              4.2712      |                19.0614   |                   10.252     |                 0.817376  |
|       16777216 |              5.5565      |                25.6322   |                   11.6855    |                 0.968672  |
|       25165824 |              8.36957     |                38.2471   |                   22.8252    |                 1.38563   |
|       33554432 |             11.0085      |                51.163    |                   24.2024    |                 1.80775   |
|       50331648 |             16.6486      |                79.1942   |                   45.7786    |                 3.07234   |
|       67108864 |             22.9489      |               105.132    |                   45.7996    |                 4.00884   |
|      100663296 |             36.7225      |               188.356    |                   89.5298    |                 5.36092   |
|      134217728 |             47.4704      |               217.434    |                   93.295     |                 8.06231   |

  
</details>

### Compaction Runtimes

Here I ran the compaction implementations over the same range of array sizes as in the scan tests, again averaging over three runs.

![Compaction vs. runtime](images/compact_runtimes.png)

At small sizes the CPU version is still the best option by a big margin and both GPU versions sit in the sub-millisecond range but don’t make sense given the launch overhead. As the arrays grow, the work-efficient compaction cleanly separates from the naive version and stays lower all the way through. The crossover with the CPU happens around 1M elements where at 786,432 elements the CPU is at 1.465 ms vs efficient at 1.889 ms, but at 1048576 elements the CPU is at 1.872 ms while efficient is at 1.580 ms. The gap only keeps growing and in the end at $2^{28}$ elements the efficient version is almost twice as fast as the CPU version. The naive version however never manages to beat even the CPU version (although it comes close to it towards the end). This is likely caused by the extra passes of the naive method that add on more latency.

<details>
  <summary>Raw Data</summary>
  
| Array Length | CPU Naive Runtime (ms) | GPU Naive Runtime (ms) | GPU Efficient Runtime (ms) |
|--------------|-------------------------|-------------------------|-----------------------------|
| 6            | 0.000633333             | 1.16342                 | 0.740416                    |
| 8            | 0.0006                  | 1.06086                 | 0.806944                    |
| 12           | 0.0006                  | 1.07486                 | 0.634955                    |
| 16           | 0.000533333             | 0.963936                | 0.594699                    |
| 24           | 0.000533333             | 1.10773                 | 0.750315                    |
| 32           | 0.000733333             | 1.26435                 | 0.948021                    |
| 48           | 0.0006                  | 1.05438                 | 0.765728                    |
| 64           | 0.000633333             | 1.01956                 | 0.751317                    |
| 96           | 0.0006                  | 1.12853                 | 0.676181                    |
| 128          | 0.000733333             | 1.1933                  | 0.600555                    |
| 192          | 0.0008                  | 1.11116                 | 0.707243                    |
| 256          | 0.000766667             | 1.0735                  | 0.661845                    |
| 384          | 0.00133333              | 1.2729                  | 0.644501                    |
| 512          | 0.001                   | 1.53061                 | 0.805259                    |
| 768          | 0.0014                  | 1.23926                 | 0.656139                    |
| 1024         | 0.0016                  | 1.79857                 | 0.664971                    |
| 1536         | 0.00196667              | 1.36533                 | 1.04387                     |
| 2048         | 0.00303333              | 1.34903                 | 0.824064                    |
| 3072         | 0.00396667              | 1.75286                 | 0.679339                    |
| 4096         | 0.0053                  | 1.52986                 | 0.649621                    |
| 6144         | 0.0119667               | 1.37739                 | 0.810453                    |
| 8192         | 0.0102333               | 1.56447                 | 0.662987                    |
| 12288        | 0.0170333               | 2.17771                 | 0.685152                    |
| 16384        | 0.0226667               | 1.53073                 | 0.700832                    |
| 24576        | 0.0402333               | 1.70816                 | 1.10944                     |
| 32768        | 0.0485667               | 1.54975                 | 0.864875                    |
| 49152        | 0.0777                  | 1.65796                 | 0.711979                    |
| 65536        | 0.0939667               | 1.8689                  | 0.707723                    |
| 98304        | 0.1375                  | 2.49754                 | 0.764885                    |
| 131072       | 0.222367                | 2.68748                 | 1.03817                     |
| 196608       | 0.586867                | 2.13043                 | 1.21746                     |
| 262144       | 0.5522                  | 1.96441                 | 1.12611                     |
| 393216       | 0.9594                  | 2.44464                 | 1.19506                     |
| 524288       | 1.04773                 | 2.91038                 | 1.19393                     |
| 786432       | 1.4651                  | 2.78883                 | 1.88878                     |
| 1048576      | 1.87157                 | 2.90023                 | 1.58005                     |
| 1572864      | 2.8029                  | 3.90234                 | 1.95265                     |
| 2097152      | 3.85527                 | 4.5592                  | 2.06318                     |
| 3145728      | 5.84377                 | 5.97142                 | 2.72289                     |
| 4194304      | 7.64867                 | 9.08323                 | 3.22245                     |
| 6291456      | 15.1688                 | 13.2843                 | 7.83981                     |
| 8388608      | 18.245                  | 17.2119                 | 8.05589                     |
| 12582912     | 22.7045                 | 24.3959                 | 15.5481                     |
| 16777216     | 30.7822                 | 31.6889                 | 16.2754                     |
| 25165824     | 46.4148                 | 47.568                  | 28.7986                     |
| 33554432     | 60.8522                 | 62.5561                 | 31.3732                     |
| 50331648     | 93.09                   | 96.0548                 | 58.7957                     |
| 67108864     | 135.012                 | 125.333                 | 62.9178                     |
| 100663296    | 199.087                 | 249.501                 | 115.834                     |
| 134217728    | 248.921                 | 287.466                 | 122.879                     |

  
</details>

### Compaction Subtiming

For this test I recorded the separate runtimes of map, scan and scatter during compaction at a few different array sizes.

![CPU Naive compaction subtiming](images/cpu_naive_compaction_subtiming.png)
![GPU Naive compaction subtiming](images/gpu_naive_compaction_subtiming.png)
![GPU Efficient compaction subtiming](images/gpu_efficient_compaction_subtiming.png)

If we compare the percentages on the naive CPU compaction, the distribution between the different steps is relatively balanced. Going from a ratio of 21% map, 36% scan and 43% scatter of the total time at the smallest displayed size to 33% map, 36% scan and 31% scatter at the largest displayed size. It seems that as the arrays get larger that map starts to take up a bigger share of the time. The naive GPU version clearly shows that the scan step is the slowest part. Going from a ratio of 8% map, 79% scan and 13% scatter of the total time at the smallest displayed size to 9% map, 84% scan and 7% scatter at the largest displayed size. The efficient GPU version also looks relatively balanced, with a ratio of 20% map, 32% scan and 48% scatter of the total time at the smallest displayed size to 24% map, 55% scan and 21% scatter of the total time at the largest size. We can see that scan increasingly grows to be the dominant factor as the arrays get larger. So the main takeaway is that on GPU, scan is where most of the time gets spent (particularly as the array size increases), while on CPU all three steps stay balanced for the most part.

<details>
  <summary>Raw Data</summary>

#### GPU Efficient

| Array Length | Total Runtime (ms) | Map Subtime (ms) | Scan Subtime (ms) | Scatter Subtime (ms) |
| ------------ | ------------------ | ---------------- | ----------------- | -------------------- |
| 192          | 0.288277           | 0.0577387        | 0.0928853         | 0.137653             |
| 6144         | 0.306838           | 0.063616         | 0.125611          | 0.117611             |
| 196608       | 0.611435           | 0.099136         | 0.399691          | 0.112608             |
| 1048576      | 0.917856           | 0.218859         | 0.502485          | 0.196512             |

#### GPU Naive

| Array Length | Total Runtime (ms) | Map Subtime (ms) | Scan Subtime (ms) | Scatter Subtime (ms) |
| ------------ | ------------------ | ---------------- | ----------------- | -------------------- |
| 192          | 0.7414717          | 0.062304         | 0.585013          | 0.0941547            |
| 6144         | 0.9921173          | 0.0566613        | 0.835424          | 0.100032             |
| 196608       | 1.521096           | 0.122987         | 1.29333           | 0.104779             |
| 1048576      | 1.99925            | 0.187253         | 1.67925           | 0.132747             |

#### CPU Naive

| Array Length | Total Runtime (ms) | Map Subtime (ms) | Scan Subtime (ms) | Scatter Subtime (ms) |
| ------------ | ------------------ | ---------------- | ----------------- | -------------------- |
| 192          | 0.000466667        | 0.0001           | 0.000166667       | 0.0002               |
| 6144         | 0.01126667         | 0.00156667       | 0.00476667        | 0.00493333           |
| 196608       | 0.586167           | 0.2273           | 0.2422            | 0.116667             |
| 1048576      | 1.870467           | 0.609            | 0.676467          | 0.585                |

  
</details>

### Sort Runtimes

For this test I compared my GPU radix sort against `std::sort` from 16 up to $2^{22}$ (4194304) elements.

![Sort subtiming](images/sort_runtimes.png)

`std::sort` is faster across the entire range in my runs. At the small end it’s not even close, for example `std::sort` is 0.083 ms while radix is 11.87 ms. At the largest array size, `std::sort` is 72.95 ms compared to 86.69 ms for radix, which is much closer in performance. This doesn't surprise me too much, since my radix sort is a straightforward implementation without shared-memory tuning. Right now it’s a good demonstration that a work-efficient full sort can work, but it’s not competitive with a highly optimized sort on a server grade CPU without further optimization.

<details>
  <summary>Raw Data</summary>
  
| Array Length | CPU std::sort (ms) | GPU Radix Sort (ms) |
| ------------ | ------------------ | ------------------- |
| 16           | 0.0004             | 8.15411             |
| 32           | 0.0007             | 8.62720             |
| 64           | 0.0018             | 8.89139             |
| 128          | 0.0033             | 9.20220             |
| 256          | 0.0067             | 9.88770             |
| 512          | 0.0165             | 10.1560             |
| 1024         | 0.0277             | 11.2896             |
| 2048         | 0.0424             | 10.1396             |
| 4096         | 0.0834             | 11.8725             |
| 8192         | 0.1704             | 12.2368             |
| 16384        | 0.4609             | 12.2040             |
| 32768        | 0.5909             | 12.6436             |
| 65536        | 1.2412             | 11.3041             |
| 131072       | 2.4341             | 24.8056             |
| 262144       | 4.7751             | 25.6925             |
| 524288       | 10.0772            | 27.2557             |
| 1048576      | 19.2595            | 31.8132             |
| 2097152      | 36.7872            | 46.9588             |
| 4194304      | 72.9481            | 86.6870             |

  
</details>

---

## Analysis

Overall, the results match what I expected. The CPU is unbeatable at small sizes and only starts slowing down after about one million elements and thrust is the clear winner above that because of its optimizations with shared memory, occupancy, and memory access. But it was surprising to see that the work-efficient GPU scan didn't manage to beat the CPU scan even for very large arrays. The siutation is relatively similar for compaction and the subtiming breakdown shows that scan is the main bottleneck on GPU while map and scatter are more important on CPU. For radix sort, the GPU version scales well but doesn't yet manage to beat the CPU without further optimizations.

### Nsight Observations

To analyze why the different GPU scans behave the way they do, I ran Nsight Compute in isolation on the scan kernels with an input size of $2^{26}$.

![Naive Nsight](images/naive_nsight.png)

For the naive scan, Nsight shows that every pass is heavily memory limited, with global memory bandwidth at over 85% while compute utilization hovers around 15%. Since the algorithm requires $\log_2(n)$ full passes plus an additional conversion step, the overall runtime grows with repeated bandwidth-limited kernel launches. This makes it slow even though each individual kernel launch is efficient in isolation.

![Efficient Nsight](images/efficient_nsight.png)

The work-efficient scan reduces the total work to $O(n)$, but the upsweep and downsweep phases each launch a series of kernels whose grid size halves every level. At the deeper levels this leaves only a few blocks active and most of the GPU idle but still the same launch and synchronization overhead per kernel. This results in good utilization at the top of the tree but bad utilization near the bottom which brings down the overall performance.

![Thrust Nsight](images/thrust_nsight.png)

Thrust only launches a couple of kernels, with the main one hitting over 90% memory throughput while avoiding the repeated global passes that slow down the custom implementations. This explains why Thrust quickly overtakes the CPU once the arrays become large, while the naive and work-efficient versions both struggle to keep up.

---

## Test output

Output from the pre-written test + a custom test comparing and validating my radix sort against `std::sort`

```
****************
** SCAN TESTS **
****************
    [  36  44  23  27   6   7  41   5  42   2  14  30  34 ...  25   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1.4204ms    (std::chrono Measured)
    [   0  36  80 103 130 136 143 184 189 231 233 247 277 ... 102713478 102713503 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1.3675ms    (std::chrono Measured)
    [   0  36  80 103 130 136 143 184 189 231 233 247 277 ... 102713409 102713440 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 7.02432ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 6.96771ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 1.38426ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 1.35088ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.410848ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.387072ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   2   3   3   0   3   3   3   2   2   0   0   2 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 8.2765ms    (std::chrono Measured)
    [   2   2   3   3   3   3   3   2   2   2   3   1   3 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 8.8411ms    (std::chrono Measured)
    [   2   2   3   3   3   3   3   2   2   2   3   1   3 ...   1   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 14.249ms    (std::chrono Measured)
    [   2   2   3   3   3   3   3   2   2   2   3   1   3 ...   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 3.45242ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 3.82582ms    (CUDA Measured)
    passed

*******************
** SORTING TESTS **
*******************
    [  36  44  23  27   6   7  41   5  42   2  14  30  34 ...  25  25 ]
==== cpu std::sort ====
   elapsed time: 73.1832ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
==== work-efficient radix sort ====
   elapsed time: 87.3332ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
    passed
```
