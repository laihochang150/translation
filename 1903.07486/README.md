Dissecting the NVIDIA Turing T4 GPU via Microbenchmarking



Contents

1 Low-level details make a difference 5

2 How Turing encodes instructions 9

2.1 Control information . . . . . . . . . . . . . . . . . . . . . . . . . . 10

2.2 Processing Blocks and Schedulers . . . . . . . . . . . . . . . . . . 12

2.3 Instruction word format . . . . . . . . . . . . . . . . . . . . . . . 13

3 Memory hierarchy 15

3.1 L1 data cache . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

3.2 Unified L2 cache . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

3.3 Instruction cache hierarchy . . . . . . . . . . . . . . . . . . . . . 25

3.4 Constant memory hierarchy . . . . . . . . . . . . . . . . . . . . . 28

3.5 Registers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

3.6 Shared memory . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

3.7 Global memory . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

3.8 TLBs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

4 Instruction latency and throughput 39

4.1 Native instructions . . . . . . . . . . . . . . . . . . . . . . . . . . 39

4.2 Atomic operations . . . . . . . . . . . . . . . . . . . . . . . . . . 40

4.3 New Tensor Core instructions . . . . . . . . . . . . . . . . . . . . 43

4.4 Arithmetic performance . . . . . . . . . . . . . . . . . . . . . . . 44

4.5 Performance throttling . . . . . . . . . . . . . . . . . . . . . . . . 46

5 Conclusions 51

Bibliography 61

Contents 63



通过微基准测试的方法详细分析了NVIDIA Turing T4 GPU的架构特性，发表在arXiv上，旨在通过精心设计的基准测试来探究Turing架构的内部细节。研究涵盖了多个方面，包括SM（Streaming Multiprocessor）的微架构、内存系统、指令执行效率等。文中揭示了Turing架构在访存方面的改进，如L1/Shared Cache的带宽和访问模式，以及如何通过微基准测试来优化代码，避免银行冲突（bank conflict）。此外，论文还讨论了Turing架构与前代Kepler架构的对比，特别是在Shared Memory的Bank数量和宽度上。通过这些深入的分析，研究者能够更好地理解Turing架构的性能特点，为开发者提供优化建议。



由 Citadel 的高性能计算研发团队撰写，作者包括 Zhe Jia、Marco Maggioni、Jeffrey Smith 和 Daniele Paolo Scarpazza。报告通过微基准测试深入剖析了 NVIDIA 的 Turing T4 GPU 架构，并与之前的 Pascal P4 GPU 进行了比较。文章提供了关于 NVIDIA Turing T4 GPU 架构的深入分析，通过详细的微基准测试揭示了其性能特点和改进之处，为高性能计算和深度学习应用提供了宝贵的参考。



背景知识

文章指出，GPU 制造商快速更新设计且不愿公开微架构细节，给追求 GPU 最高性能的软件设计者带来了挑战。作者去年对 Volta GPU 架构进行了剖析，并在 NVIDIA 的 GPU 技术大会（GTC2018）上展示了他们的发现。随着 2018 年 8 月 Turing 架构的推出，作者更新了他们的研究，重点关注 T4 GPU，这是一款面向推理应用的低功耗、小尺寸板卡。



研究方法

作者使用微基准测试来剖析 GPU 架构，这种方法允许他们深入理解 GPU 的指令编码、内存层次结构和性能行为。他们通过编写和运行特定的测试代码来测量各种硬件特性和性能指标。



关键结论

性能提升：T4 GPU 在低精度操作数上的吞吐量远高于 P4 GPU，特别是在 Tensor Cores 的性能上。Turing 架构引入了新指令，更简洁地表达矩阵数学运算，并且在指令空间上进行了扩展。

内存层次结构：Turing 的内存层次结构深度与 Volta 相同，但某些缓存级别的大小通常是 Pascal 的两倍。作者对 T4 的每个内存层次组成部分进行了基准测试，发现与 P4 前代产品相比有显著的整体性能提升。

时钟限制对性能的影响：研究了时钟限制如何影响计算密集型工作负载，特别是当它们达到功率或热限制时。

指令编码：Turing 和 Volta 使用相同的指令编码格式，与 Pascal 和 Maxwell 不同。Turing 和 Volta 使用 128 位来编码指令及其相关调度控制信息，而之前的架构使用 64 位来编码指令信息，并在每几条指令后使用一个单独的 64 位字来编码控制信息。

控制信息：控制信息从 Kepler 架构开始出现，用于编码编译器做出的指令调度决策。Turing 和 Volta 的控制信息包含在每个 128 位指令中，而 Pascal 和 Maxwell 的控制信息则与多个指令打包在一起。

处理块和调度器：Turing 的流式多处理器（SM）被划分为四个处理块，每个块包含一个专用的 warp 调度器和分发单元。作者通过实验发现，Turing 和 Volta 的 warp 映射到调度器的规则与 Volta 相同。

指令字格式：Turing 和 Volta 的指令字格式与 Pascal、Maxwell 和 Kepler 不同，它们将操作码放在第一个 64 位字的最低有效位，而之前的架构将操作码放在最高有效位。

内存层次结构：Turing 的内存层次结构包括多个缓存级别和翻译后备缓冲区（TLBs）。作者详细描述了每个缓存级别的大小、属性和性能，并与 Volta、Pascal、Maxwell 和 Kepler 架构进行了比较。

指令缓存层次结构：Turing 和 Volta 有三个级别的指令缓存（L0、L1、L2），而 Pascal、Maxwell 和 Kepler 有 L1、L1.5 和 L2。作者通过实验检测了每个缓存级别的大小，并确定了它们在 GPU 架构块中的分布。

常量内存层次结构：Turing 的常量内存层次结构与之前的架构相比没有显著变化，但作者通过实验验证了其性能特性。

寄存器：Turing 和 Volta 使用 16,384 个 32 位元素的物理寄存器文件，组织成两个银行，每个银行有两个 32 位端口。作者还讨论了寄存器银行冲突对指令延迟的影响。

共享内存：Turing 的共享内存提供了低延迟和高带宽。作者对共享内存的性能进行了表征，包括在竞争条件下的性能。

全局内存：作者测量了全局内存的实际带宽，并与理论极限进行了比较。

TLBs：Turing 和其他架构一样，L1 数据缓存由虚拟地址索引，L2 数据缓存由物理地址索引。作者通过实验验证了这一点，并测量了 TLBs 的性能。



实验结果

L1 数据缓存：T4 的 L1 数据缓存命中延迟为 32 个时钟周期，带宽为 58.83 字节/周期/SM，比 P4 高出 3.7 倍。

L2 数据缓存：T4 的 L2 数据缓存命中延迟约为 188 个时钟周期，带宽比 P4 高出 30%。

指令缓存层次结构：Turing 的 L0 指令缓存私有于每个调度器，L1 指令缓存私有于每个 SM，而 L2 指令缓存由所有 SM 共享。

常量内存层次结构：Turing 的常量内存层次结构包括三个级别的缓存，具有不同的延迟和大小。

寄存器银行冲突：Turing 的寄存器银行冲突会导致指令延迟增加，尤其是在所有三个源寄存器都映射到同一个银行时。

共享内存：Turing 的共享内存延迟在不同架构中相对较低，带宽在 V100 和 P100 之后排名第三。

全局内存：T4 的全局内存带宽为 220 GiB/s，实际与理论带宽比为 68.8%，低于 P4 的 84.4%。



结论

文章总结了 Turing 架构与之前 NVIDIA 架构的比较，强调了 T4 和 P4 GPU 之间的比较。Turing 在指令编码、内存层次结构和处理单元行为方面与 Volta 代显示连续性，并与 Kepler 和更老的代一起代表了一个显著的背离。Turing 继续了调度器与核心比率增长的趋势，这一趋势与指令吞吐量的增长相关。Turing 和 Volta 引入的 L0 指令缓存减轻了与更长指令相关的惩罚。改进的 L1 数据缓存提供了更低的延迟和更高的带宽。从 Pascal 的 4 个单端口寄存器银行变为 2 个双端口银行有助于防止银行冲突。



相关链接：

https://zhuanlan.zhihu.com/p/502089583
