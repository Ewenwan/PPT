# Performance Prediction Toolkit (PPT)
Predict the performance of physics codes


GPU 一开始是用来渲染图片的, 然后逐渐发展为强力的co-processor来做复杂计算. 因此, 大家对GPU model就有了迫切需求, 需要它来评估各种应用下的性能, 各种GPU 架构下的性能. 但是因为架构太复杂, modeling和predicting runtime 都变得很难, 也被研究了很长一段时间.

在期间, 有人提出analytical途径的model. 优点在于速度快, 但是准确性不高.  另外, 大家广泛接受的是cycle-accurate simulators, 主要因为对这种建模方式的认可并且使用简单. 比如GPGPU-Sim 就是个标准cycle-accurate simulator. 这种simulator通常更加精确, 但是仿真速度慢, 数据量有限制.

本文提出的PPT-GPU是为了解决上述困难, 它是个可扩展, 准确的性能预测GPU框架. PPT-GPU属于analytical 和 cycle-accurate 的混合模型, 能针对各种GPU架构和workloads预测runtime behavior. 它能处理大很多的dataset(不用牺牲准确度, 扩展性和模拟速度). 对于源码, 我们(指作者团队, 下同)把计算类的workload都分成更小的并行sub-workload, 这些sub-workload可以同时在SM上运行, 我们通过预测sub-workload在一个SM上的runtime 来获取整个workload的performance.

PPT-GPU是开源项目(PPT)的一部分, [G. Chennupati et al., Performance Prediction Toolkit(PPT), Los Alamos National Laboratory (LANL), 2017,
https://github.com/lanl/PPT.]   PPT已经把硬件和中间件建模成了带参数的模型, 可以直接吃source code, 预测runtime. PPT需要靠Simian, Simian 是个并行离散事件仿真引擎. [N. Santhi et al., “The simian concept: Parallel discrete event simulationwith interpreted languages and just-in-time compilation,” in 2015 Winter Sim Conf., 2015.]
之前用PPT概念提出的旧GPU model预测的runtime忽略了一些现代GPU里常用的东西, 所以结果不准确. 本次为了提升旧model的表现, 考虑了以下因素:


现代GPU的各种memory hierarchies

GPU指令的latency, 这些指令的latency仅与指令本身和GPU family有关.

与真实GPU和GPGPU-Sim验证, 使用一组10个kernel 的RODINIA, Parboil benchmark

结果显示新的PPT-GPU 大幅提升了准确性, 而且还可扩展.

这个model 假设global memory instruction 绕过L1, L2 cache, 从memory 取data. 由于考虑到cache也会很大程度影响性能, 我们也展示出增加L2 cache带来的影响(准确性方面). 如果说要建个预测L1/L2cache性能的模型那就超过本文范畴了, 不过正在进行中.  对于L2cache, 我们用了NVDIA Visual Profiler tool来预测L2 cache hit rate, 然后把结果输入到PPT-GPU里, 这样的话我们能控制L2的影响, 使预测结果不会偏离的太多.

GPU Architecture

通常一块GPU包括一些SM(streaming multiprocessors), 每个SM可以认为是一个单独的处理器. 再说仔细点, 每个SM有自己的指令buffer, scheduler, 和dispatch单元. 而且, 每个SM有一些计算资源处理算术运算. 所有SM都共用一个芯片之外的global memory. 但太经常访问global memory 会造成性能下降, 因为延迟太大. 然后架构师们就提出了on-chip L1 & L2 cache. 一般一个SM有一个L1 cache, 所有SM共用一个L2cache. 除此之外每个SM还有共有shared memory里的一小部分以便线程交互.

以CUDA的术语来说, 所有thread都由同一个kernel产生, 组成一个grid, 这些grid使用同一个global memory. grid是由很多thread block组成的, 每个block是由多个group(32 thread)组成的, 这个group也叫做warp. Grid 和 block 属于一种CUDA kernel里的logic view. warp采用SIMD方式, 意思是所有thread(同一个warp里的)任何时候都执行同一个指令, 这也是为了使并行效率达到最高. 每个SM都包含一些warp schedulers, 它们的作用是把单个或多个指令派给指定warp. CUDA的程序员有一些额外的优化性能的手段: 可以查看grid/block的数量对某个app的影响.
PPT-GPU

在PPT-GPU model里, 预测runtime被分为两步来做: 1. Static Pre-Characterization, 这时 PPT-GPU 和app源码交互得到tasklist(与architecture无关). 2. Prediction Model, 一个带参数的GPU model  预测这个tasklist下的runtime.



## Authors
1. Gopinath (Nath) Chennupati (gchennupati@lanl.gov)
2. Nanadakishore Santhi (nsanthi@lanl.gov)
3. Stephen Eidenbenz (eidenben@lanl.gov)
4. Robert Joseph (Joe) Zerr (rzerr@lanl.gov)
5. Massimiliano (Max) Rosa (maxrosa@lanl.gov)
6. Richard James Zamora (rjzamora@lanl.gov)
7. Eun Jung (EJ) Park (ejpark@lanl.gov)
8. Balasubramanya T (Balu) Nadiga (balu@lanl.gov)
###### Non-LANL Authors
9. Jason Liu (luix@cis.fiu.edu)
10. Kishwar Ahmed
11. Mohammad Abu Obaida
12. Yehia Arafa (yarafa@nmsu.edu)

## Installation

###### Dependencies
PPT depends on another LANL licensed open source software package named SIMIAN PDES located at https://github.com/pujyam/simian.

**Simian** relies on python package _greenlet_
_pip install greenlet_

PPT installation is simple, just checkout the code as follows
> git clone https://github.com/lanl/PPT.git

## Usage
Let's assume the PPT is cloned into your home directory (/home/user/PPT)
In the _code_ directory PPT is organized in three main layers:

1. **hardware** -- contains the models for various CPUs and GPUs, interconnects,
2. **middleware** -- contains the models for MPI, OpenMP, etc.
3. **apps** -- contains various examples. These examples are the stylized pseudo (mini) apps for various open sourced physics codes.

### Runnning PPT in _Serial mode_

For example, we run SNAP simulator in serial with one of the PPT hardware models as follows:

> cd ~/PPT/code/apps/snapsim

> python snapsim-orig.py in >> out

where, _in_ and _out_ in the above command are input and output files of SNAPsim.

## Classification
PPT is Unclassified and contains no Unclassified Controlled Nuclear Information. It abides with the following computer code from Los Alamos National Laboratory
* Code Name: Performance Prediction Toolkit, C17098
* Export Control Review Information: DOC-U.S. Department of Commerce, EAR99
* B&R Code: YN0100000

## License
&copy 2017. Triad National Security, LLC. All rights reserved.
 
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.
 
All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
 
Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions.
Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file:
This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
