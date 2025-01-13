#import "@preview/charged-ieee:0.1.0": ieee
#import "@preview/algo:0.3.3": algo, i, d, comment, code

// Authors: Student Name, Surname, ID, email
// Maximum 4 pages (references do not count!!!)
// References.
// Platform and computing system description.
// GIT and instruction for the reproducibility
// Contribution: a section where each student describes his/her contribution.
// Format: use IEEE conference template (2-column format, main text 10pt)

#show: ieee.with(
  title: [GPU Computing Final Project],
  abstract: [
    This report evaluates three methods for transposing square, non-symmetric, dense matrices on CUDA-capable devices using NVIDIA's Cooperative Groups, compared against the cuBLAS library. Experiments, conducted on the University of Trento GPU Cluster, reveal that Cooperative Groups provide no significant performance benefits for this task. While cuBLAS is not always the fastest, it demonstrates greater stability in throughput, highlighting its robustness for dense matrix transposition.
  ],
  authors: (
    (
      name: "Ettore Saggiorato \n247178",
      organization: [Universita' di Trento],
      location: [Trento, Italy],
      email: "ettore.saggiorato@studenti.unitn.it"
    ),
  ),
  // index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.bib"),
)

#set table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    stroke: (_, y) => (
       top: if y <= 1 { 0.5pt } else { 0pt },
       bottom: 0.5pt,
   )
)

= Introduction
The aim of this project is to develop an efficient algorithm for transposing square, dense matrices on a GPU, utilizing NVIDIA's Cooperative Groups@nvidia:coop. The matrices considered in this work are dense, non-symmetric, square matrices with dimensions $cal(N) times cal(N)$, where $cal(N)$ is a power of 2, ranging from to $2^2$ to $2^15$.

Dense matrices are central to numerous applications in fields such as computational mathematics, numerical analysis, machine learning, data science, physics, engineering, and finance. Matrix transposition, a fundamental operation in linear algebra, is widely used across these domains for tasks such as data manipulation and optimization.

As matrix transposition is a core operation, optimizing its performance is critical for enhancing computational efficiency. This report investigates various strategies for leveraging NVIDIA's Cooperative Groups to optimize matrix transposition on the GPU, and evaluates their effectiveness in improving performance.

== Instructions for Reproducibility

The code and instructions for reproducing the results discussed in this report are available on #link("https://github.com/sa1g/gpu-computing")[GitHub].

= State of the Art

The state-of-the-art solution for transposing dense matrices on NVIDIA GPUs is provided by cuBLAS (@nvidia:cublas). This library supports dense matrices and vectors and offers GPU-accelerated, optimized routines for fundamental linear algebra operations. Matrix transposition is implemented through two key functions, tailored for single and double precision operations:

- *`cublasSgeam`*: The primary function for single precision matrix transposition. It allows users to specify matrix operations, including transposition, by utilizing the `CUBLAS_OP_T` operation flag.
- *`cublasDgeam`*: The corresponding function for double precision matrix transposition, analogous to cublasSgeam.

This project focuses on single precision operations.

== Cooperative Groups
Cooperative Groups is a programming model introduced to provide more flexible and structured ways of manaigng and synchronizing threads. It allows to organize threads into hierarchical groups, enabling fine-grained control over parallelism and synhronization at different levels, like withing a warp, a block, or across the entire grid. Cooperative Groups enable thread organization by creating logical grouping of threads beyond the default block and grid structure, flexible synchronizatoin by synchronizing subsets of threads within a group instead of all theads in a block and scalability by supporting scalable algorithms by enabling cooperation between threads at multiple levels.
Given the foundational nature of this work, the primary focus will be on experimenting with different algorithms and benchmarking their performance rather than introducing groundbreaking advancements in the field.

= Contribution and Methodology

Given the capabilities of Cooperative Groups (CG), which provide APIs for granular thread management, and the relatively straightforward requirements of a dense square matrix transpose (as opposed to more complex operations like full reductions), it is anticipated that kernels developed with Cooperative Groups for this task may face performance limitations.

This project explores two primary approaches:
- *Intra-block synchronization using Cooperative Groups*: This method is expected to perform comparably to traditional implementations using `__syncthreads`, as the synchronization scope is confined to threads within the same block.
- *Inter-block synchronization using Cooperative Groups*: This method is expected to underperform due to the significant overhead introduced by grid-wide synchronization, which is inherently more complex and time-consuming.

To evaluate these approaches, four kernels were developed:
- One reference kernel for copying operations.
- Three kernels for matrix transposition.

== Synchronization Strategies and Kernel Descriptions
1. *Kernels with intra-block synchronization*\
  Three kernels, described in @algo:copyCoop, @algo:transposeSharedCoop, and @algo:transposeSharedCoopNoConflict, utilize the `cg::sync(cta)` function from the Cooperative Groups (CG) API to synchronize threads within a block. Here, `cta` represents the cooperative thread block group, which encompasses all threads within the block. Functionally, `cg::sync(cta)` operates similarly to the traditional `__syncthreads()` mechanism.
2. *Kernel with inter-block synchronization*\
  The kernel described in @algo:transposeInterBlock demonstrates the challenges associated with inter-block synchronization for matrix transposition. This kernel employs `cudaLaunchCooperativeKernel` to synchronize across the entire grid. Due to the inherent constraints of inter-block synchronization, this kernel is limited to matrices with dimensions $cal(N) times cal(N)$ with $cal(N) in [2^2, 2^5]$. The narrow range is a limitation of the current implementation due to the absence of subtitling.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: "CUDA Copy Coop",  
  algo(
    // header: "transposeCoalescedCoop",
    // parameters: ([#math.italic("n")],),
    comment-prefix: [#sym.triangle.stroked.r ],
    // comment-styles: (fill: rgb(20%, 20%, 20%)),
    indent-size: 15pt,
    indent-guides: 1pt + gray,
    // row-gutter: 5pt,
    column-gutter: 5pt,
    inset: 5pt,
    stroke: 0.5pt + black,
    row-gutter: 0.6em,
    main-text-styles: (size: 0.8em),
    comment-styles: (size: 0.8em),
    line-number-styles: (size: 0.8em),
    // fill: none,
  )[
  *set* shared tile[tile_dim][tile_dim]\
  let cta $<-$ cg::this_thread_block()\
  let x $<- "blockIdx.x" star "tile_dim" + "threadIdx.x"$\
  let y $<- "blockIdx.y" star "tile_dim" + "threadIdx.y"$\
  let width $<- "gridDim.x" star "tile_dim"$#footnote[`x`,`y` and `width` are common in all CUDA implementations, for clarity they will be omitted in other pseudocodes.]\

  for $i <- 0$ to tile_dim *by* block_rows do#i:\
      $"tile"["tIdx".y + i]["tIdx".x] <- "idata"[(y + i) * "width" + x]$#d\
  cs::sync(cta)\
  for $i <- 0$ to tile_dim *by* block_rows do#i:\
      $"odata"[(y + i) * "width" + x] = "tile"["tIdx".y + i]["tIdx".x]$;
]
)<algo:copyCoop>

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: "CUDA Transpose Coalesced Coop (CC)",  
  algo(
    // header: "transposeCoalescedCoop",
    // parameters: ([#math.italic("n")],),
    comment-prefix: [#sym.triangle.stroked.r ],
    // comment-styles: (fill: rgb(20%, 20%, 20%)),
    indent-size: 15pt,
    indent-guides: 1pt + gray,
    // row-gutter: 5pt,
    column-gutter: 5pt,
    inset: 5pt,
    stroke: 0.5pt + black,
    row-gutter: 0.6em,
    main-text-styles: (size: 0.8em),
    comment-styles: (size: 0.8em),
    line-number-styles: (size: 0.8em),
    // fill: none,
  )[
    *set* shared tile[tile_dim][tile_dim]\
    let cta $<-$ cg::this_thread_block()\
    for $i <- 0$ to tile_dim *by* block_rows do#i:\
      $"tile"["tIdx".y + i]["tIdx".x] <- "idata"[(y + i) * "width" + x]$#d\
    cs::sync(cta)\
    for $i <- 0$ to tile_dim *by* block_rows do#i:\
      $"odata"[(y + i) * "width" + x] = "tile"["tIdx".x][("tIdx".y + i)]$;
  ]
)<algo:transposeSharedCoop>


#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: "CUDA Transpose Coalesced No Conflict (CNBCC)",  
  algo(
    // header: "transposeCoalescedNoConflictCoop",
    // parameters: ([#math.italic("n")],),
    comment-prefix: [#sym.triangle.stroked.r ],
    // comment-styles: (fill: rgb(20%, 20%, 20%)),
    indent-size: 15pt,
    indent-guides: 1pt + gray,
    // row-gutter: 5pt,
    column-gutter: 5pt,
    inset: 5pt,
    stroke: 0.5pt + black,
    row-gutter: 0.6em,
    main-text-styles: (size: 0.8em),
    comment-styles: (size: 0.8em),
    line-number-styles: (size: 0.8em),
    // fill: none,
  )[
    *set* shared tile[tile_dim][tile_dim*+1*]
    #comment[Same as @algo:transposeSharedCoop, but with padding to avoid bank conflicts]
  ]
)<algo:transposeSharedCoopNoConflict>


#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: "CUDA Transpose Inter Block",  
  algo(
    // header: "transposeInterBlock",
    // parameters: ([#math.italic("n")],),
    comment-prefix: [#sym.triangle.stroked.r ],
    // comment-styles: (fill: rgb(20%, 20%, 20%)),
    indent-size: 15pt,
    indent-guides: 1pt + gray,
    // row-gutter: 5pt,
    column-gutter: 5pt,
    inset: 5pt,
    stroke: 0.5pt + black,
    row-gutter: 0.6em,
    main-text-styles: (size: 0.8em),
    comment-styles: (size: 0.8em),
    line-number-styles: (size: 0.8em),
    // fill: none,
  )[
    *set* shared tile[tile_dim][tile_dim+1]\
    let grid $<-$ cg::this_grid()\

    if (x < width and y < width):#i\
      $"tile"["tIdx".y]["tIdx".x] <- "idata"[y * "width" + x]$#d\
    
    grid.sync()\

    if (x < width and y < width):#i\
      $"odata"[y * "width" + x] <- "tile"["tIdx".x]["tIdx".y]$#d\
  ]
)<algo:transposeInterBlock>



= System Description and Experiments
The experiments were conducted on the University of Trento GPU Cluster, equipped with NVIDIA A30 GPUs. Detailed hardware and software specifications are presented in @table:hardware and @table:software. Compilation of the code was performed using the nvcc compiler with the flags `-O2`, `-code=sm_80`, and `-arch=compute_80`. Hyper-threading is enabled on the cluster.

To mitigate the impact of operating system scheduling and hyper-threading noise, each experiment was repeated 10 times, with each kernel being launched 100 times per experiment. The results were averaged to ensure reliability. This setup is designed to minimize inter-program noise by maintaining consistent priority levels assigned by the operating system during each benchmark. By averaging across multiple runs, the setup also reduces variability caused by potential fluctuations in OS priorities.

The experiments were conducted on matrices with dimensions ranging from $2^2 times 2^2$ up to $2^15 times 2^15$.

#figure(
  table(
  columns: (auto, auto, auto),
  table.header([*Component*], [*Version*], [*Docs*]),
  [GPU], [NVidia A30], [@nvidia:A30],
  [Max. theor. bandwidth], [$933$ GB/s], [@nvidia:bandwidth],
  [CPU], [Intel XEON Gold 6238R], [@intel:cpu],
),
  caption: [Hardware Description.],
) <table:hardware>

#figure(
  table(
  columns: (auto, auto, auto),
  table.header([*Component*], [*Version*], [*Docs*]),
  [OS: Rocky Linux], [8.7], [@rocky],
  [Scheduler: Slurm], [20.11.9], [@slurm],
  [CUDA], [12.1.66], [@nvidia:cuda],
  [cuBLAS], [12.1.0],[@nvidia:cublas],
  [C++], [14], [@cpp], 
),
  caption: [Software Description#footnote[CMake was used for local development.].],
) <table:software>



= Results
== Intra-block Synchronization
The results for intra-block synchronization, shown in @fig:copy, indicate no significant benefit from using Cooperative Groups for this purpose. A similar trend is observed in @fig:shared and @fig:CNBC, confirming that intra-block synchronization with Cooperative Groups does not enhance performance for matrix transposition. The experimental results are summarized in @table:intra.

Interestingly, cuBLAS performs marginally slower than the presented kernels for matrix side sizes in the range $[2^5, 2^11]$.





== Inter-block Synchronization
The results in @fig:inter demonstrate that inter-block synchronization provides no performance benefits for matrix transposition. The data in @table:inter further reinforces this conclusion.

#figure(
  image("../plots/CUDA Matrix Copy.png", width: 100%),
  caption: [CUDA Matrix Copy Coop. Copy with shared memory and no bank conflicts (`C Shared`), Copy with shared memory, no bank conflicts, cooperative (`C Coop`, @algo:copyCoop). The two kernels using shared memory exhibit similar behavior. The average standard deviation (std) for the `C Coop` kernel is $7.12$Gbps, for `C Shared` it is $8.79$Gbps, and for `C Naive` it is $8.36$Gbps.]
)<fig:copy>

#figure(
  image("../plots/CUDA Matrix Transpose Coalesced.png", width: 100%),
  caption: [CUDA Matrix Transpose Coalesced Coop. Copy with shared memory (`C Shared`), Transpose with shared memory (`T Coalesced`) and Transpose Cooperative with shared memory (T CoalescedCoop, @algo:transposeSharedCoop). Notably, `T cublas` outperforms all other kernels. Unexpectedly, `T CoalescedCoop` performs significantly worse than `T Coalesced`, despite the only difference being the synchronization method.]
)<fig:shared>

#figure(
  image("../plots/CUDA Matrix Transpose Coalesced No Bank Conflicts.png", width: 100%),
  caption: [CUDA Matrix Transpose Coalesced No Bank Conflicts Coop. Transpose with Shared Memory, Coalesced No Bank Conflict (T CNBC) and Transpose Cooperative with Shared Memory, Coalesced No Bank Conflict (`T CNBC Coop` @algo:transposeSharedCoopNoConflict). For $cal(N) >= 2^12$, `T cublas` is the best performer. However, within $2^5 <=cal(N) <= 2^11$, `T CNBC Coop` outperforms the others.]
)<fig:CNBC>

#figure(
  image("../plots/CUDA Transpose Inter Block.png", width: 100%),
  caption: [CUDA Transpose Inter Block. T interblock (@algo:transposeInterBlock). The plot clearly shows that inter-block synchronization is not an effective choice for dense matrix transposition.]
)<fig:inter>

#figure(
  table(
    table.header(
    [*Size*], [*InterBlock*], [*cuBLAS*]
  ),
    // stroke: ,
    columns: (auto, auto, auto, ),
    $2^2$, $0.0098 plus.minus	0.0008$, math.bold($0.0106 plus.minus	0.0012$),
    $2^3$, $0.0392 plus.minus	0.0033$, math.bold($0.0425 plus.minus	0.005$),
    $2^4$, $0.1568 plus.minus	0.0132$, math.bold($0.1712 plus.minus	0.0202$),
    $2^5$, $0.6288 plus.minus	0.0553$, math.bold($0.6949 plus.minus	0.0734$),
    [*avg std*], $0.01815$, $0.02495$,
  ),
  caption: [Inter-Block Synchronization effective bandwidths. Best results are in *bold*. All measurements are in Gbps.]
)<table:inter>


#set text(size: 5pt)
#figure(
  table(
    table.header(
    [*Size*], [*Coalesced*], [*Coalesced\ Cooperative*], [*Coalesced\ No Bank\ Conflict*], [*Coalesced\ No Bank\ Conflict\ Cooperative*], [*cuBLAS*]
  ),
    // stroke: ,
    columns: (auto, auto, auto, auto, auto, auto),
    $2^2$, $0.01 plus.minus 0$ , "-", $0.01 plus.minus 0$, $0.01 plus.minus 0$, $0.01 plus.minus 0$,
    $2^3$, $0.05 plus.minus 0.01$ , "-", $0.05 plus.minus 0.01$, $0.05 plus.minus 0.01$, $0.04 plus.minus 0.01$,
    $2^4$, $0.18 plus.minus 0.02$ , "-", $0.18 plus.minus 0.02$, "-", $0.17 plus.minus 0.02$,
    $2^5$, $0.68 plus.minus 0.07$, math.underline($0.7 plus.minus 0.06$), $0.74 plus.minus 0.07$, math.underline(math.bold($0.77 plus.minus 0.08$)), $0.69 plus.minus 0.07$,
    $2^6$, $2.67 plus.minus 0.25$, math.underline($2.73 plus.minus 0.25$), $2.88 plus.minus 0.26$, math.bold(math.underline($2.98 plus.minus 0.33$)), $2.62 plus.minus 0.26$,
    $2^7$, $10.52 plus.minus 0.96$, math.underline($10.82 plus.minus 1.16$), $11.41 plus.minus 1.03$, math.bold(math.underline($11.78 plus.minus 1.13$)), $10.24 plus.minus 1.01$,
    $2^8$, $38.1 plus.minus 3.16$, math.underline($38.58 plus.minus 3.79$), $43.95 plus.minus 4.26$, math.bold(math.underline($45.43 plus.minus 4.27$)), $40.22 plus.minus 3.89$,
    $2^9$, $119.73 plus.minus 11.15$, math.underline($120.31 plus.minus 10.45$), $157.87 plus.minus 14.52$, math.bold(math.underline($161.59 plus.minus 14.27$)), $145.37 plus.minus 12.97$,
    $2^10$, math.underline($245.4 plus.minus 22.72$), $243.11 plus.minus 19.54$, $438.17 plus.minus 34.48$, math.bold(math.underline($454.57 plus.minus 34.61$)), $392.82 plus.minus 31.06$,
    $2^11$, math.underline($327.11 plus.minus 24.79$), $304.7 plus.minus 24.4$, $633.16 plus.minus 40.31$,math.bold(math.underline($642.05 plus.minus 40.29$)), $553.35 plus.minus 28.08$,
    $2^12$, $393.42 plus.minus 35.48$, math.underline($408.88 plus.minus 54.57$), $734.62 plus.minus 16.46$, math.underline($737.71 plus.minus 17.19$), (math.bold($767.68 plus.minus 13.34$)),
    $2^13$, math.underline($544.33 plus.minus 1.37$), $504.85 plus.minus 1.44$, $753.4 plus.minus 2.82$, $753.16 plus.minus 3.06$, math.bold($816.1 plus.minus 5.01$),
    $2^14$, math.underline($549.14 plus.minus 0.97$), $509.65 plus.minus 0.74$, $749.55 plus.minus 1.49$, $748.17 plus.minus 2.32$, math.bold($824.54 plus.minus 1.43$),
    $2^15$, math.underline($534.8 plus.minus 44.6$), $497.79 plus.minus 42.56$, $739.21 plus.minus 11.23$, math.underline($740.21 plus.minus 10.56$), math.bold($818.46 plus.minus 5.42$),
    [*avg std*], $10.39$, $14.45$, $9.06$, $9.85$,$7.32$,
  ),
  caption: [Intra-Block Synchronization effective bandwidths. *C*: coalesced, *CC*: coalesced cooperative, *CNBC* and *CNBCC*:  Coalesced No Bank Conflicts (+coop). Best results are in *bold*, the best for each algorithm type are #underline[underline]. All measurements are in Gbps.]
)<table:intra>
#set text(size: 10pt)

\
\
// \
= Conclusions
As anticipated, the application of Cooperative Groups for square dense matrix transposition does not yield significant performance advantages. This holds true for both intra-block synchronization and inter-block synchronization. While cuBLAS does not consistently outperform the custom kernels in terms of raw throughput, it exhibits a consistently lower average standard deviation. This characteristic makes cuBLAS not only competitive but also the "most stable" solution in terms of performance reliability.

These results shed light on the challenges of optimizing matrix transposition. Specifically, the straightforward nature of dense matrix transposition requires less granular control, rendering advanced thread management techniques, such as those offered by Cooperative Groups, less impactful for this task.

Future research could investigate more advanced memory management techniques, such as TMA@nvidia:tma, or optimize memory access patterns to improve kernel efficiency further.
