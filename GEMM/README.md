# Parallel Computing Tutorial

This repository introduces several optimization techniques that can be applied to improve the parallelism of matrix multiplication. The techniques include loop unrolling, loop reordering, loop tiling, multithreading, SIMD programming, and CUDA programming. Each technique is implemented in a separate source file (*.cpp inside [src/](src/)) and all techniques use the common header file [matmul.h](include/matmul.h). In addition, we also provide a benchmark.cpp and a Makefile to compile and benchmark the different matrix multiplication implementations.

## Learning Resources

If your want to learn more about optimization techniques of efficient deep learning, please check out lectures on [TinyML and Efficient Deep Learning Computing](https://efficientml.ai/).

## Directory Structure

Here is an outline of the main files and directories:

```ccs
├── src
│   ├── loop_unrolling.cpp
│   ├── loop_reordering.cpp
│   ├── loop_tiling.cpp
│   ├── naive.cpp
│   ├── multithreading.cpp
│   ├── SIMD_programming.cpp
│   └── cuda_programming.cpp
├── include
│   └── matmul.h
├── benchmark.cpp
└── Makefile
```

## Directory Structure

### Prerequisites

To compile and run the examples, you will need:

* A C++ compiler (GCC, Clang, MSVC, etc.)
* CUDA Toolkit (optional, only if you want to enable CUDA programming.)

### Compilation

To compile the code, navigate to the repository root and execute:

```bash
make -j
```

This will produce an executable named `benchmark`.

### Running the Benchmarks

To run the benchmark, execute:

```bash
./benchmark
```

The benchmark will run matrix multiplication using all techniques and output the time taken by each technique.

You can also measure the performance improvement achieved by a specific technique with an extra argument:

Available arguments are:

* CUDA_SGEMM
* CUDA
* SIMD_programming
* loop_reodering
* loop_tiling
* loop_unrolling
* multithreading

For example, to measure the performance improvement of the CUDA kernel:

```bash
./benchmark CUDA
```

## Contributions

We welcome contributions! If you have a suggestion, bug report, or want to contribute to the code, feel free to open an issue or create a pull request. Please make sure your code follows the current code style.

## License

This project is open-source and is licensed under the MIT License.

## Contact

If you have any questions or suggestions, feel free to open an issue or reach out to the maintainers.

## Acknowledgements

We would like to thank everyone who contributed to this repository, providing feedback and bug reports, making this project possible. I run this benchmark result on my NVIDIA GeForce RTX 2070.

<!-- benchmark_results -->
| Kernel                              |  ms |
|:------------------------------------|----------:|
| 1: Naive                            |   `19443` |
| 2: Unrolling                        |  `3042`   |
| 3: Reordering                       |  `1315`   |
| 3: Tiling                           |  `1102`   |
| 4: Multithreading                   |  `6630`   |
| 5: Navie Cuda                       | `1290`    |
| 6: Cuda SGEMM                       | `14`      |

<!-- benchmark_results -->