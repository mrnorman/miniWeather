# CUDASTF port of the miniWeather benchmark

CUDASTF is a task-based programming model implemented as a header-only C++
library. It is shipped as part of NVIDIA's CCCL project
<https://github.com/nvidia/cccl>.

After identifying physical fields as `logical data`, and expressing
computations as tasks which operate on these fields, CUDASTF automatically
manages data, and infers concurrency opportunities.

This example illustrates how to automatically generate CUDA kernels based on
CUDASTF's `ctx.parallel_for`. It also demonstrates how these constructs can be
seamlessly spread over multiple GPUs, and how to leverage CUDA graphs to
improve performance for small problem sizes.
