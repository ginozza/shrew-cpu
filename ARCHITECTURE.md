# Architecture: shrew-cpu

`shrew-cpu` is the default execution backend for the Shrew framework. It provides a multi-threaded CPU implementation of all tensor operations, serving as both a functional baseline and a high-performance engine for non-GPU environments.

## Core Concepts

- **CpuBackend**: The concrete implementation of `shrew-core::Backend`. It manages data as `Vec<T>` in host memory.
- **Parallelism**: Utilizes `rayon` to parallelize element-wise operations and reductions across available CPU cores.
- **Reference Implementation**: All operations defined in `shrew-core` must be implemented here first. This backend is used for testing and validating other backends (like CUDA).

## File Structure

| File | Description | Lines of Code |
| :--- | :--- | :--- |
| `ops.rs` | Implements the `UnaryOp`, `BinaryOp`, `ReduceOp`, and other operation traits defined in `core`. Contains the logic for math operations (matmul, add, relu, etc.) using Rayon. | 1821 |
| `lib.rs` | Defines the `CpuBackend` struct, memory allocation logic, and the `CpuDevice` identifier. | 426 |
