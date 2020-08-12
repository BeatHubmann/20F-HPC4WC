# 20F-HPC4WC
Final project for the course [High Performance Computing for Weather and Climate](http://vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=138658&semkez=2020S&ansicht=KATALOGDATEN&lang=de), FS2020 @ETHZ

## Project: Implementing a Diffusion Stencil on a Sphere using MPI

### Task
Extend an MPI stencil code from a rectangular domain to a cubed-sphere grid, allowing to run on a full sphere.
Investigate weak and strong scalability of the code on moderate node counts.

## Course description

### Synopsis

State-of-the-art weather and climate simulations rely on large and complex software running on supercomputers. This course focuses on programming methods and tools for understanding, developing and optimizing the computational aspects of weather and climate models. Emphasis will be placed on the foundations of parallel computing, practical exercises and emerging trends such as heterogeneous comput


### Objective

After attending this course, students will be able to:
- understand a broad variety of high performance computing concepts relevant for weather and climate simulations
- work with weather and climate simulation codes that run on large supercomputers

### Content

- HPC Overview:
  - Why does weather and climate require HPC?
  - Today's HPC: Beowulf-style clusters, massively parallel architectures, hybrid computing, accelerators
  - Scaling / Parallel efficiency
  - Algorithmic motifs in weather and climate

- Writing HPC code:
  - Data locality and single node efficiency
  - Shared memory parallelism with OpenMP
  - Distributed memory parallelism with MPI
  - GPU computing
  - High-level programming and domain-specific languages
