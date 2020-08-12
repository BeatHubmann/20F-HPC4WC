
# Group 02 Project: Implementing a Diffusion Stencil on a Sphere using MPI

## Group Members

Shruti Nath & Beat Hubmann

## Project Synopsis

Implement an MPI stencil code on a cubed-sphere grid, allowing to run on a full sphere.
Investigate weak and strong scalability of the code on moderate node counts.

## Running the Code

### Command Line Options

* ```--nx``` Number of gridpoints in x-direction. Typical value: 120
* ```--ny``` Number of gridpoints in y-direction. Typical value: 120
* ```--nz``` Number of gridpoints in z-direction. Typical value: 60
* ```--num_iter``` Number of iterations. Typical value: 1020
* ```--num_halo``` Number of halo-pointers in x- and y-direction. Default value: 2
* ```--plot_result``` Save plots of the result? Default value: False
* ```--verify``` Save verifications plots and output subdomains for up to 8 x/y gridpoints plus 2*2 halo points? No diffusion, overrides num_iter, plot_result options. Default value: False

### Typical Run Command for Production
```mpirun -n 6 python3 ./stencil3d-mpi.py --nx=120 --ny=120 --nz=60 --num_iter=1020 --plot_result=true```

### Typical Run Command for Verification
```mpirun -n 6 python3 ./stencil3d-mpi.py --nx=8 --ny=8 --nz=3 --num_iter=1020 --verify=true```

## Contents of Submission

### Root Directory

* ```cubedspherepartitioner.py``` Contains CubedSpherePartitioner class required for main program
* ```stencil3d-mpi.py``` Main cubed sphere program
* ```stencil2d-mpi.py``` Slightly modified 2D counterpart to main cubed sphere program for reference
* ```requirements.txt``` Python ```pip``` requirements file
* ```LICENSE``` MIT License file
* ```README.md``` This file

### scaling_analysis Directory

* ```submit_strong_scaling.sh``` Slurm submission script for strong scaling analysis runs on CSCS Piz Daint
* ```submit_weak_scaling.sh``` Slurm submission script for weak scaling analysis runs on CSCS Piz Daint
* ```cubed_sphere_strong_scaling.*.out``` Strong scaling analysis run outputs from CSCS Piz Daint
* ```cubed_sphere_weak_scaling.*.out``` Weak scaling analysis run outputs from CSCS Piz Daint

### report Directory

* ```group02_project_report.pdf``` Project report

