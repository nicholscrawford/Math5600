# Numerical Analysis Code - Written for MATH 5600 @ Utah

## Description

Code for the numerical analysis class. Solving linear systems of equations using elimination with pivoting. Solving systems of linear equations with iterative methods (great for sparse problems). Finding roots by various methods, and finding all roots by Horner's + Newtons.

## Installation

Build the project 

```sh
mkdir build
cd build
cmake ..
make
```

Run tests (in build dir).

```
./tests
```

Run linear equation solver which demos the methods (in build dir).

```
./linear_solver
```

Run the rootfinding solver which demos methods (in build dir).
```
./rootfinding
```

[Linear Solver Example Output](example_output.txt)

[Rootfinding Example Output](rootfinding_sample_output.md)