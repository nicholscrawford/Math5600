# Numerical Analysis Code - Written for MATH 5600 @ Utah

## Description

Code for the numerical analysis class. Solving linear systems of equations using elimination with pivoting. Solving systems of linear equations with iteritive methods, (great for sparse probelms).

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

Output with n = 3.

**Hilbertian Matrix:**

1.00000e+00	5.00000e-01	3.33333e-01	

5.00000e-01	3.33333e-01	2.50000e-01	

3.33333e-01	2.50000e-01	2.00000e-01	



**LU Decomp, L:**

1.00000e+00	0.00000e+00	0.00000e+00	

5.00000e-01	1.00000e+00	0.00000e+00	

3.33333e-01	1.00000e+00	1.00000e+00	


**LU Decomp, U:**

1.00000e+00	5.00000e-01	3.33333e-01	

0.00000e+00	8.33333e-02	8.33333e-02	

0.00000e+00	-1.38778e-17	5.55556e-03	


**LUP Decomp, L:**

1.00000e+00	0.00000e+00	0.00000e+00	

5.00000e-01	1.00000e+00	0.00000e+00	

3.33333e-01	1.00000e+00	1.00000e+00	


**LUP Decomp, U:**

1.00000e+00	5.00000e-01	3.33333e-01	

0.00000e+00	8.33333e-02	8.33333e-02	

0.00000e+00	-1.38778e-17	5.55556e-03	


**LUP Decomp, P:**

1.00000e+00	0.00000e+00	0.00000e+00	

0.00000e+00	1.00000e+00	0.00000e+00	

0.00000e+00	0.00000e+00	1.00000e+00	


**Cholesky Decomp, L:**

1.00000e+00	0.00000e+00	0.00000e+00	

5.00000e-01	2.88675e-01	0.00000e+00	

3.33333e-01	2.88675e-01	7.45356e-02	


**LU Error for single column:**

3.8858e-16	-2.4425e-15	2.4980e-15	

**LU Error for linear combination:**

0.0000e+00	0.0000e+00	0.0000e+00	

**LUP Error for single column:**

3.8858e-16	-2.4425e-15	2.4980e-15	

**LUP Error for linear combination:**

0.0000e+00	0.0000e+00	0.0000e+00	

**Cholesky Error for single column:**

-4.9960e-16	2.6645e-15	-2.4980e-15	

**Cholesky Error for linear combination:**

0.0000e+00	0.0000e+00	0.0000e+00	
