**Problem 1:**


**Problem 1a: x^2 = 2**

Bisection Method: 1.41421			 Number of Itterations:28

False Position Method: 1.41421			 Number of Itterations:25

Secant Method: 1.41421			 Number of Itterations:8

Newton Method: 1.41421			 Number of Itterations:5

Steffensen Method: 1.41421			 Number of Itterations:8

**Problem 1b: x^3 = 0**

Bisection Method: 0.000195313			 Number of Itterations:10

False Position Method: 0.0140137			 Number of Itterations:10000

Secant Method: 0.000368582			 Number of Itterations:24

Newton Method: 0.000400972			 Number of Itterations:22

Steffensen Method: 0.000349383			 Number of Itterations:58

**Problem 1c: x^(1/3) = 0**

Bisection Method: 1.64731e-31			 Number of Itterations:101

False Position Method: -7.5462e-31			 Number of Itterations:73

Secant Method: -0.00894747			 Number of Itterations:10000

Newton Method: -inf			 Number of Itterations:10000

Steffensen Method: 1.06855e+23			 Number of Itterations:10000

Notes: All converge on 1a. False position doesn't converge fast enough to actually reach tolerance on 1b, since only one side ends up updating. 1c makes Secant too slow, and makes Newton and Steffensen diverge since the derivative is much lower than the value at the point. Bisection and false position are great on 1c though. Newton is of course fastest on 1a.

**Problem 2:**



T4(x) = 8x^4 - 8x^2 + 1

**Roots of T4(x)**

3.8268e-01	-9.2388e-01	-3.8268e-01	9.2388e-01	

P4(x) = x^4 - 6/7 x^2 + 3/35

**Roots of P4(x)**

3.3998e-01	-8.6114e-01	-3.3998e-01	8.6114e-01	



**Problem 3:**



Desired initial guess for E3 = 0.001: 3.0774e+00 Based on 25 itterations.

Confirmed by running newtons method with this initial guess giving error: 1.0000e-03 with 3 itterations.

