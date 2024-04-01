#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lineq.h"
#include "rootfinding.h"
#include "utils.h"

int main(int, char **) {

  //   P1. Implement and compare the behavior of the bisection, false position,
  //   Secant, Newton,
  // and Steffensen methods to find solutions of
  // a) x2 = 2
  // b) x3 = 0
  // c) x1/3 = 0
  // A single program with cases for choice of method. problem, and initial
  // approximation might facilitate numerically and/or graphical analysis of
  // error and stopping criteria, but it is usually a good idea to begin with
  // the most basic version of one particular method and problem, for instance,
  // false position applied to x2 = 2. P2.

  // Implement the Horner algorithm for
  // Newton’s method to find all zeros of a polynomial. Demonstrate the results
  // using the degree 4 Tchebyshev polynomial T4(x) = 8x4 − 8x2 + 1 and the
  // degree 4 Legendre polynomial P4(x) = x4 − 6 7 x2 + 3 35 . Soon we will see
  // why the zeros of T4 are an optimal choice of 4 points at which to evaluate
  // a function for polynomial interpolation on the reference interval [−1, 1].
  // and why the zeros of P4 are an optimal choice of 4 points at which to
  // evaluate a function for numerical integration on the reference interval
  // [−1, 1]. (In both cases, the results may be applied to a more general
  // interval [a, b] by a linear shift and scaling.

  // P3. Find an initial
  // approximation x0 that produces an absolute error of E3 = |x3 − √2| = 0.001,
  // of the third approximation to √2 when the xj , j = 1, 2, . . . are found by
  // Newton’s method applied to x2 − 2 = 0 as in problem 1a

  std::cout << "\n\033[1mProblem 1:\033[0m\n\n";

  std::cout << "\033[1mProblem 1a: x^2 = 2\033[0m\n";

  double tolerance = 1e-10;
  int maxIterations = 10000;

  double lowerBound = -0.3;
  double upperBound = 3.0;
  double x0 = 3;

  run_all_rootfinding_methods([](double x) { return x * x - 2; },
                              [](double x) { return 2 * x; }, lowerBound,
                              upperBound, x0, tolerance, maxIterations);

  std::cout << "\033[1mProblem 1b: x^3 = 0\033[0m\n";

  lowerBound = -2;
  upperBound = 9.0 / 5.0;
  x0 = 3;

  run_all_rootfinding_methods([](double x) { return x * x * x; },
                              [](double x) { return 3 * x * x; }, lowerBound,
                              upperBound, x0, tolerance, maxIterations);

  std::cout << "\033[1mProblem 1c: x^(1/3) = 0\033[0m\n";

  lowerBound = -4;
  upperBound = 9.0 / 5.0;
  x0 = 3;

  run_all_rootfinding_methods(
      [](double x) {
        if (x < 0) {
          return -pow(std::abs(x), 1.0 / 3.0);
        } else {
          return pow(x, 1.0 / 3.0);
        }
      },
      [](double x) {
        if (x < 0) {
          return -(1.0 / 3.0) * pow(std::abs(x), -2.0 / 3.0);
        } else {
          return (1.0 / 3.0) * pow(x, -2.0 / 3.0);
        }
      },
      lowerBound, upperBound, x0, tolerance, maxIterations);

  std::cout
      << "Notes: All converge on 1a. False position doesn't converge fast "
         "enough to actually reach tolerance on 1b, since only one side ends "
         "up updating. 1c makes Secant too slow, and makes Newton and "
         "Steffensen diverge since the derivative is much lower than the value "
         "at the point. Bisection and false position are great on 1c though. "
         "Newton is of course fastest on 1a.";

  std::cout << "\n\033[1mProblem 2:\033[0m\n\n";

  std::cout << "T4(x) = 8x^4 - 8x^2 + 1\n";

  std::vector<double> coefficients = {1, 0, -8, 0, 8};

  x0 = 0.5;
  tolerance = 1e-10;
  maxIterations = 10000;
  int numIterations = 0;

  std::vector<double> roots =
      newton_horners(coefficients, x0, tolerance, maxIterations, numIterations);

  printVec("Roots of T4(x)", roots);

  std::cout << "P4(x) = x^4 - 6/7 x^2 + 3/35\n";

  coefficients = {3.0 / 35.0, 0, -6.0 / 7.0, 0, 1};

  std::vector<double> roots2 =
      newton_horners(coefficients, x0, tolerance, maxIterations, numIterations);

  printVec("Roots of P4(x)", roots2);

  std::cout << "\n\033[1mProblem 3:\033[0m\n\n";

  // To find that exact error, we can apply a derivative free method to a
  // max-iterations limited newtons method where error is our desired output.

  double x3 = sqrt(2);

  lowerBound = 1;
  upperBound = 4;
  tolerance = 1e-10;
  maxIterations = 100000;
  numIterations = 0;
  x0 = 3;

  double desiredInitialGuess = bisection(
      [](double newton_initial_guess) {
        int numIterations =
            0; // Discarded in this case, just to track iterations.
        double E3 =
            0.001; // Error value we want to find at x_3 for our initial guess.
        double newtons_x_3 = newton(
            [](double x) { return x * x - 2; }, [](double x) { return 2 * x; },
            newton_initial_guess, 1e-10, 3,
            numIterations); // Run newtons for 3 itters, will return x_3
        double error = std::abs(newtons_x_3 - sqrt(2)); // Calculate error
        return error - E3; // And we want to solve for this to be 0, since we
                           // want our error to be equal to E3.
      },
      lowerBound, upperBound, tolerance, maxIterations, numIterations);

  std::cout << "Desired initial guess for E3 = 0.001: " << desiredInitialGuess
            << " Based on " << numIterations << " itterations.\n";

  std::cout << "Confirmed by running newtons method with this initial guess "
               "giving error: "
            << std::abs(newton([](double x) { return x * x - 2; },
                               [](double x) { return 2 * x; },
                               desiredInitialGuess, tolerance, 3,
                               numIterations) -
                        sqrt(2))
            << " with " << numIterations << " itterations.\n";
}
