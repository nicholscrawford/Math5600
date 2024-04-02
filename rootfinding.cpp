#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

/**
 * Performs the bisection method to find the root of a given function within a
 * specified interval.
 *
 * @param function The function for which the root is to be found.
 * @param lowerBound The lower bound of the interval.
 * @param upperBound The upper bound of the interval.
 * @param tolerance The tolerance level for the root approximation.
 * @param maxIterations The maximum number of iterations allowed.
 * @param[out] numIterations The number of iterations performed.
 * @return The approximate root of the function within the specified interval.
 */
double bisection(std::function<double(double)> function, double lowerBound,
                 double upperBound, double tolerance, int maxIterations,
                 int &numIterations) {

  double functionAtLowerBound = function(lowerBound);
  double functionAtUpperBound = function(upperBound);
  double midpoint = (lowerBound + upperBound) / 2;
  double functionAtMidpoint = function(midpoint);

  numIterations = 0;

  while (std::abs(functionAtMidpoint) > tolerance &&
         numIterations < maxIterations) {
    midpoint = (lowerBound + upperBound) / 2;
    functionAtMidpoint = function(midpoint);
    if (functionAtMidpoint == 0) {
      return midpoint;
    }
    if (functionAtLowerBound * functionAtMidpoint < 0) {
      upperBound = midpoint;
      functionAtUpperBound = functionAtMidpoint;
    } else {
      lowerBound = midpoint;
      functionAtLowerBound = functionAtMidpoint;
    }
    numIterations++;
  }

  return midpoint;
}

/**
 * Performs the false position method to find the root of a given function
 * within a specified interval.
 *
 * @param function The function for which the root is to be found.
 * @param lowerBound The lower bound of the interval.
 * @param upperBound The upper bound of the interval.
 * @param tolerance The desired tolerance for the root.
 * @param maxIterations The maximum number of iterations allowed.
 * @param[out] numIterations The number of iterations performed.
 * @return The estimated root of the function.
 */
double falsePosition(std::function<double(double)> function, double lowerBound,
                     double upperBound, double tolerance, int maxIterations,
                     int &numIterations) {

  double functionAtLowerBound = function(lowerBound);
  double functionAtUpperBound = function(upperBound);
  double midpoint =
      (functionAtUpperBound * lowerBound - functionAtLowerBound * upperBound) /
      (functionAtUpperBound - functionAtLowerBound);
  double functionAtMidpoint = function(midpoint);

  numIterations = 0;

  while (std::abs(functionAtMidpoint) > tolerance &&
         numIterations < maxIterations) {
    midpoint = (functionAtUpperBound * lowerBound -
                functionAtLowerBound * upperBound) /
               (functionAtUpperBound - functionAtLowerBound);
    functionAtMidpoint = function(midpoint);
    if (functionAtMidpoint == 0) {
      return midpoint;
    }
    if (functionAtLowerBound * functionAtMidpoint < 0) {
      upperBound = midpoint;
      functionAtUpperBound = functionAtMidpoint;
    } else {
      lowerBound = midpoint;
      functionAtLowerBound = functionAtMidpoint;
    }
    numIterations++;
  }
  return midpoint;
}

/**
 * Performs the secant method to find the root of a given function.
 *
 * @param function The function for which the root is to be found.
 * @param x0 The first initial guess.
 * @param x1 The second initial guess.
 * @param tolerance The desired tolerance for the root.
 * @param maxIterations The maximum number of iterations allowed.
 * @param[out] numIterations The number of iterations performed.
 * @return The estimated root of the function.
 */
double secant(std::function<double(double)> function, double x0, double x1,
              double tolerance, int maxIterations, int &numIterations) {

  double f0 = function(x0);
  double f1 = function(x1);
  double x2 = 0.0;
  double f2 = 0.0;

  numIterations = 0;

  while (std::abs(f1) > tolerance && numIterations < maxIterations) {
    x2 = (f1 * x0 - f0 * x1) / (f1 - f0);
    f2 = function(x2);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = f2;
    numIterations++;
  }

  return x2;
}

/**
 * Performs Newton's method to find the root of a given function.
 *
 * @param function The function for which the root is to be found.
 * @param derivative The derivative of the function.
 * @param x0 The initial guess.
 * @param tolerance The desired tolerance for the root.
 * @param maxIterations The maximum number of iterations allowed.
 * @param[out] numIterations The number of iterations performed.
 * @return The estimated root of the function.
 */
double newton(std::function<double(double)> function,
              std::function<double(double)> derivative, double x0,
              double tolerance, int maxIterations, int &numIterations) {

  double f0 = function(x0);
  double fPrime0 = derivative(x0);
  double x1 = 0.0;
  double f1 = 0.0;

  numIterations = 0;

  while (std::abs(f0) > tolerance && numIterations < maxIterations) {
    x1 = x0 - f0 / fPrime0;
    f1 = function(x1);
    x0 = x1;
    f0 = f1;
    fPrime0 = derivative(x0);
    numIterations++;
  }

  return x1;
}

/**
 * Performs the Steffensen method to find the root of a given function.
 *
 * @param function The function for which the root is to be found.
 * @param x0 The initial guess.
 * @param tolerance The desired tolerance for the root.
 * @param maxIterations The maximum number of iterations allowed.
 * @param[out] numIterations The number of iterations performed.
 * @return The estimated root of the function.
 */
double steffensen(std::function<double(double)> function, double x0,
                  double tolerance, int maxIterations, int &numIterations) {

  double g0 = 0.0;
  double x1 = 0.0;
  double f0 = function(x0);

  numIterations = 0;

  while (std::abs(f0) > tolerance && numIterations < maxIterations) {
    g0 = function(x0 + f0) / function(x0) - 1;
    x1 = x0 - f0 / g0;
    x0 = x1;
    f0 = function(x0);
    numIterations++;
  }

  return x0;
}

/**
 * Runs all rootfinding methods on a given function.
 *
 * @param function The function for which the root is to be found.
 * @param derivative The derivative of the function.
 * @param lowerBound The lower bound of the interval.
 * @param upperBound The upper bound of the interval.
 * @param x0 The initial guess.
 * @param tolerance The desired tolerance for the root.
 * @param maxIterations The maximum number of iterations allowed.
 */
void run_all_rootfinding_methods(std::function<double(double)> function,
                                 std::function<double(double)> derivative,
                                 double lowerBound, double upperBound,
                                 double x0, double tolerance,
                                 int maxIterations) {

  int numIterations = 0;
  std::cout << "Bisection Method: "
            << bisection(function, lowerBound, upperBound, tolerance,
                         maxIterations, numIterations)
            << "\t\t\t Number of Itterations:" << numIterations << "\n";

  std::cout << "False Position Method: "
            << falsePosition(function, lowerBound, upperBound, tolerance,
                             maxIterations, numIterations)
            << "\t\t\t Number of Itterations:" << numIterations << "\n";

  std::cout << "Secant Method: "
            << secant(function, lowerBound, upperBound, tolerance,
                      maxIterations, numIterations)
            << "\t\t\t Number of Itterations:" << numIterations << "\n";

  std::cout << "Newton Method: "
            << newton(function, derivative, x0, tolerance, maxIterations,
                      numIterations)
            << "\t\t\t Number of Itterations:" << numIterations << "\n";

  std::cout << "Steffensen Method: "
            << steffensen(function, x0, tolerance, maxIterations, numIterations)
            << "\t\t\t Number of Itterations:" << numIterations << "\n";
}

/**
 * Calculates the value of a polynomial function at a given point.
 *
 * @param coefficients The coefficients of the polynomial in ascending order of
 * degree.
 * @param x The value at which to evaluate the polynomial.
 * @return The value of the polynomial at the given point.
 */
double polynomial(std::vector<double> coefficients, double x) {
  double result = 0.0;
  for (int i = 0; i < coefficients.size(); i++) {
    result += coefficients[i] * pow(x, i);
  }
  return result;
}

/**
 * Calculates the derivative of a polynomial function at a given point.
 *
 * @param coefficients The coefficients of the polynomial in descending order.
 * @param x The point at which to evaluate the derivative.
 * @return The value of the derivative at the given point.
 */
double polynomial_prime(std::vector<double> coefficients, double x) {
  double result = 0.0;
  for (int i = 1; i < coefficients.size(); i++) {
    result += i * coefficients[i] * pow(x, i - 1);
  }
  return result;
}

/**
 * Performs polynomial division on the given coefficients with the specified
 * root of the polynomial.
 *
 * @param coefficients The coefficients of the polynomial in descending order of
 * degree.
 * @param root The root at which to perform the division.
 * @return The quotient of the polynomial division.
 */
std::vector<double> polynomialDivision(const std::vector<double> &coefficients,
                                       double root) {
  int n = coefficients.size();
  std::vector<double> quotient(n - 1);
  std::vector<double> coefficients_copy = coefficients;

  // Initialize the quotient vector with zeros
  for (int i = 0; i < n - 1; ++i) {
    quotient[i] = 0;
  }

  for (int i = n - 2; i >= 0; --i) {
    quotient[i] = coefficients_copy[i + 1];
    coefficients_copy[i] = quotient[i] * root + coefficients_copy[i];
  }

  return quotient;
}

/**
 * @brief Performs Newton-Horner's method to find the roots of a polynomial.
 *
 * This function takes a vector of polynomial coefficients, an initial guess for
 * the root, a tolerance value, a maximum number of iterations, and a reference
 * to store the number of iterations performed. It iteratively applies
 * Newton-Horner's method to find the all roots of the polynomial. The roots are
 * stored in a vector and returned.
 *
 * @param coefficients The vector of polynomial coefficients.
 * @param x0 The initial guess for the root.
 * @param tolerance The tolerance value to determine convergence.
 * @param maxIterations The maximum number of iterations allowed.
 * @param[out] numIterations The reference to store the number of iterations
 * performed.
 * @return std::vector<double> The vector containing the roots of the
 * polynomial.
 */
std::vector<double> newton_horners(std::vector<double> coefficients, double x0,
                                   double tolerance, int maxIterations,
                                   int &numIterations) {

  double f0 = polynomial(coefficients, x0);
  double fPrime0 = polynomial_prime(coefficients, x0);

  std::vector<double> roots = {};

  numIterations = 0;

  while (coefficients.size() > 1) {

    while (std::abs(f0) > tolerance && numIterations < maxIterations) {
      x0 = x0 - f0 / fPrime0;
      f0 = polynomial(coefficients, x0);
      fPrime0 = polynomial_prime(coefficients, x0);
      numIterations++;
    }
    roots.push_back(x0);
    coefficients = polynomialDivision(coefficients, x0);
    f0 = polynomial(coefficients, x0);
    fPrime0 = polynomial_prime(coefficients, x0);

    numIterations = 0;
  }

  return roots;
}