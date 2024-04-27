#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "combinations.h"
#include "lineq.h"
#include "rootfinding.h"
#include "utils.h"

int main(int, char **) {
  int n = 20; // Number of points for interpolation and quadrature
  double sum, tmp;

  std::vector<double> equally_spaced_points(n);
  for (int i = 0; i < n; i++) {
    equally_spaced_points[i] = -1.0 + 2.0 * i / (n - 1.0);
  }

  // Define f from probelm definition
  auto f = [](double x) { return 1.0 / (1.0 + 25.0 * x * x); };

  auto coefficients = get_tchebyshev_coefficients(n);
  printVec("Tchebyshev Polynomial Coefficients", coefficients);
  int num_iterations = 0;     // Initialize number of iterations
  double initial_guess = 0.5; // Set an initial guess
  double tolerance = 1e-21;   // Set tolerance
  int max_iterations = 10000; // Set maximum number of iterations
  auto roots = newton_horners(coefficients, initial_guess, tolerance,
                              max_iterations, num_iterations);
  printVec("Tchebyshev Polynomial Roots", roots);

  std::vector<double> coeff_tchebyshev =
      get_interpolation_coefficients(roots, f);
  printVec("Coefficients of Interpolation Polynomial", coeff_tchebyshev);

  std::vector<double> coeff_equally_spaced =
      get_interpolation_coefficients(equally_spaced_points, f);
  //   printVec("Coefficients of Interpolation Polynomial (Equally Spaced)",
  //            coeff_equally_spaced);

  // Compare accuracy of interpolation using Tchebyshev nodes vs. equally spaced
  double n_points = 10000;
  double total_tchebyshev_error = 0;
  double total_equally_spaced_error = 0;
  for (int i = 0; i < n_points; i++) {
    double x = -1 + 2 * i / (n_points - 1);
    total_tchebyshev_error += std::abs(f(x) - polynomial(coeff_tchebyshev, x));
    total_equally_spaced_error +=
        std::abs(f(x) - polynomial(coeff_equally_spaced, x));
  }

  std::cout << "Average Tchebyshev Error: " << total_tchebyshev_error / n_points
            << std::endl;
  std::cout << "Average Equally Spaced Error: "
            << total_equally_spaced_error / n_points << std::endl;

  auto coefficients_legendre = get_legendre_coefficients(n);
  printVec("Legendre Polynomial Coefficients", coefficients_legendre);
  auto roots_legendre =
      newton_horners(coefficients_legendre, initial_guess, tolerance,
                     max_iterations, num_iterations);

  printVec("Legendre Polynomial Roots", roots_legendre);

  auto quadrature_weights_legendre = get_quadrature_weights(roots_legendre, f);
  printVec("Quadrature Weights", quadrature_weights_legendre);

  auto quadrature_weights_equally_spaced =
      get_quadrature_weights(equally_spaced_points, f);
  //   printVec("Quadrature Weights (Equally Spaced)",
  //            quadrature_weights_equally_spaced);

  // Print sum of quadrature weights, x_legendre
  //   double sum_weights = 0;
  //   for (auto weight : quadrature_weights) {
  //     sum_weights += weight;
  //   }
  //   std::cout << "Sum of Quadrature Weights: " << sum_weights << std::endl;

  //   std::cout << "Quadrature of f(x) = 1/(1 + 25x^2) on [-1,1] = "
  //             << quadrature(roots_legendre, quadrature_weights_legendre, f)
  //             << std::endl;
  //   std::cout
  //       << "Quadrature of f(x) = 1/(1 + 25x^2) on [-1,1] (Equally Spaced) = "
  //       << quadrature(equally_spaced_points,
  //       quadrature_weights_equally_spaced, f)
  //       << std::endl;
  double true_integtal = (2 / 5) * atan(5);
  std::cout << "Error of Quadrature (Tchebyshev): "
            << std::abs(true_integtal - quadrature(roots_legendre,
                                                   quadrature_weights_legendre,
                                                   f))
            << std::endl;
  std::cout << "Error of Quadrature (Equally Spaced): "
            << std::abs(true_integtal -
                        quadrature(equally_spaced_points,
                                   quadrature_weights_equally_spaced, f))
            << std::endl;
}