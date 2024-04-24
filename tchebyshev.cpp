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
  // This programming project leverages two codes you wrote previously (Horner’s
  // Algorithm and LU factorization) to implement optimized polynomial
  // interpolation (Tchebyshev nodes) and numerical integration (Gaussian
  // Quadrature). Most of the new code (not many lines) involves ‘front ends’ to
  // generate coefficients of two important families of polynomials that arise
  // in best approximation methods for the two most important norms, Tchebyshev
  // Poly- nomials, Tn(x) for the ∞−norm (minimax/uniform approximation) and
  // Legendre Polyno- mials Pn(x) for the 2−norm (least squares approximation).
  // The amount of new code is surprisingly small. It mostly consists of
  // generating coefficients of two families of polynomials. The first family is
  // the Tchebyshev polynomials, 1, x, 2x2 − 1, 4x3 −3x, . . . using the
  // two-term recurrence formula Tn+1(x) = 2xTn(x)−Tn−1(x), n > 1 with initial
  // conditions T0(x) = 1, T1(x) = x. These formulas arise from an alternate
  // definition, Tn(x) = cos(n arccos(x)), x ∈ [−1, 1]. The second family is the
  // Legendre polynomials. There are various conventions for scaling, but the
  // one we will implement is the orthogonal basis for polynomials of degree ≤ n
  // obtained by performing the Gram- Schmidt orthogonalization process on the
  // standard basis 1, x, x2, . . . , xn. Once the roots are found, to find
  // interpolation polynomials of a given function f (x) at the Tchebyshev
  // roots, we construct the Vandermonde matrix for those roots. To find
  // quadrature weights for the Legendre roots, we construct the transpose of
  // the Vandermonde matrix for those roots. Apply your Gaussian elimination
  // with scaled partial pivoting to find the LU factorization of each matrix.
  // Compare the accuracy of interpolation and numerical integration using these
  // points vs. equally spaced points for f (x) = 1/(1 + (5x)2) on the interval
  // [−1, 1].4
  int n = 10;
  int i, j, k, l;
  double sum, tmp;
  double T[n + 1][n + 1], P[n + 1][n + 1], a[n + 1][n + 1], y[n + 1], w[n],
      oneovernormsq[n];

  // Define f as specified
  auto f = [](double x) { return 1.0 / (1.0 + 25.0 * x * x); };

  // Constructs coefficients of ith Chebyshev polynomial in row i, 0<=i<=n
  for (i = 0; i <= n; i++) {
    for (j = 0; j <= n; j++) {
      T[i][j] = 0.0; // initialize to zero
    }
  }
  T[1][1] = T[0][0] = 1.0; // initialize T0 = 1, T1 = x

  for (i = 2; i <= n; i++) {
    for (j = 1; j <= i; j++)
      T[i][j] = 2 * T[i - 1][j - 1]; // Tn=2xTni-1 ...
    for (j = 0; j <= i - 2; j++)
      T[i][j] -= T[i - 2][j]; // ...-Tn-2
  }

  std::vector<double> coefficients(T[n], T[n] + n + 1);
  printVec("Coefficients", coefficients);
  int num_iterations = 0;     // Initialize number of iterations
  double initial_guess = 0.5; // Set an initial guess
  double tolerance = 1e-6;    // Set tolerance
  int max_iterations = 1000;  // Set maximum number of iterations
  auto roots = newton_horners(coefficients, initial_guess, tolerance,
                              max_iterations, num_iterations);
  printVec("Roots", roots);

  // Set up linear system: solution a[j] is coefficients of interpolation
  // polynomial of f(x) at Tchebyshev nodes
  for (i = 0; i <= n; i++)
    for (j = 0; j <= n; j++)
      a[i][j] = pow(roots[i], j); // matrix: ith row evaluates at ith node
  for (i = 0; i <= n; i++)
    y[i] = f(roots[i]); // right side values of f(x) at ith node

  // Run your Gaussian Elimination with Scaled Partial Pivoting here
  std::vector<std::vector<double>> A(n + 1, std::vector<double>(n + 1));
  for (i = 0; i <= n; i++) {
    for (j = 0; j <= n; j++) {
      A[i][j] = a[i][j];
    }
  }
  std::vector<double> y_vec(y, y + n + 1);
  auto result = get_LUP(A);
  auto L = std::get<0>(result);
  auto U = std::get<1>(result);
  auto Permutation = std::get<2>(result);
  std::vector<double> x = solve_LUP(L, U, Permutation, y_vec);
  printVec("Coefficients of interpolation polynomial", x);

  // Constructs coefficients of ith Legendre polynomial in row i, 0<=i<=n by
  // Gram-Schmidt
  for (i = 0; i <= n; i++)
    for (j = 0; j <= n; j++) // initialize coefficients of standard x^i basis
      if (i == j)
        P[i][j] = 1.0;
      else
        P[i][j] = 0.0;

  for (i = 1; i <= n; i++) { // Gram-Schmidt: Orthogonalize x^i w.r.t.
                             // orthogonalized basis for lower degree
    for (j = 0; j < i; j++) {
      // Compute power rule integral inner products of x ^ i with e_j, (v_i,
      // e_j), 0 <= j < i x^k coefficient times x^l coefficient times
      // integral_[-1,1] x^k x^l dx
      sum = 0.0;
      for (k = 0; k <= i; k++)
        for (l = 0; l <= j; l++)
          if ((k + l) % 2 == 0) {

            sum += P[i][k] * P[j][l] * 2.0 /
                   (k + l + 1); // ( integral_[-1,1] x^{k+l} dx )
          }
      sum *= oneovernormsq[j]; // divide by (e_j,e_j) already computed and saved
      for (l = 0; l <= j; l++)
        P[i][l] -=
            sum *
            P[j][l]; // subtract that multiple of the jth orthogonalized e_j
    }                // next j
    // when orthogonal w.r.t. all previous, compute 1/(e_i,e_i) for use in
    // subsequent
    sum = 0.0;
    for (k = 0; k <= i; k++)
      for (l = 0; l <= i; l++)
        if ((k + l) % 2 == 0)
          sum += P[i][k] * P[i][l] * 2.0 / (k + l + 1);
    oneovernormsq[i] = 1.0 / sum;
  } // next i

  // Run your Horner-Newton here
  std::vector<double> coefficients_legendre(P[n], P[n] + n + 1);
  printVec("Coefficients for Legendre", coefficients_legendre);
  auto roots_legendre =
      newton_horners(coefficients_legendre, initial_guess, tolerance,
                     max_iterations, num_iterations);
  printVec("Roots", roots_legendre);

  // Set up linear system whose solution w[j] is quadrature weights at n
  // roots[j], 0<=j<=n-1
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      a[i][j] = pow(roots_legendre[j], i); // matrix for system
  for (i = 0; i < n; i++)
    if (i == 0)
      y[i] = tmp = 2.0; // right side integral_[-1,1] dx
    else if (i % 2 == 0)
      y[i] = tmp = 2.0 / i;
    else
      y[i] = 0.0; // right side integral_[-1,1] x^i dx

  //   // Run your Gaussian Elimination with Scaled Partial Pivoting here

  std::vector<std::vector<double>> A_L(n, std::vector<double>(n));
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      A_L[i][j] = a[i][j];
    }
  }

  std::vector<double> y_vec_legendre(y, y + n);
  result = get_LUP(A_L);
  L = std::get<0>(result);
  U = std::get<1>(result);
  Permutation = std::get<2>(result);
  std::vector<double> x_legendre = solve_LUP(L, U, Permutation, y_vec_legendre);
  printVec("Quadrature Weights", x_legendre);

  // Print sum of quadrature weights, x_legendre
  double sum_weights = 0;
  for (auto weight : x_legendre) {
    sum_weights += weight;
  }
  std::cout << "Sum of Quadrature Weights: " << sum_weights << std::endl;

  for (j = 0; j < n; j++)
    w[j] = x_legendre[j];

  // When the solution w[j] is obtained, perform numerical quadrature of a
  // function f( )
  sum = 0.0;
  for (j = 0; j < n; j++)
    sum += w[j] * f(roots_legendre[j]); // That’s quadrature: Weighted Sum

  std::cout << "Quadrature of f(x) = 1/(1 + 25x^2) on [-1,1] = " << sum
            << std::endl;
}