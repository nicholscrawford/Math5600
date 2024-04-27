#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lineq.h"
#include "rootfinding.h"
#include "utils.h"

std::vector<double> get_tchebyshev_coefficients(int n) {
  int i, j;
  double T[n + 1][n + 1];
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
  return coefficients;
}

std::vector<double> get_legendre_coefficients(int n) {
  int i, j, k, l;
  double P[n + 1][n + 1], sum, oneovernormsq[n + 1];
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

  std::vector<double> coefficients_legendre(P[n], P[n] + n + 1);
  return coefficients_legendre;
}

std::vector<double>
get_interpolation_coefficients(std::vector<double> &roots,
                               std::function<double(double)> f) {

  int n = roots.size() - 1;
  int i, j;
  double a[n + 1][n + 1], y[n + 1];
  // Set up linear system: solution a[j] is coefficients of interpolation
  // polynomial of f(x) at Tchebyshev nodes
  for (i = 0; i <= n; i++)
    for (j = 0; j <= n; j++)
      a[i][j] = pow(roots[i], j); // matrix: ith row evaluates at ith node
  for (i = 0; i <= n; i++)
    y[i] = f(roots[i]); // right side values of f(x) at ith node

  // Run Gaussian Elimination with Scaled Partial Pivoting to interpolate
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
  return x;
}

std::vector<double> get_quadrature_weights(std::vector<double> &roots,
                                           std::function<double(double)> f) {
  int n = roots.size() - 1;
  int i, j;
  double a[n + 1][n + 1], y[n + 1];
  // Set up linear system whose solution w[j] is quadrature weights at n
  // roots[j], 0<=j<=n-1
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      a[i][j] = pow(roots[j], i); // matrix for system
  for (i = 0; i < n; i++)
    if (i % 2 == 0)
      y[i] = 2.0 / (i + 1);
    else
      y[i] = 0.0; // right side integral_[-1,1] x^i dx

  // Run Gaussian Elimination with Scaled Partial Pivoting to get quadrature
  // weights
  std::vector<std::vector<double>> A_L(n, std::vector<double>(n));
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      A_L[i][j] = a[i][j];
    }
  }

  std::vector<double> y_vec_legendre(y, y + n);
  auto result = get_LUP(A_L);
  auto L = std::get<0>(result);
  auto U = std::get<1>(result);
  auto Permutation = std::get<2>(result);
  std::vector<double> x_legendre = solve_LUP(L, U, Permutation, y_vec_legendre);
  return x_legendre;
}

double quadrature(std::vector<double> &roots, std::vector<double> &weights,
                  std::function<double(double)> f) {
  double sum = 0;
  for (int i = 0; i < weights.size(); i++) {
    sum += weights[i] * f(roots[i]);
  }
}
