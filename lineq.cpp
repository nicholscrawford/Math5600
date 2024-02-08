
#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include "lineq.h"

/**
 * @brief Creates a Hilbertian matrix of size n.
 *
 * This function generates a Hilbertian matrix of size n, where each element
 * is the reciprocal of the sum of its row and column indices minus one.
 *
 * @param n The size of the matrix.
 * @return The generated Hilbertian matrix.
 */
std::vector<std::vector<double>> get_hilbertian(int n) {

  std::vector<std::vector<double>> hilbertMatrix(n,
                                                 std::vector<double>(n, 0.0));

  for (int i = 1; i <= n; ++i) {
    for (int j = 1; j <= n; ++j) {
      hilbertMatrix[i - 1][j - 1] = 1.0 / static_cast<double>(i + j - 1);
    }
  }

  return hilbertMatrix;
}

std::vector<double> get_y_single_column(int n, int column_j) {
  std::vector<double> hilbertMatrix_column = std::vector<double>(n, 0.0);

  for (int i = 1; i <= n; ++i) {
    hilbertMatrix_column[i - 1] = 1.0 / static_cast<double>(i + column_j - 1);
  }
  return hilbertMatrix_column;
}

std::vector<double> get_y_linear_combination(int n, int first_column_j,
                                             double first_column_coef,
                                             int second_column_j,
                                             double second_column_coef) {
  std::vector<double> hilbertMatrix_linear_combination =
      std::vector<double>(n, 0.0);

  for (int i = 1; i <= n; ++i) {
    hilbertMatrix_linear_combination[i - 1] =
        first_column_coef * 1.0 / static_cast<double>(i + first_column_j - 1) +
        second_column_coef * 1.0 / static_cast<double>(i + second_column_j - 1);
  }
  return hilbertMatrix_linear_combination;
}

/**
 * @brief Performs LU decomposition on a given matrix A.
 *
 * @param A The square input matrix to be decomposed.
 * @return A std::pair of matrices (L, U) representing the LU decomposition of
 * A.
 */
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_LU_decomposition(std::vector<std::vector<double>> &A) {
  int n = A.size();

  // Initialize matrices L with zeros
  std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> U = A;

  // Create L and U through Gaussian elimination
  for (int i = 0; i < n; i++) { // For each element along the diagonal
    double div_factor = 1.0 / U[i][i];
    L[i][i] = 1.0; // The diagonal of L is 1.0
    for (int j = i + 1; j < n;
         j++) { // For each row below the current row, subtract the current row
                // multiplied by the factor, to make all elents below the
                // diagonal zero.
      double factor = U[j][i] * div_factor;
      L[j][i] = factor;
      for (int k = i; k < n; k++) {
        U[j][k] -= factor * U[i][k];
      }
    }
  }
  return std::make_pair(L, U);
}

/**
 * @brief Performs LU decomposition with scaled partial pivoting on a given
 * matrix A.
 *
 * This function performs LU decomposition with scaled partial pivoting on a
 * given square matrix A. It returns the matrices L, U, and P, where L is the
 * lower triangular matrix, U is the upper triangular matrix, and P is the
 * permutation matrix. LU = PA
 *
 * @param A The square input matrix to be decomposed.
 * @return A std::tuple of matrices (L, U, P) representing the LU decomposition
 * of A.
 */
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
get_LUP(std::vector<std::vector<double>> &A) {
  int n = A.size();

  // Initialize matrices L, U, and P
  std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> U = A;
  std::vector<std::vector<double>> P(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i) {
    P[i][i] = 1.0;
  }

  // Perform LU decomposition with scaled partial pivoting
  for (int i = 0; i < n; ++i) { // For each element in the diagonal
    // Find the pivot element and swap rows if necessary
    int pivot_row = i;
    double max_ratio = 0.0;
    for (int j = i; j < n; ++j) {
      auto max_it = std::max_element(U[j].begin(), U[j].end());
      double ratio = std::abs(U[j][i]) / *max_it;
      if (ratio > max_ratio) {
        max_ratio = ratio;
        pivot_row = j;
      }
    }
    std::swap(U[i], U[pivot_row]);
    std::swap(P[i], P[pivot_row]);
    std::swap(L[i], L[pivot_row]);

    double div_factor = 1.0 / U[i][i];
    L[i][i] = 1.0;
    for (int j = i + 1; j < n; ++j) {
      double factor = U[j][i] * div_factor;
      L[j][i] = factor;
      for (int k = i; k < n; ++k) {
        U[j][k] -= factor * U[i][k];
      }
    }
  }

  return std::make_tuple(L, U, P);
}

/**
 * Performs Cholesky decomposition on a given matrix A.
 *
 * @param A The matrix to be decomposed.
 * @return The lower triangular matrix L resulting from the Cholesky
 * decomposition.
 */
std::vector<std::vector<double>>
choleskyDecomposition(std::vector<std::vector<double>> A) {
  int n = A.size();
  std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0;
      if (j == i) { // Diagonal elements
        for (int k = 0; k < j; k++)
          sum += pow(L[j][k], 2);
        L[j][j] = sqrt(A[j][j] - sum);
      } else { // Off-diagonal elements
        for (int k = 0; k < j; k++)
          sum += (L[i][k] * L[j][k]);
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }

  return L;
}

/**
 * Solves a system of linear equations using LUP decomposition.
 *
 * @param L The lower triangular matrix of the LUP decomposition.
 * @param U The upper triangular matrix of the LUP decomposition.
 * @param P The permutation matrix of the LUP decomposition.
 * @param y The right-hand side vector.
 * @return The solution vector x.
 */
std::vector<double> solve_LUP(std::vector<std::vector<double>> &L,
                              std::vector<std::vector<double>> &U,
                              std::vector<std::vector<double>> &P,
                              std::vector<double> &y) {
  int n = L.size();

  // Forward substitution to solve Ly = Pb
  std::vector<double> b = y;
  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      b[i] -= L[i][j] * b[j];
    }
  }

  // Backward substitution to solve Ux = y
  std::vector<double> x(n, 0.0);
  for (int i = n - 1; i >= 0; --i) {
    x[i] = b[i];
    for (int j = i + 1; j < n; ++j) {
      x[i] -= U[i][j] * x[j];
    }
    x[i] /= U[i][i];
  }

  return x;
}

/**
 * Solve a linear system using an LU decomposition.
 *
 * @param L The lower triangular matrix of the LU decomposition.
 * @param U The upper triangular matrix of the LU decomposition.
 * @param y The right-hand side vector.
 * @return The solution vector.
 */
std::vector<double> solve_LU(std::vector<std::vector<double>> &L,
                             std::vector<std::vector<double>> &U,
                             std::vector<double> &y) {
  int n = L.size();
  // We want to solve LUx = y
  // Let Ux = b, and Lb=x i.e. L(Ux) = L(b) = y
  // So we solve the outer section first, then use the result to solve Ux=b.
  // Forward substitution to solve Lb = y
  std::vector<double> b = std::vector<double>(n, 0.0);
  for (int i = 0; i < n; ++i) {
    b[i] = y[i];
    for (int j = 0; j < i; ++j) {
      b[i] -= L[i][j] * b[j];
    }
    b[i] /= L[i][i];
  }

  // Backward substitution to solve Ux = b
  std::vector<double> x(n, 0.0);
  for (int i = n - 1; i >= 0; --i) {
    x[i] = b[i];
    for (int j = i + 1; j < n; ++j) {
      x[i] -= U[i][j] * x[j];
    }
    x[i] /= U[i][i];
  }

  return x;
}

/**
 * Solves a linear system using the Cholesky decomposition.
 *
 * @param L The lower triangular matrix of the Cholesky decomposition.
 * @param y The right-hand side vector.
 * @return The solution vector.
 */
std::vector<double> solve_cholesky(std::vector<std::vector<double>> &L,
                                   std::vector<double> &y) {
  auto LT = transpose(L);
  return solve_LU(L, LT, y);
}

/**
 * Multiplies two matrices represented as vectors of vectors of doubles.
 *
 * @param A The first matrix.
 * @param B The second matrix.
 * @return The resulting matrix after multiplication.
 */
std::vector<std::vector<double>>
multiply(const std::vector<std::vector<double>> &A,
         const std::vector<std::vector<double>> &B) {
  int n1 = A.size();
  int n2 = B[0].size();
  int n3 = B.size();

  std::vector<std::vector<double>> result(n1, std::vector<double>(n2, 0));

  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      for (int k = 0; k < n3; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result;
}

/**
 * @brief Subtracts two vectors element-wise.
 *
 * This function subtracts each element of vector A from the corresponding
 * element of vector B. The resulting vector has the same size as the input
 * vectors.
 *
 * @param A The first vector.
 * @param B The second vector.
 * @return The resulting vector after element-wise subtraction.
 */
std::vector<double> subtract(const std::vector<double> &A,
                             const std::vector<double> &B) {
  int n = A.size();
  std::vector<double> result(n, 0);
  for (int i = 0; i < n; i++) {
    result[i] = A[i] - B[i];
  }
  return result;
}

/**
 * Transposes a matrix represented by a vector of vectors.
 *
 * @param A The input matrix.
 * @return The transposed matrix.
 */
std::vector<std::vector<double>>
transpose(const std::vector<std::vector<double>> &A) {
  int n = A.size();
  if (n == 0) {
    return std::vector<std::vector<double>>(0, std::vector<double>(0, 0));
  }
  int m = A[0].size();
  if (m == 0) {
    return std::vector<std::vector<double>>(0, std::vector<double>(n, 0));
  }

  std::vector<std::vector<double>> result(m, std::vector<double>(n, 0));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      result[j][i] = A[i][j];
    }
  }
  return result;
}