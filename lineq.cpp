
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
  std::vector<double> max_elements(n, 0.0);
  for (int i = 0; i < n; ++i) {
    P[i][i] = 1.0;
    max_elements[i] = *std::max_element(U[i].begin(), U[i].end());
  }

  // Perform LU decomposition with scaled partial pivoting
  for (int i = 0; i < n; ++i) { // For each element in the diagonal
    // Find the pivot row and swap rows if necessary
    int pivot_row = i;
    double max_ratio = 0.0;
    for (int j = i + 1; j < n; ++j) {
      double ratio = std::abs(U[j][i]) / max_elements[j];
      if (ratio > max_ratio) {
        max_ratio = ratio;
        pivot_row = j;
      }
    }
    std::swap(U[i], U[pivot_row]);
    std::swap(P[i], P[pivot_row]);
    std::swap(L[i], L[pivot_row]);
    std::swap(max_elements[i], max_elements[pivot_row]);

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
  std::vector<double> b(n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      b[i] += P[i][j] * y[j];
    }
  }

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
 * Multiplies a matrix represented as vectors of vectors of doubles, and a
 * vector of doubles. i.e. A * x = y
 *
 * @param A The matrix.
 * @param B The vector.
 * @return The resulting vector after multiplication.
 */
std::vector<double> multiply(const std::vector<std::vector<double>> &A,
                             const std::vector<double> &x) {
  int n1 = A.size();
  int n2 = x.size();

  std::vector<double> result(n1, 0);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      result[i] += A[i][j] * x[j];
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

/**
 * @brief Separates the given matrix A into two matrices D and R.
 *
 * The matrix A is separated into two matrices D and R, where D is a diagonal
 * matrix containing the diagonal elements of A, and R is a matrix containing
 * the remaining elements of A.
 *
 * @param A The input matrix.
 * @return A std::pair containing the matrices D and R.
 */
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
DR_seperation(const std::vector<std::vector<double>> &A) {
  int n = A.size();
  std::vector<std::vector<double>> D(n, std::vector<double>(n, 0));
  std::vector<std::vector<double>> R(n, std::vector<double>(n, 0));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        D[i][j] = A[i][j];
      } else {
        R[i][j] = A[i][j];
      }
    }
  }
  return std::make_pair(D, R);
}

/**
 * @brief Inverts a diagonal matrix.
 *
 * This function takes a diagonal matrix represented as a 2D vector and returns
 * its inverse. The input matrix must be square and have non-zero diagonal
 * elements.
 *
 * @param D The diagonal matrix to be inverted.
 * @return The inverse of the input diagonal matrix.
 */
std::vector<std::vector<double>>
invert_diagonal_matrix(const std::vector<std::vector<double>> &D) {
  int n = D.size();
  std::vector<std::vector<double>> inv_D(n, std::vector<double>(n, 0));
  for (int i = 0; i < n; i++) {
    inv_D[i][i] = 1.0 / D[i][i];
  }
  return inv_D;
}

/**
 * @brief Inverts a diagonal matrix.
 *
 * This function takes a diagonal matrix represented as a 2D vector and returns
 * its inverse. The input matrix must be square and have non-zero diagonal
 * elements.
 *
 * @param D The diagonal matrix to be inverted.
 * @return The inverse of the input diagonal matrix.
 */
std::vector<double> invert_diagonal_matrix(const std::vector<double> &D) {
  int n = D.size();
  std::vector<double> inv_D(n, 0);
  for (int i = 0; i < n; i++) {
    inv_D[i] = 1.0 / D[i];
  }
  return inv_D;
}

/**
 * @brief Perform DR Seperation on a compressed matrix.
 *
 * @param A compressed values of a, where each row has it's zeros removed, and
 * the indicies are kept in
 * @param A_indices here.
 * @return std::tuple<std::vector<double>, std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
 * returns diagonal as a vector, and the remainder as two vector of vectors
 representing a compressed matrix.
 */
std::tuple<std::vector<double>, std::vector<std::vector<double>>,
           std::vector<std::vector<int>>>
compressed_DR_seperation(const std::vector<std::vector<double>> &A,
                         const std::vector<std::vector<int>> &A_indices) {
  int n = A.size();
  std::vector<double> D(n, 0);
  std::vector<std::vector<double>> R(n, std::vector<double>());
  std::vector<std::vector<int>> R_indices(n, std::vector<int>());

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < A[i].size(); j++) {
      if (i == A_indices[i][j]) {
        D[i] = A[i][j];
      } else {
        R[i].push_back(A[i][j]);
        R_indices[i].push_back(A_indices[i][j]);
      }
    }
  }
  return std::make_tuple(D, R, R_indices);
}

/**
 * Performs a single Jacobi iteration to solve a linear system of equations.
 *
 * @param inv_D The inverse of the diagonal matrix D.
 * @param R The matrix R.
 * @param y The vector y.
 * @param x_k The vector x_k.
 * @return The updated vector x^(k+1) after the Jacobi iteration.
 */
std::vector<double>
single_jacobi_itteration(const std::vector<std::vector<double>> &inv_D,
                         const std::vector<std::vector<double>> &R,
                         const std::vector<double> &y,
                         const std::vector<double> &x_k) {
  // x^(k+1) = D-1(y – Rx^(k))
  std::vector<double> Rx_k = multiply(R, x_k);
  std::vector<double> y_minus_Rx_k = subtract(y, Rx_k);
  std::vector<double> D_inv_y_minus_Rx_k = multiply(inv_D, y_minus_Rx_k);
  return D_inv_y_minus_Rx_k;
}

/**
 * Multiplies a sparse matrix represented by a vector of vectors with a vector.
 *
 * @param A The sparse matrix represented by a vector of vectors.
 * @param A_idxs The indices of non-zero elements in the sparse matrix.
 * @param x The vector to be multiplied with the sparse matrix.
 * @return The result of multiplying the sparse matrix with the vector.
 */
std::vector<double> multiply_sparse(const std::vector<std::vector<double>> &A,
                                    const std::vector<std::vector<int>> &A_idxs,
                                    const std::vector<double> &x) {
  int n = A.size();
  std::vector<double> result(n, 0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < A[i].size(); j++) {
      result[i] += A[i][j] * x[A_idxs[i][j]];
    }
  }
  return result;
}

/**
 * Performs element-wise multiplication of two vectors.
 *
 * @param A The first vector.
 * @param B The second vector.
 * @return The resulting vector after element-wise multiplication.
 */
std::vector<double> elementwise_multiply(const std::vector<double> &A,
                                         const std::vector<double> &B) {
  int n = A.size();
  std::vector<double> result(n, 0);
  for (int i = 0; i < n; i++) {
    result[i] = A[i] * B[i];
  }
  return result;
}

/**
 * Performs a sparse single Jacobi iteration.
 *
 * This function calculates the next iteration of the Jacobi method for solving
 * a linear system of equations. It takes the inverted diagonal matrix, the
 * remainder matrix, the compression information of the remainder matrix, the
 * vector y, and the current approximation x_k as input. It returns the updated
 * approximation x^(k+1) according to the formula: x^(k+1) = D^(-1)(y - Rx^(k)),
 * where D^(-1) is the inverted diagonal matrix, R is the remainder matrix, and
 * x^(k) is the current approximation.
 *
 * @param inv_D The inverted diagonal matrix as a vector. It can be obtained by
 *             element-wise inversion of the diagonal matrix.
 * @param R The remainder matrix in dense form. It is represented as a 2D vector
 *          with each row containing the non-zero elements of a row in the
 *          remainder matrix.
 * @param R_indices The compression information of the remainder matrix. It is
 *                  a 2D vector with each row containing the indices of the
 *                  non-zero elements in the corresponding row of the remainder
 *                  matrix.
 * @param y The vector y in the equation.
 * @param x_k The current approximation x^(k).
 * @return The updated approximation x^(k+1).
 */
std::vector<double> sparse_single_jacobi_itteration(
    const std::vector<double> &inv_D, // inverted diagonal matrix as a vector,
                                      // can just elementewise multiply.
    const std::vector<std::vector<double>>
        &R, // Remainder matrix in dense form -- i.e [[0, 0, 3], [0, 3, 0]]
            // represented as [[3],[3]]
    const std::vector<std::vector<int>>
        &R_indices, // Remainder matrix compression information, the indexes
                    // inorder of the R mat. i.e. [[2], [3]] for the above
                    // example
    const std::vector<double> &y, const std::vector<double> &x_k) {
  // x^(k+1) = D-1(y – Rx^(k))
  std::vector<double> Rx_k = multiply_sparse(R, R_indices, x_k);
  std::vector<double> y_minus_Rx_k = subtract(y, Rx_k);
  std::vector<double> D_inv_y_minus_Rx_k =
      elementwise_multiply(inv_D, y_minus_Rx_k);
  return D_inv_y_minus_Rx_k;
}

std::vector<double> sparse_single_gauss_seidel_itteration(
    const std::vector<std::vector<double>> &A,
    const std::vector<std::vector<int>> &A_indices,
    const std::vector<double> &x_k, const std::vector<double> &y) {
  int n = A.size();
  std::vector<double> x_k_plus_1(n, 0);
  for (int i = 0; i < n; i++) {
    double sum = 0;
    double a_ii = 0; // Note this will cause problems if there is no diagonal
                     // element, but I'm not checking for that absence.
    for (int _ = 0; _ < A[i].size(); _++) {
      int j = A_indices[i][_];
      if (j > i) {
        sum += A[i][_] * x_k[j];
      }
      if (j < i) {
        sum += A[i][_] * x_k_plus_1[j];
      }
      if (j == i) {
        a_ii = A[i][_];
      }
    }
    x_k_plus_1[i] = (y[i] - sum) / a_ii;
  }
  return x_k_plus_1;
}

/**
 * Multiplies each element of a matrix by a scalar value.
 *
 * @param k The scalar value to multiply the matrix by.
 * @param A The matrix to be multiplied.
 * @return The resulting matrix after multiplying each element by the scalar
 * value.
 */
const std::vector<std::vector<double>>
multiply(double k, const std::vector<std::vector<double>> &A) {
  int n = A.size();
  int m = A[0].size();
  std::vector<std::vector<double>> result(n, std::vector<double>(m, 0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      result[i][j] = k * A[i][j];
    }
  }
  return result;
}

/**
 * Multiplies each element of a vector by a scalar value.
 *
 * @param k The scalar value to multiply the vector by.
 * @param A The vector to be multiplied.
 * @return The resulting vector after multiplying each element by the scalar
 * value.
 */
const std::vector<double> multiply(double k, const std::vector<double> &A) {
  int n = A.size();
  std::vector<double> result(n, 0);
  for (int i = 0; i < n; i++) {
    result[i] = k * A[i];
  }
  return result;
}

/**
 * @brief Returns a vector of vectors representing the interior points of a
 * square.
 *
 * The square [0, π]×[0, π] with zero boundary condition is divided into a grid
 * of n−1×n−1 equally spaced interior points.
 *
 * @param n The number (+1) of points in each dimension of the grid.
 * @return std::vector<std::vector<std::pair<double, double>>>

 A vector of vectors of pairs, representing the
 * interior points' x and y.
 */
std::vector<std::vector<std::pair<double, double>>> get_interior_points(int n) {
  //   the square [0, π]×[0, π] with zero boundary
  // condition. Use a grid of n−1×n−1 equally spaced interior points. (xj , yi)
  // = (jh, ih) where h = π n . both j, i = 1, . . . , n − 1.
  std::vector<std::vector<std::pair<double, double>>> interior_points(
      n - 1, std::vector<std::pair<double, double>>(n - 1));

  double h = M_PI / n;
  for (int i = 1; i < n; i++) {
    for (int j = 1; j < n; j++) {
      interior_points[i - 1][j - 1].first = j * h;
      interior_points[i - 1][j - 1].second = i * h;
    }
  }
  return interior_points;
}

/**
 * Calculates the Laplacian matrix for the 2D square finite differencing
 * discrete Laplacian approximation, ignoring spacing term.
 *
 * @param A The input matrix.
 * @return The Laplacian approximation matrix, and it's compression information.
 */
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>>
get_discrete_laplacian_matrix(int size) {
  // Create the laplacian matrix (ignoring spacing term)
  int points_size = size;
  int matrix_size = size * size;

  std::vector<std::vector<double>> laplacian(matrix_size,
                                             std::vector<double>(0));
  std::vector<std::vector<int>> laplacian_indices(matrix_size,
                                                  std::vector<int>(0));

  // Ajacency matrix basically, with the diagonals as -4, and their neighbors
  // as 1. The ordering determines which are neighbors. Following the convention
  // in class, 0  is bottom left, 1 is bottom right, 2 is top left, 3 is top
  // right. etc. So, for any diagonal, the neigbors are 1, and +- the
  // points_size are 1.

  /**
   * * 1  *
   * 1 -4 1
   * * 1  *
   *
   * [
   * [0 1 0 1 -4 1 0 1 0],
   * [...],
   * ...
   * ]
   *
   */

  //
  for (int i = 0; i < matrix_size; i++) {
    // Below
    if (i - points_size >= 0) {
      laplacian[i].push_back(1);
      laplacian_indices[i].push_back(i - points_size);
    }
    // Above
    if (i + points_size < matrix_size) {
      laplacian[i].push_back(1);
      laplacian_indices[i].push_back(i + points_size);
    }
    // Right
    if (i % points_size + 1 < points_size) {
      laplacian[i].push_back(1);
      laplacian_indices[i].push_back(i + 1);
    }
    // Left
    if (i % points_size - 1 >= 0) {
      laplacian[i].push_back(1);
      laplacian_indices[i].push_back(i - 1);
    }
    laplacian[i].push_back(-4);
    laplacian_indices[i].push_back(i);
  }

  return std::make_pair(laplacian, laplacian_indices);
}

/**
 * Calculates the dot product of two vectors.
 *
 * @param A The first vector.
 * @param B The second vector.
 * @return The dot product of the two vectors.
 */
double dot_product(const std::vector<double> &A, const std::vector<double> &B) {
  double result = 0;
  for (int i = 0; i < A.size(); i++) {
    result += A[i] * B[i];
  }
  return result;
}

/**
 * Calculates the magnitude of a vector.
 *
 * @param vec The vector for which to calculate the magnitude.
 * @return The magnitude of the vector.
 */
double magnitude(const std::vector<double> &vec) {
  double sum = 0.0;
  for (double val : vec) {
    sum += val * val;
  }
  return sqrt(sum);
}