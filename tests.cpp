#include "lineq.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

TEST(LineqTest, TestGetHilbertian) {
  auto matrix = get_hilbertian(3);
  ASSERT_EQ(matrix.size(), 3);
  ASSERT_EQ(matrix[0].size(), 3);
  // Assert all values are correct
  ASSERT_EQ(matrix[0][0], 1);
  ASSERT_EQ(matrix[0][1], 1.0 / 2);
  ASSERT_EQ(matrix[0][2], 1.0 / 3);
  ASSERT_EQ(matrix[1][0], 1.0 / 2);
  ASSERT_EQ(matrix[1][1], 1.0 / 3);
  ASSERT_EQ(matrix[1][2], 1.0 / 4);
  ASSERT_EQ(matrix[2][0], 1.0 / 3);
  ASSERT_EQ(matrix[2][1], 1.0 / 4);
  ASSERT_EQ(matrix[2][2], 1.0 / 5);
}

TEST(LineqTest, TestGetYSingleColumn) {
  auto column = get_y_single_column(
      3, 1); // Gets the first column of the Hilbertian matrix
  ASSERT_EQ(column.size(), 3);
  ASSERT_EQ(column[0], 1.0 / 1);
  ASSERT_EQ(column[1], 1.0 / 2);
  ASSERT_EQ(column[2], 1.0 / 3);
}

TEST(LineqTest, TestGetYLinearCombination) {
  auto combination = get_y_linear_combination(
      3, 1, 0.5, 2, 0.5); // Gets the linear combination of the first and second
                          // columns of the Hilbertian matrix
  ASSERT_EQ(combination.size(), 3);
  ASSERT_EQ(combination[0], 0.5 * 1.0 / 1 + 0.5 * 1.0 / 2);
  ASSERT_EQ(combination[1], 0.5 * 1.0 / 2 + 0.5 * 1.0 / 3);
  ASSERT_EQ(combination[2], 0.5 * 1.0 / 3 + 0.5 * 1.0 / 4);
}

TEST(LineqTest, TestGetLUDecomposition) {
  std::vector<std::vector<double>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto [L, U] = get_LU_decomposition(A);
  ASSERT_EQ(L.size(), 3);
  ASSERT_EQ(U.size(), 3);
  ASSERT_EQ(L[0].size(), 3);
  ASSERT_EQ(U[0].size(), 3);

  std::vector<std::vector<double>> LU(3, std::vector<double>(3, 0.0));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        LU[i][j] += L[i][k] * U[k][j];
      }
    }
  }

  ASSERT_EQ(LU, A);
}

TEST(LineqTest, TestGetLUP) {
  std::vector<std::vector<double>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto [L, U, P] = get_LUP(A);
  ASSERT_EQ(L.size(), 3);
  ASSERT_EQ(U.size(), 3);
  ASSERT_EQ(P.size(), 3);
  // Check if L is lower triangular
  for (int i = 0; i < L.size(); i++) {
    for (int j = i + 1; j < L[i].size(); j++) {
      ASSERT_NEAR(L[i][j], 0, 1e-9);
    }
  }

  // Check if U is upper triangular
  for (int i = 0; i < U.size(); i++) {
    for (int j = 0; j < i; j++) {
      ASSERT_NEAR(U[i][j], 0, 1e-9);
    }
  }

  // Check if PA = LU
  auto PA = multiply(P, A);
  auto LU = multiply(L, U);
  for (int i = 0; i < PA.size(); i++) {
    for (int j = 0; j < PA[i].size(); j++) {
      ASSERT_NEAR(
          PA[i][j], LU[i][j],
          1e-9); // use ASSERT_NEAR because of potential floating point errors
    }
  }
}

TEST(LineqTest, TestGetLUPLarger) {
  std::vector<std::vector<double>> A = {{1, 0, -1, 0, 0},
                                        {0, 1, 0, 0, 0},
                                        {0, 0, 1, 0, -1e-3},
                                        {-200, 0, 0, 1, 0},
                                        {0, 0, 1e-5, 0, 1}};
  auto [L, U, P] = get_LUP(A);
  ASSERT_EQ(L.size(), 5);
  ASSERT_EQ(U.size(), 5);
  ASSERT_EQ(P.size(), 5);
  // Check if L is lower triangular
  for (int i = 0; i < L.size(); i++) {
    for (int j = i + 1; j < L[i].size(); j++) {
      ASSERT_NEAR(L[i][j], 0, 1e-9);
    }
  }

  // Check if U is upper triangular
  for (int i = 0; i < U.size(); i++) {
    for (int j = 0; j < i; j++) {
      ASSERT_NEAR(U[i][j], 0, 1e-9);
    }
  }

  // Check if PA = LU
  auto PA = multiply(P, A);
  auto LU = multiply(L, U);
  for (int i = 0; i < PA.size(); i++) {
    for (int j = 0; j < PA[i].size(); j++) {
      ASSERT_NEAR(
          PA[i][j], LU[i][j],
          1e-9); // use ASSERT_NEAR because of potential floating point errors
    }
  }
}

TEST(LineqTest, TestSolveLU) {
  std::vector<std::vector<double>> L = {
      {1, 0, 0}, {0.5, 1, 0}, {1.0 / 3, 1, 1}};
  std::vector<std::vector<double>> U = {
      {1, 0.5, 1.0 / 3}, {0, 1.0 / 12, 1.0 / 12}, {0, 0, 1.0 / 180}};
  std::vector<double> y = {1, 1.0 / 2, 1.0 / 3};
  auto x = solve_LU(L, U, y);
  ASSERT_EQ(x.size(), 3);
  ASSERT_NEAR(x[0], 1, 1e-9);
  ASSERT_NEAR(x[1], 0, 1e-9);
  ASSERT_NEAR(x[2], 0, 1e-9);
}

TEST(LineqTest, TestSolveLUP) {
  std::vector<std::vector<double>> L = {
      {1, 0, 0}, {0.5, 1, 0}, {1.0 / 3, 1, 1}};
  std::vector<std::vector<double>> U = {
      {1, 0.5, 1.0 / 3}, {0, 1.0 / 12, 1.0 / 12}, {0, 0, 1.0 / 180}};
  std::vector<std::vector<double>> P = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  std::vector<double> y = {1, 1.0 / 2, 1.0 / 3};
  auto x = solve_LUP(L, U, P, y);
  ASSERT_EQ(x.size(), 3);
  ASSERT_NEAR(x[0], 1, 1e-9);
  ASSERT_NEAR(x[1], 0, 1e-9);
  ASSERT_NEAR(x[2], 0, 1e-9);
}

TEST(TransposeTest, TransposeSquareMatrix) {
  std::vector<std::vector<double>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  std::vector<std::vector<double>> expected = {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
  std::vector<std::vector<double>> result = transpose(A);
  EXPECT_EQ(expected, result);
}

TEST(TransposeTest, TransposeRectangularMatrix) {
  std::vector<std::vector<double>> A = {{1, 2, 3}, {4, 5, 6}};
  std::vector<std::vector<double>> expected = {{1, 4}, {2, 5}, {3, 6}};
  std::vector<std::vector<double>> result = transpose(A);
  EXPECT_EQ(expected, result);
}

TEST(TransposeTest, TransposeEmptyMatrix) {
  std::vector<std::vector<double>> A;
  std::vector<std::vector<double>> expected;
  std::vector<std::vector<double>> result = transpose(A);
  EXPECT_EQ(expected, result);
}

TEST(TransposeTest, TransposeSingleRowMatrix) {
  std::vector<std::vector<double>> A = {{1, 2, 3}};
  std::vector<std::vector<double>> expected = {{1}, {2}, {3}};
  std::vector<std::vector<double>> result = transpose(A);
  EXPECT_EQ(expected, result);
}

TEST(TransposeTest, TransposeSingleColumnMatrix) {
  std::vector<std::vector<double>> A = {{1}, {2}, {3}};
  std::vector<std::vector<double>> expected = {{1, 2, 3}};
  std::vector<std::vector<double>> result = transpose(A);
  EXPECT_EQ(expected, result);
}

TEST(CholeskyDecompositionTest, SquareMatrix) {
  std::vector<std::vector<double>> A = {
      {4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
  std::vector<std::vector<double>> expected_L = {
      {2, 0, 0}, {6, 1, 0}, {-8, 5, 3}};
  std::vector<std::vector<double>> L = choleskyDecomposition(A);
  ASSERT_EQ(L, expected_L);
}

TEST(CholeskyTest, SolveCholesky) {
  // Test case 1
  std::vector<std::vector<double>> L1 = {{2, 0}, {1, 1}};
  std::vector<double> y1 = {4.0, 2.0};
  std::vector<double> expected1 = {1.0, 0.0};
  std::vector<double> result1 = solve_cholesky(L1, y1);
  ASSERT_EQ(result1.size(), expected1.size());
  for (size_t i = 0; i < result1.size(); ++i) {
    EXPECT_DOUBLE_EQ(result1[i], expected1[i]);
  }
}

TEST(CholeskyTest, SolveCholesky2) {
  // Test case 2
  std::vector<std::vector<double>> L2 = {{2, 0}, {1, 1}};
  std::vector<double> y2 = {2.0, 2.0};
  std::vector<double> expected2 = {0.0, 1.0};
  std::vector<double> result2 = solve_cholesky(L2, y2);
  ASSERT_EQ(result2.size(), expected2.size());
  for (size_t i = 0; i < result2.size(); ++i) {
    EXPECT_DOUBLE_EQ(result2[i], expected2[i]);
  }
}

// Test case for the multiply function
TEST(MultiplyTest, Multiplication) {
  // Test input matrices
  std::vector<std::vector<double>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  std::vector<double> x = {1, 2, 3};

  // Expected output vector
  std::vector<double> expected_result = {14, 32, 50};

  // Call the multiply function
  std::vector<double> result = multiply(A, x);

  // Check if the result matches the expected output
  ASSERT_EQ(result, expected_result);
}

TEST(DRSeperationTest, Handles3x3Matrix) {
  std::vector<std::vector<double>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  auto [D, R] = DR_seperation(A);

  std::vector<std::vector<double>> expectedD = {
      {1, 0, 0}, {0, 5, 0}, {0, 0, 9}};

  std::vector<std::vector<double>> expectedR = {
      {0, 2, 3}, {4, 0, 6}, {7, 8, 0}};

  EXPECT_EQ(D, expectedD);
  EXPECT_EQ(R, expectedR);
}

TEST(InvertDiagonalMatrixTest, Handles3x3Matrix) {
  std::vector<std::vector<double>> D = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};

  auto inv_D = invert_diagonal_matrix(D);

  std::vector<std::vector<double>> expected_inv_D = {
      {1, 0, 0}, {0, 0.5, 0}, {0, 0, 1.0 / 3}};

  EXPECT_EQ(inv_D, expected_inv_D);
}

TEST(SingleJacobiIterationTest, Handles3x3Matrix) {
  std::vector<std::vector<double>> inv_D = {
      {1, 0, 0}, {0, 0.5, 0}, {0, 0, 1.0 / 3}};

  std::vector<std::vector<double>> R = {{0, 2, 3}, {4, 0, 6}, {7, 8, 0}};

  std::vector<double> y = {1, 2, 3};
  std::vector<double> x_k = {0, 0, 0};

  auto result = single_jacobi_itteration(inv_D, R, y, x_k);

  std::vector<double> expected_result = {1, 1, 1};

  for (int i = 0; i < expected_result.size(); i++) {
    EXPECT_NEAR(result[i], expected_result[i], 1e-9);
  }
}

TEST(GetInteriorPointsTest, TestSize) {
  int n = 5;
  auto result = get_interior_points(n);
  EXPECT_EQ(result.size(), n - 1);
  for (const auto &row : result) {
    EXPECT_EQ(row.size(), n - 1);
  }
}

// TEST(GetInteriorPointsTest, TestValues) {
//   int n = 5;
//   double h = M_PI / n;
//   auto result = get_interior_points(n);
//   for (int i = 0; i < n - 1; i++) {
//     for (int j = 0; j < n - 1; j++) {
//       EXPECT_DOUBLE_EQ(result[i][j], (i + 1) * h);
//     }
//   }
// }

// TEST(GetInteriorPointsTest, TestBoundary) {
//   int n = 5;
//   auto result = get_interior_points(n);
//   EXPECT_DOUBLE_EQ(result[0][0], M_PI / n);
//   EXPECT_DOUBLE_EQ(result[n - 2][n - 2], (n - 1) * M_PI / n);
// }

TEST(LaplacianMatrixTest, Size2) {
  auto result = get_discrete_laplacian_matrix(2);
  auto laplacian = result.first;
  auto laplacian_indices = result.second;
  std::vector<std::vector<double>> expected_laplacian = {
      {-4, 1, 1, 0}, {1, -4, 0, 1}, {1, 0, -4, 1}, {0, 1, 1, -4}};
  for (int i = 0; i < laplacian.size(); i++) {
    for (int j = 0; j < laplacian[0].size(); j++) {
      EXPECT_DOUBLE_EQ(laplacian[i][j],
                       expected_laplacian[i][laplacian_indices[i][j]]);
    }
  }
}

TEST(compressed_DR_seperationTest, Case1) {
  auto result = compressed_DR_seperation({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                                         {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}});
  auto D = std::get<0>(result);

  std::vector<double> expectedD = {1, 5, 9};

  auto R = std::get<1>(result);
  auto R_indices = std::get<2>(result);

  std::vector<std::vector<double>> expectedR = {{2, 3}, {4, 6}, {7, 8}};
  std::vector<std::vector<int>> expectedR_indices = {{1, 2}, {0, 2}, {0, 1}};

  EXPECT_EQ(D, expectedD);
  EXPECT_EQ(R, expectedR);
  EXPECT_EQ(R_indices, expectedR_indices);
}

TEST(MultiplySparseTest, Test1) {
  std::vector<std::vector<double>> A = {{1, 2}, {3, 4}};
  std::vector<std::vector<int>> A_idxs = {{0, 1}, {0, 1}};
  std::vector<double> x = {1, 2};
  std::vector<double> expected = {5, 11};
  ASSERT_EQ(multiply_sparse(A, A_idxs, x), expected);
}

TEST(MultiplySparseTest, Test2) {
  std::vector<std::vector<double>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  std::vector<std::vector<int>> A_idxs = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
  std::vector<double> x = {1, 2, 3};
  std::vector<double> expected = {14, 32, 50};
  ASSERT_EQ(multiply_sparse(A, A_idxs, x), expected);
}

TEST(MultiplySparseTest, Test3) {
  std::vector<std::vector<double>> A = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  std::vector<std::vector<int>> A_idxs = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
  std::vector<double> x = {1, 2, 3};
  std::vector<double> expected = {1, 2, 3};
  ASSERT_EQ(multiply_sparse(A, A_idxs, x), expected);
}