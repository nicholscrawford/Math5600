#include "lineq.h"
#include <gtest/gtest.h>

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