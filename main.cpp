#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "lineq.h"
#include "utils.h"

int main(int, char **) {

  int n = 3;
  int y_single_col_idx = 2;   // 1 indexed
  int y_linear_comb_col1 = 1; // 1 indexed
  int y_linear_comb_col2 = 3; // 1 indexed
  int y_linear_comb_col1_coef = 0.08;
  int y_linear_comb_col2_coef = 0.2;

  std::vector<std::vector<double>> hilbertMatrix = get_hilbertian(n);
  printVec("Hilbertian Matrix: ", hilbertMatrix);

  // Get the LU decomposition of the Hilbert Matrix
  std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
      LU_Pair = get_LU_decomposition(hilbertMatrix);
  std::vector<std::vector<double>> L = LU_Pair.first;
  std::vector<std::vector<double>> U = LU_Pair.second;
  printVec("LU Decomp, L: ", L);
  printVec("LU Decomp, U: ", U);
  //   printVec("L * U: ", multiply(L, U));

  // Decompose Hilbert to LUP
  std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>,
             std::vector<std::vector<double>>>
      LUP_Tuple = get_LUP(hilbertMatrix);
  std::vector<std::vector<double>> L_LUP = std::get<0>(LUP_Tuple);
  std::vector<std::vector<double>> U_LUP = std::get<1>(LUP_Tuple);
  std::vector<std::vector<double>> P_LUP = std::get<2>(LUP_Tuple);
  printVec("LUP Decomp, L: ", L_LUP);
  printVec("LUP Decomp, U: ", U_LUP);
  printVec("LUP Decomp, P: ", P_LUP);
  //   printVec("LUP Decomp, L * U: ", multiply(L_LUP, U_LUP));
  //   printVec("LUP Decomp, P * Hilbert: ", multiply(P_LUP, hilbertMatrix));

  // Decompose Hilbert to Cholesky
  std::vector<std::vector<double>> L_cholesky =
      choleskyDecomposition(hilbertMatrix);
  printVec("Cholesky Decomp, L: ", L_cholesky);
  //   printVec("Cholesky Decomp, L * L^T: ",
  //            multiply(L_cholesky, transpose(L_cholesky)));

  std::vector<double> y1 = get_y_single_column(n, y_single_col_idx);
  std::vector<double> x1 = std::vector<double>(n, 0.0);
  x1[y_single_col_idx - 1] = 1.0;

  std::vector<double> y2 =
      get_y_linear_combination(n, y_linear_comb_col1, y_linear_comb_col1_coef,
                               y_linear_comb_col2, y_linear_comb_col2_coef);
  std::vector<double> x2 = std::vector<double>(n, 0.0);
  x2[y_linear_comb_col1 - 1] = y_linear_comb_col1_coef;
  x2[y_linear_comb_col2 - 1] = y_linear_comb_col2_coef;

  // Now solve the system for both y1 and y2
  std::vector<double> x1_LU = solve_LU(L, U, y1);
  std::vector<double> x2_LU = solve_LU(L, U, y2);
  std::vector<double> x1_LUP = solve_LUP(L_LUP, U_LUP, P_LUP, y1);
  std::vector<double> x2_LUP = solve_LUP(L_LUP, U_LUP, P_LUP, y2);
  std::vector<double> x1_cholesky = solve_cholesky(L_cholesky, y1);
  std::vector<double> x2_cholesky = solve_cholesky(L_cholesky, y2);

  // Compute the error for all four solutions and print.
  printVec("LU Error for single column: ", subtract(x1, x1_LU));
  printVec("LU Error for linear combination: ", subtract(x2, x2_LU));
  printVec("LUP Error for single column: ", subtract(x1, x1_LUP));
  printVec("LUP Error for linear combination: ", subtract(x2, x2_LUP));
  printVec("Cholesky Error for single column: ", subtract(x1, x1_cholesky));
  printVec("Cholesky Error for linear combination: ",
           subtract(x2, x2_cholesky));

  // Perform jacobi iteritive method on the discrete laplacian on a square
  auto pair = get_discrete_laplacian_matrix(2);

  auto eigenvector_1 = std::vector<double>{1, -1, -1, 1};
  auto eigenvector_2 = std::vector<double>{1, 1, 1, 1};
  auto eigenvector_3 = std::vector<double>{0, -1, 1, 0};
  auto eigenvector_4 = std::vector<double>{-1, 0, 0, 1};

  auto eigenvectors = std::vector<std::vector<double>>{
      eigenvector_1, eigenvector_2, eigenvector_3, eigenvector_4};

  auto compressed_laplacian = pair.first;
  auto compressed_laplacian_indices = pair.second;

  auto tuple = compressed_DR_seperation(compressed_laplacian,
                                        compressed_laplacian_indices);
  auto D = std::get<0>(tuple);
  auto R = std::get<1>(tuple);
  auto R_indices = std::get<2>(tuple);

  auto inv_D = invert_diagonal_matrix(D);

  std::vector<double> x0 = {0, 0, 0, 0};

  // For each eigenvector, solve the system using the Jacobi method, and
  // estimate the eigenvalue.
  for (int i = 0; i < eigenvectors.size(); i++) {
    auto y = eigenvectors[i];
    auto x = x0;
    // Do 50 itteraions of the Jacobi method
    for (int j = 0; j < 50; j++) {
      x = sparse_single_jacobi_itteration(inv_D, R, R_indices, y, x);
    }
    // Estimate the eigenvalue
    auto eigenvalue = dot_product(y, y) / dot_product(y, x);
    std::cout << "Eigenvalue " << i + 1 << " Jacobi estimate is: " << eigenvalue
              << std::endl;
  }

  // For each eigenvector, solve the system using the Gauss siedel method, and
  // estimate the eigenvalue.
  for (int i = 0; i < eigenvectors.size(); i++) {
    auto y = eigenvectors[i];
    auto x = x0;
    // Do 50 itteraions of the Gauss siedel method
    for (int j = 0; j < 50; j++) {
      x = sparse_single_gauss_seidel_itteration(
          compressed_laplacian, compressed_laplacian_indices, x, y);
    }
    // Estimate the eigenvalue
    auto eigenvalue = dot_product(y, y) / dot_product(y, x);
    std::cout << "Eigenvalue " << i + 1
              << " Gauss Seidel estimate is: " << eigenvalue << std::endl;
  }

  std::vector<double> eigenvalues = {-6.0, -2.0, -4.0, -4.0};
  std::vector<std::vector<double>> expected_x_per_eigenvector = {
      multiply(1 / -6.0, eigenvector_1), multiply(1 / -2.0, eigenvector_2),
      multiply(1 / -4.0, eigenvector_3), multiply(1 / -4.0, eigenvector_4)};

  // For each eigenvector, solve the system using the Jacobi method, and examine
  // and print convergence rates for each eigenvector
  for (int i = 0; i < eigenvectors.size(); i++) {
    auto y = eigenvectors[i];
    auto x = x0;
    std::vector<double> x_expected = expected_x_per_eigenvector[i];
    int iterations = 0;
    double error_norm = 0.0;
    double epsilon = 1e-25; // Convergence threshold
    do {
      x = sparse_single_jacobi_itteration(inv_D, R, R_indices, y, x);
      auto error = subtract(x, x_expected);
      error_norm = dot_product(error, error);
      iterations++;
      //   std::cout << "Error norm for eigenvector " << i + 1 << " after "
      //             << iterations << " iterations is: " << error_norm <<
      //             std::endl;
    } while (error_norm > epsilon);
    std::cout << "Convergence achieved for eigenvector " << i + 1 << " after "
              << iterations << " Jacobi iterations with epsilon " << epsilon
              << "." << std::endl;
  }

  // For each eigenvector, solve the system using the Gauss siedel method, and
  // examine and print convergence rates for each eigenvector
  for (int i = 0; i < eigenvectors.size(); i++) {
    auto y = eigenvectors[i];
    auto x = x0;
    std::vector<double> x_expected = expected_x_per_eigenvector[i];
    int iterations = 0;
    double error_norm = 0.0;
    double epsilon = 1e-25; // Convergence threshold
    do {
      x = sparse_single_gauss_seidel_itteration(
          compressed_laplacian, compressed_laplacian_indices, x, y);
      auto error = subtract(x, x_expected);
      error_norm = dot_product(error, error);
      iterations++;
      //   std::cout << "Error norm for eigenvector " << i + 1 << " after "
      //             << iterations << " iterations is: " << error_norm <<
      //             std::endl;
    } while (error_norm > epsilon);
    std::cout << "Convergence achieved for eigenvector " << i + 1 << " after "
              << iterations << " Gauss Seidel iterations with epsilon "
              << epsilon << "." << std::endl;
  }

  // Confirm that each pair of eigenvectors are orthogonal, and print
  // confirmation.
  for (int i = 0; i < eigenvectors.size(); i++) {
    for (int j = i + 1; j < eigenvectors.size(); j++) {
      auto dot_product_result = dot_product(eigenvectors[i], eigenvectors[j]);
      if (dot_product_result == 0.0) {
        std::cout << "Eigenvectors " << i + 1 << " and " << j + 1
                  << " are orthogonal." << std::endl;
      } else {
        std::cout << "Eigenvectors " << i + 1 << " and " << j + 1
                  << " are not orthogonal." << std::endl;
      }
    }
  }
  return 0;
}
