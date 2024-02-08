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
  return 0;
}
