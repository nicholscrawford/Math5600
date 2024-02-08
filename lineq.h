#ifndef LINEQ_H
#define LINEQ_H

#include <utility>
#include <vector>

std::vector<std::vector<double>> get_hilbertian(int n);

std::vector<double> get_y_single_column(int n, int column_j);

std::vector<double> get_y_linear_combination(int n, int first_column_j,
                                             double first_column_coef,
                                             int second_column_j,
                                             double second_column_coef);

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_LU_decomposition(std::vector<std::vector<double>> &A);

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>,
           std::vector<std::vector<double>>>
get_LUP(std::vector<std::vector<double>> &A);

std::vector<std::vector<double>>
choleskyDecomposition(std::vector<std::vector<double>> A);

std::vector<double> solve_LU(std::vector<std::vector<double>> &L,
                             std::vector<std::vector<double>> &U,
                             std::vector<double> &y);

std::vector<double> solve_LUP(std::vector<std::vector<double>> &L,
                              std::vector<std::vector<double>> &U,
                              std::vector<std::vector<double>> &P,
                              std::vector<double> &y);

std::vector<double> solve_cholesky(std::vector<std::vector<double>> &L,
                                   std::vector<double> &y);

std::vector<std::vector<double>>
multiply(const std::vector<std::vector<double>> &A,
         const std::vector<std::vector<double>> &B);

std::vector<double> subtract(const std::vector<double> &A,
                             const std::vector<double> &B);

std::vector<std::vector<double>>
transpose(const std::vector<std::vector<double>> &A);

#endif