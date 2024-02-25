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

std::vector<double> multiply(const std::vector<std::vector<double>> &A,
                             const std::vector<double> &x);

const std::vector<std::vector<double>>
multiply(double k, const std::vector<std::vector<double>> &A);

const std::vector<double> multiply(double k, const std::vector<double> &A);

std::vector<double> subtract(const std::vector<double> &A,
                             const std::vector<double> &B);

std::vector<std::vector<double>>
transpose(const std::vector<std::vector<double>> &A);

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
DR_seperation(const std::vector<std::vector<double>> &A);

std::tuple<std::vector<double>, std::vector<std::vector<double>>,
           std::vector<std::vector<int>>>
compressed_DR_seperation(const std::vector<std::vector<double>> &A,
                         const std::vector<std::vector<int>> &A_indices);

std::vector<std::vector<double>>
invert_diagonal_matrix(const std::vector<std::vector<double>> &D);

std::vector<double> invert_diagonal_matrix(const std::vector<double> &D);

std::vector<double> multiply_sparse(const std::vector<std::vector<double>> &A,
                                    const std::vector<std::vector<int>> &A_idxs,
                                    const std::vector<double> &x);

std::vector<double> sparse_single_jacobi_itteration(
    const std::vector<double> &inv_D, const std::vector<std::vector<double>> &R,
    const std::vector<std::vector<int>> &R_indices,
    const std::vector<double> &y, const std::vector<double> &x_k);

std::vector<double> sparse_single_gauss_seidel_itteration(
    const std::vector<std::vector<double>> &A,
    const std::vector<std::vector<int>> &A_indices,
    const std::vector<double> &x_k, const std::vector<double> &y);

std::vector<double>
single_jacobi_itteration(const std::vector<std::vector<double>> &inv_D,
                         const std::vector<std::vector<double>> &R,
                         const std::vector<double> &y,
                         const std::vector<double> &x_k);

std::vector<std::vector<std::pair<double, double>>> get_interior_points(int n);

std::vector<std::vector<double>>
get_laplacian_approximation(const std::vector<std::vector<double>> &X,
                            const std::vector<std::vector<double>> &Y);

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>>>
get_discrete_laplacian_matrix(int size);

double dot_product(const std::vector<double> &A, const std::vector<double> &B);

#endif