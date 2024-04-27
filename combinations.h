
#include <functional>
#include <vector>

std::vector<double> get_legendre_coefficients(int n);

std::vector<double> get_tchebyshev_coefficients(int n);

std::vector<double>
get_interpolation_coefficients(std::vector<double> &roots,
                               std::function<double(double)> f);

std::vector<double> get_quadrature_weights(std::vector<double> &roots,
                                           std::function<double(double)> f);

double quadrature(std::vector<double> &roots, std::vector<double> &weights,
                  std::function<double(double)> f);
