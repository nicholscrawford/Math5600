#include "utils.h"
#include <iomanip>
#include <iostream>

void printVec(const std::string &label,
              const std::vector<std::vector<double>> &matrix) {
  std::cout << "\033[1m" << label << "\033[0m" << std::endl;
  for (const auto &row : matrix) {
    for (const auto &element : row) {
      std::cout << std::setprecision(5) << std::scientific << element
                << "\t"; // Round to 5 significant digits
    }
    std::cout << "\n";
  }
}

void printVec(const std::string &label, const std::vector<double> &vec) {
  std::cout << "\033[1m" << label << "\033[0m" << std::endl;
  for (const auto &element : vec) {
    std::cout << std::setprecision(4) << std::scientific << element
              << "\t"; // Round to 5 significant digits
  }
  std::cout << "\n";
}