#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

void printVec(const std::string &label,
              const std::vector<std::vector<double>> &matrix);
void printVec(const std::string &label, const std::vector<double> &vec);
void printVec(const std::string &label, double value);

#endif // UTILS_H