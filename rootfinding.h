#include <functional>

double bisection(std::function<double(double)> function, double lowerBound,
                 double upperBound, double tolerance, int maxIterations,
                 int &numIterations);

double falsePosition(std::function<double(double)> function, double lowerBound,
                     double upperBound, double tolerance, int maxIterations,
                     int &numIterations);

double secant(std::function<double(double)> function, double x0, double x1,
              double tolerance, int maxIterations, int &numIterations);

double newton(std::function<double(double)> function,
              std::function<double(double)> derivative, double x0,
              double tolerance, int maxIterations, int &numIterations);

double steffensen(std::function<double(double)> function, double x0,
                  double tolerance, int maxIterations, int &numIterations);

double run_all_rootfinding_methods(std::function<double(double)> function,
                                   std::function<double(double)> derivative,
                                   double lowerBound, double upperBound,
                                   double x0, double tolerance,
                                   int maxIterations);

std::vector<double> newton_horners(std::vector<double> coefficients, double x0,
                                   double tolerance, int maxIterations,
                                   int &numIterations);

std::vector<double> polynomialDivision(const std::vector<double> &coefficients,
                                       double root);
