/*
 * functions.h
 *
 */

#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_


#include "core/layer.h"
#include "core/nn.h"
#include "core/types.h"

namespace yann {

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Activation functions
//

// identity function:
//  f(x) = x
//  d f(x<i>) / d(x<j>) = 1 if i == j and 0 if i != j
class IdentityFunction: public ActivationFunction {
public:
  virtual std::string get_info() const;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign);
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output);
  virtual std::unique_ptr<ActivationFunction> copy() const;
}; // class IdentityFunction

// rectified linear unit function or leaky ReLU if a != 0:
//  f(x) = x for x > 0 ; a*x for x < 0
//  d f(x<i>) / d(x<j>) = 1 if x > 0; a if x < 0
class ReluFunction: public ActivationFunction {
public:
  ReluFunction(const Value & a = 0.0) : _a(a) { }

  virtual std::string get_info() const;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign);
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output);
  virtual std::unique_ptr<ActivationFunction> copy() const;

private:
  const Value _a;
}; // class ReluFunction

// sigmoid function:
//  f(x) = 1 / (1 + exp(-x))
class SigmoidFunction: public ActivationFunction {
public:
  virtual std::string get_info() const;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign);
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output);
  virtual std::unique_ptr<ActivationFunction> copy() const;

public:
  static Value sigmoid_scalar(const Value & x);
  static Value sigmoid_derivative_scalar(const Value & x);
}; // class SigmoidFunction


// fast sigmoid function:
//  f(x) = 1 / (1 + exp(-x))
// Approximation is done through a pre-calculated table
class FastSigmoidFunction: public ActivationFunction {
  typedef std::vector<Value> SigmoidTable;

public:
  FastSigmoidFunction(const size_t & table_size = 1000, const Value & max_value = 6);

  virtual std::string get_info() const;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign);
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output);
  virtual std::unique_ptr<ActivationFunction> copy() const;

private:
  FastSigmoidFunction(const SigmoidTable & table, const Value & max_value);

private:
  SigmoidTable _table;
  const Value  _max_value;
}; // class FastSigmoidFunction

// tanh function:
//  f(x) = A*tanh(S*x)
//  df/dx = A * S * (1−(tanh(S*x))^2)
class TanhFunction: public ActivationFunction {
public:
  TanhFunction(const Value & AA = 1.0, const Value & SS = 1.0) : _AA(AA), _SS(SS) { }
  virtual std::string get_info() const;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign);
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output);
  virtual std::unique_ptr<ActivationFunction> copy() const;

private:
  const Value _AA;
  const Value _SS;
}; // class TanhFunction


////////////////////////////////////////////////////////////////////////////////////////////////
//
// Cost functions
//

// quadratic cost:
//  f(actual, expected) = sum((actual<i> - expected<i>)^2)
//  d f(actual, expected) / d (actual) = (actual<i> - expected<i>)    -- we drop 1/2 coefficient since it doesn't matter
class QuadraticCost: public CostFunction {
public:
  virtual std::string get_info() const;
  virtual Value f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected);
  virtual void derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output);
  virtual std::unique_ptr<CostFunction> copy() const;
}; // class QuadraticCost

// Exponential cost:
//  f(actual, expected) = tau * exp(sum((actual<i> - expected<i>)^2) / tau)
//  d f(actual, expected) / d (actual) = 2 * f(actual, expected) * (actual<i> - expected<i>)
class ExponentialCost: public CostFunction {
public:
  ExponentialCost(const Value & tau = 1.0) : _tau(tau) { }

  virtual std::string get_info() const;
  virtual Value f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected);
  virtual void derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output);
  virtual std::unique_ptr<CostFunction> copy() const;

private:
  const Value _tau;
}; // class ExponentialCost

// cross entropy cost:
//  f(actual, expected) = sum(-(expected * ln(actual) + (1 - expected) * ln(1 - actual)))
//  d f(actual, expected) / d (actual) = (actual - expected) / ((1 - actual) * actual)
class CrossEntropyCost: public CostFunction {
public:
  CrossEntropyCost(const Value & epsilon = 1.0e-10) : _epsilon(epsilon) { }
  virtual std::string get_info() const;
  virtual Value f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected);
  virtual void derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output);
  virtual std::unique_ptr<CostFunction> copy() const;

private:
  const Value _epsilon;
}; // class CrossEntropyCost

// Hellinger distance cost (we drop coefficients since they don't matter):
//  f(actual, expected) = sum(sqrt(actual) - sqrt(expected))^2
//  d f(actual, expected) / d (actual) =  (1 -  sqrt(expected) /  sqrt(actual))
class HellingerDistanceCost: public CostFunction {
public:
  // epsilon shifts the values to avoid division by 0 in the derivative
  HellingerDistanceCost(const Value & epsilon = 0) : _epsilon(epsilon) { }

  virtual std::string get_info() const;
  virtual Value f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected);
  virtual void derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output);
  virtual std::unique_ptr<CostFunction> copy() const;

private:
  const Value _epsilon;
}; // class HellingerDistanceCost


// Squared Hinge Loss:
//  f(actual, expected) = (max(0, 1 - actual * expected))^2
//  d f(actual, expected) / d (actual(i)) =  if (actual * expected) > 1 then 0; otherwise -2 * (1 - actual * expected) * expected(i)
class SquaredHingeLoss: public CostFunction {
public:
  virtual std::string get_info() const;
  virtual Value f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected);
  virtual void derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output);
  virtual std::unique_ptr<CostFunction> copy() const;
}; // class SquaredHingeLoss

}; // namespace yann

#endif /* FUNCTIONS_H_ */
