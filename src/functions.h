/*
 * functions.h
 *
 */

#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_


#include "nnlayer.h"
#include "nn.h"
#include "types.h"

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
  virtual std::string get_name() const;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign);
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output);
  virtual std::unique_ptr<ActivationFunction> copy() const;
}; // class IdentityFunction

// rectified linear unit function:
//  f(x) = x for x > 0 ; 0 for x < 0
//  d f(x<i>) / d(x<j>) = 1 if i == j and (x<i>) > 0 and 0 if i != j
class ReluFunction: public ActivationFunction {
public:
  virtual std::string get_name() const;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign);
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output);
  virtual std::unique_ptr<ActivationFunction> copy() const;
}; // class ReluFunction

// sigmoid function:
//  f(x) = 1 / (1 + exp(-x))
class SigmoidFunction: public ActivationFunction {
public:
  virtual std::string get_name() const;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign);
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output);
  virtual std::unique_ptr<ActivationFunction> copy() const;

private:
  static Value sigmoid_scalar(const Value & x);
  static Value sigmoid_derivative_scalar(const Value & x);
}; // class SigmoidFunction

// tanh function:
//  f(x) = tanh(x)
//  df/dx = 1−(tanh(x))^2
class TanhFunction: public ActivationFunction {
public:
  virtual std::string get_name() const;
  virtual void f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode = Operation_Assign);
  virtual void derivative(const RefConstVectorBatch & input, RefVectorBatch output);
  virtual std::unique_ptr<ActivationFunction> copy() const;
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
  virtual std::string get_name() const;
  virtual Value f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected);
  virtual void derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output);
  virtual std::unique_ptr<CostFunction> copy() const;
}; // class QuadraticCost

// cross entropy cost:
//  f(actual, expected) = sum(-(expected * ln(actual) + (1 - expected) * ln(1 - actual)))
//  d f(actual, expected) / d (actual) = (actual - expected) / ((1 - actual) * actual)
class CrossEntropyCost: public CostFunction {
public:
  virtual std::string get_name() const;
  virtual Value f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected);
  virtual void derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output);
  virtual std::unique_ptr<CostFunction> copy() const;
}; // class CrossEntropyCost

// Hellinger distance cost:
//  f(actual, expected) = sum(sqrt(actual) - sqrt(expected))^2 / sqrt(2)
//  d f(actual, expected) / d (actual) =  sqrt(2) * (1 -  sqrt(expected) /  sqrt(actual))
class HellingerDistanceCost: public CostFunction {
public:
  virtual std::string get_name() const;
  virtual Value f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected);
  virtual void derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output);
  virtual std::unique_ptr<CostFunction> copy() const;
}; // class HellingerDistanceCost


}; // namespace yann

#endif /* FUNCTIONS_H_ */
