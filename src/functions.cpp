  /*
 * functions.cpp
 *
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>

#include <boost/assert.hpp>

#include "utils.h"
#include "types.h"
#include "functions.h"

using namespace std;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Activation functions implementation
//

// identity function:
//  f(x) = x
//  d f(x<i>) / d(x<j>) = 1 if i == j and 0 if i != j
string yann::IdentityFunction::get_name() const
{
  return "Identity";
}
void yann::IdentityFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  BOOST_VERIFY(is_same_size(input, output));
  switch(mode) {
  case Operation_Assign:
    output.noalias() = input;
    break;
  case Operation_PlusEqual:
    output.noalias() += input;
    break;
  }
}
void yann::IdentityFunction::derivative(const RefConstVectorBatch & input, RefVectorBatch output)
{
  BOOST_VERIFY(is_same_size(input, output));
  output.setOnes();
}
unique_ptr<ActivationFunction> yann::IdentityFunction::copy() const
{
  return make_unique<IdentityFunction>();
}

// rectified linear unit function:
//  f(x) = x for x > 0 ; 0 for x < 0
//  d f(x<i>) / d(x<j>) = 1 if i == j and (x<i>) > 0 and 0 if i != j
string yann::ReluFunction::get_name() const
{
  return "Relu";
}
void yann::ReluFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  BOOST_VERIFY(is_same_size(input, output));

  switch(mode) {
  case Operation_Assign:
    output.setZero();
    break;
  case Operation_PlusEqual:
    // do nothing
    break;
  }

  const auto batch_size = get_batch_size(output);
  const auto batch_item_size = get_batch_item_size(output);
  for(MatrixSize ii = 0 ; ii < batch_size; ++ii) {
    for(MatrixSize jj = 0 ; jj < batch_item_size; ++jj) {
      if(get_batch(output, ii)(jj) >= 0) {
        get_batch(output, ii)(jj) += get_batch(input, ii)(jj);
      }
    }
  }
}
void yann::ReluFunction::derivative(const RefConstVectorBatch & input, RefVectorBatch output)
{
  BOOST_VERIFY(is_same_size(input, output));

  const auto batch_size = get_batch_size(output);
  const auto batch_item_size = get_batch_item_size(output);
  for(MatrixSize ii = 0 ; ii < batch_size; ++ii) {
    for(MatrixSize jj = 0 ; jj < batch_item_size; ++jj) {
      if(get_batch(output, ii)(jj) >= 0) {
        get_batch(output, ii)(jj) =  1;
      } else {
        get_batch(output, ii)(jj) = 0;
      }
    }
  }
}
unique_ptr<ActivationFunction> yann::ReluFunction::copy() const
{
  return make_unique<ReluFunction>();
}

// sigmoid function:
//  f(x) = 1 / (1 + exp(-x))
string yann::SigmoidFunction::get_name() const
{
  return "Sigmoid";
}
Value yann::SigmoidFunction::sigmoid_scalar(const Value & x)
{
  return 1 / (1 + exp(-x));
}

Value yann::SigmoidFunction::sigmoid_derivative_scalar(const Value & x)
{
  Value s = sigmoid_scalar(x);
  return s * (1 - s);
}

void yann::SigmoidFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  BOOST_VERIFY(is_same_size(input, output));

  switch(mode) {
  case Operation_Assign:
    output.array() = 1 / (1 + exp(-input.array()));
    break;
  case Operation_PlusEqual:
    output.array() += 1 / (1 + exp(-input.array()));
    break;
  }
}

void yann::SigmoidFunction::derivative(const RefConstVectorBatch & input, RefVectorBatch output)
{
  BOOST_VERIFY(is_same_size(input, output));
  this->f(input, output);
  output.array() =  output.array() * (1 -  output.array());
}
unique_ptr<ActivationFunction> yann::SigmoidFunction::copy() const
{
  return make_unique<SigmoidFunction>();
}

// tanh function:
//  f(x) = tanh(x)
//  df/dx = 1−(tanh(x))^2
string yann::TanhFunction::get_name() const
{
  return "Tanh";
}

void yann::TanhFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  BOOST_VERIFY(is_same_size(input, output));

  switch(mode) {
  case Operation_Assign:
    output.array() = tanh(input.array());
    break;
  case Operation_PlusEqual:
    output.array() += tanh(input.array());
    break;
  }
}

void yann::TanhFunction::derivative(const RefConstVectorBatch & input, RefVectorBatch output)
{
  BOOST_VERIFY(is_same_size(input, output));
  this->f(input, output);
  output.array() = 1 - (output.array() * output.array());
}
unique_ptr<ActivationFunction> yann::TanhFunction::copy() const
{
  return make_unique<TanhFunction>();
}

// quadratic cost function:
//  f(actual, expected) = sum((actual<i> - expected<i>)^2)
//  d f(actual, expected) / d (actual) = (actual<i> - expected<i>)    -- we drop 1/2 coefficient since it doesn't matter
string yann::QuadraticCost::get_name() const
{
  return "Quadratic";
}
Value yann::QuadraticCost::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  BOOST_VERIFY(is_same_size(actual, expected));
  return ((actual.array() - expected.array()).square()).sum();
}
void yann::QuadraticCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  BOOST_VERIFY(is_same_size(actual, expected));
  BOOST_VERIFY(is_same_size(actual, output));
  output.noalias() = actual - expected;
}
unique_ptr<CostFunction> yann::QuadraticCost::copy() const
{
  return make_unique<QuadraticCost>();
}

// cross entropy function:
//  f(actual, expected) = sum(-(expected * ln(actual) + (1 - expected) * ln(1 - actual)))
//  d f(actual, expected) / d (actual) = (actual - expected) / ((1 - actual) * actual)
string yann::CrossEntropyCost::get_name() const
{
  return "CrossEntropy";
}
Value yann::CrossEntropyCost::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  BOOST_VERIFY(is_same_size(actual, expected));
  return -(expected.array() * log(actual.array()) + (1 - expected.array()) * log(1 - actual.array())).sum();
}

void yann::CrossEntropyCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  BOOST_VERIFY(is_same_size(actual, expected));
  BOOST_VERIFY(is_same_size(actual, output));
  output.array() = (actual.array() - expected.array()) / ((1 - actual.array()) * actual.array());
}
unique_ptr<CostFunction> yann::CrossEntropyCost::copy() const
{
  return make_unique<CrossEntropyCost>();
}

// Hellinger distance cost:
//  f(actual, expected) = sum(sqrt(actual) - sqrt(expected))^2 / sqrt(2)
//  d f(actual, expected) / d (actual) =  sqrt(2) * (1 -  sqrt(expected) /  sqrt(actual))
string yann::HellingerDistanceCost::get_name() const
{
  return "HellingerDistance";
}
Value yann::HellingerDistanceCost::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  BOOST_VERIFY(is_same_size(actual, expected));
  return ((actual.array().sqrt() - expected.array().sqrt()).square()).sum();
}

void yann::HellingerDistanceCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  BOOST_VERIFY(is_same_size(actual, expected));
  BOOST_VERIFY(is_same_size(actual, output));
  output.array() = (1 - expected.array().sqrt() / actual.array().sqrt());
}
unique_ptr<CostFunction> yann::HellingerDistanceCost::copy() const
{
  return make_unique<HellingerDistanceCost>();
}

