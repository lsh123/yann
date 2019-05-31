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
  YANN_CHECK(is_same_size(input, output));
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
  YANN_CHECK(is_same_size(input, output));
  output.setOnes();
}
unique_ptr<ActivationFunction> yann::IdentityFunction::copy() const
{
  return make_unique<IdentityFunction>();
}

// rectified linear unit function or leaky ReLU if a != 0:
//  f(x) = x for x > 0 ; a*x for x < 0
//  d f(x<i>) / d(x<j>) = 1 if x > 0; a if x < 0
string yann::ReluFunction::get_name() const
{
  return "Relu";
}
void yann::ReluFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  YANN_CHECK(is_same_size(input, output));

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
      if(get_batch(input, ii)(jj) >= 0) {
        get_batch(output, ii)(jj) += get_batch(input, ii)(jj);
      } else {
        get_batch(output, ii)(jj) += _a * get_batch(input, ii)(jj);
      }
    }
  }
}
void yann::ReluFunction::derivative(const RefConstVectorBatch & input, RefVectorBatch output)
{
  YANN_CHECK(is_same_size(input, output));

  const auto batch_size = get_batch_size(output);
  const auto batch_item_size = get_batch_item_size(output);
  for(MatrixSize ii = 0 ; ii < batch_size; ++ii) {
    for(MatrixSize jj = 0 ; jj < batch_item_size; ++jj) {
      if(get_batch(input, ii)(jj) >= 0) {
        get_batch(output, ii)(jj) = 1;
      } else {
        get_batch(output, ii)(jj) = _a;
      }
    }
  }
}
unique_ptr<ActivationFunction> yann::ReluFunction::copy() const
{
  return make_unique<ReluFunction>(_a);
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
  YANN_CHECK(is_same_size(input, output));

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
  YANN_CHECK(is_same_size(input, output));
  this->f(input, output);
  output.array() =  output.array() * (1 -  output.array());
}
unique_ptr<ActivationFunction> yann::SigmoidFunction::copy() const
{
  return make_unique<SigmoidFunction>();
}

// tanh function:
//  f(x) = A*tanh(S*x)
//  df/dx = A* S* (1−(tanh(S*x))^2)
string yann::TanhFunction::get_name() const
{
  return "Tanh";
}

void yann::TanhFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  YANN_CHECK(is_same_size(input, output));

  switch(mode) {
  case Operation_Assign:
    output.array() = _A * tanh(input.array() * _S);
    break;
  case Operation_PlusEqual:
    output.array() += _A * tanh(input.array()* _S);
    break;
  }
}

void yann::TanhFunction::derivative(const RefConstVectorBatch & input, RefVectorBatch output)
{
  YANN_CHECK(is_same_size(input, output));
  this->f(input, output);
  output.array() = (1 - tanh(input.array()* _S).square()) * (_A * _S);
}
unique_ptr<ActivationFunction> yann::TanhFunction::copy() const
{
  return make_unique<TanhFunction>(_A, _S);
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
  YANN_CHECK(is_same_size(actual, expected));
  return ((actual.array() - expected.array()).square()).sum();
}
void yann::QuadraticCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  YANN_CHECK(is_same_size(actual, expected));
  YANN_CHECK(is_same_size(actual, output));
  output.noalias() = actual - expected;
}
unique_ptr<CostFunction> yann::QuadraticCost::copy() const
{
  return make_unique<QuadraticCost>();
}

// Exponential cost:
//  f(actual, expected) = thaw * exp(sum((actual<i> - expected<i>)^2) / thaw)
//  d f(actual, expected) / d (actual) = 2 * (actual<i> - expected<i>) * f(actual, expected) / thaw
string yann::ExponentialCost::get_name() const
{
  return "Exponential";
}
Value yann::ExponentialCost::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  YANN_CHECK(is_same_size(actual, expected));
  const auto val = (actual.array() - expected.array()).square().sum();
  return _thaw * exp(val / _thaw);
}
void yann::ExponentialCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  YANN_CHECK(is_same_size(actual, expected));
  YANN_CHECK(is_same_size(actual, output));
  output.noalias() = (2 * f(actual, expected) / _thaw) * (actual - expected);
}
unique_ptr<CostFunction> yann::ExponentialCost::copy() const
{
  return make_unique<ExponentialCost>(_thaw);
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
  YANN_CHECK(is_same_size(actual, expected));

  // iterate over elements manually and use small _epsilon to avoid hitting nan
  Value res = 0;
  for(MatrixSize ii = 0; ii < actual.rows(); ++ii) {
    for(MatrixSize jj = 0; jj < actual.cols(); ++jj) {
      auto aa = actual(ii, jj);
      auto ee = expected(ii, jj);
      if(aa <= 0) {
        aa = _epsilon;
      } else if(aa >= 1) {
        aa = 1 - _epsilon;
      }
      res += (ee * log(aa) + (1 - ee) * log(1 - aa));
    }
  }
  return -(res);
}

void yann::CrossEntropyCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  YANN_CHECK(is_same_size(actual, expected));
  YANN_CHECK(is_same_size(actual, output));

  // iterate over elements manually and use small _epsilon to avoid hitting nan
  for(MatrixSize ii = 0; ii < actual.rows(); ++ii) {
    for(MatrixSize jj = 0; jj < actual.cols(); ++jj) {
      auto aa = actual(ii, jj);
      auto ee = expected(ii, jj);
      if(fabs(aa - ee) > _epsilon) {
        if(aa <= 0) {
          aa = _epsilon;
        } else if(aa >= 1) {
          aa = 1 - _epsilon;
        }
        output(ii, jj) = (aa - ee) / ((1 - aa) * aa);
      } else {
        output(ii, jj) = 0;
      }
    }
  }
}

unique_ptr<CostFunction> yann::CrossEntropyCost::copy() const
{
  return make_unique<CrossEntropyCost>(_epsilon);
}

// Hellinger distance cost:
//  f(actual, expected) = sum(sqrt(actual) - sqrt(expected))^2
//  d f(actual, expected) / d (actual) =  (1 -  sqrt(expected) /  sqrt(actual))
string yann::HellingerDistanceCost::get_name() const
{
  return "HellingerDistance";
}
Value yann::HellingerDistanceCost::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  YANN_CHECK(is_same_size(actual, expected));
  return (((actual.array() + _epsilon).sqrt() - expected.array().sqrt()).square()).sum();
}

void yann::HellingerDistanceCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  YANN_CHECK(is_same_size(actual, expected));
  YANN_CHECK(is_same_size(actual, output));
  output.array() = (1 - expected.array().sqrt() / (actual.array() + _epsilon).sqrt());
}
unique_ptr<CostFunction> yann::HellingerDistanceCost::copy() const
{
  return make_unique<HellingerDistanceCost>(_epsilon);
}

// Squared Hinge Loss:
//  f(actual, expected) = (max(0, 1 - actual * expected))^2
//  d f(actual, expected) / d (actual(i)) =  if (actual * expected) > 1 then 0; otherwise -2 * (1 - actual * expected) * expected(i)
string yann::SquaredHingeLoss::get_name() const
{
  return "SquaredHingeLoss";
}
Value yann::SquaredHingeLoss::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  YANN_CHECK(is_same_size(actual, expected));

  Value res = 0;
  for(MatrixSize ii = 0; ii < get_batch_size(actual); ++ii) {
    const auto aa = get_batch(actual, ii);
    const auto ee = get_batch(expected, ii);

    const Value vv = (aa.array() * ee.array()).sum();
    if(vv < 1.0) {
      res += (1.0 - vv) * (1.0 - vv);
    }
  }
  return res;
}

void yann::SquaredHingeLoss::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  YANN_CHECK(is_same_size(actual, expected));
  YANN_CHECK(is_same_size(actual, output));

  for(MatrixSize ii = 0; ii < get_batch_size(actual); ++ii) {
    const auto aa = get_batch(actual, ii);
    const auto ee = get_batch(expected, ii);
    auto oo = get_batch(output, ii);

    const Value vv = (aa.array() * ee.array()).sum();
    if(vv < 1) {
      oo = - 2 * (1.0 - vv) * ee;
    } else {
      oo.setZero();
    }
  }
}

unique_ptr<CostFunction> yann::SquaredHingeLoss::copy() const
{
  return make_unique<SquaredHingeLoss>();
}

