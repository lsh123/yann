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

#include "core/utils.h"
#include "core/types.h"
#include "core/functions.h"

using namespace std;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Activation functions implementation
//

// identity function:
//  f(x) = x
//  d f(x<i>) / d(x<j>) = 1 if i == j and 0 if i != j
string yann::IdentityFunction::get_info() const
{
  return "Identity";
}
void yann::IdentityFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  YANN_SLOW_CHECK(is_same_size(input, output));
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
  YANN_SLOW_CHECK(is_same_size(input, output));
  output.setOnes();
}
unique_ptr<ActivationFunction> yann::IdentityFunction::copy() const
{
  return make_unique<IdentityFunction>();
}

// rectified linear unit function or leaky ReLU if a != 0:
//  f(x) = x for x > 0 ; a*x for x < 0
//  d f(x<i>) / d(x<j>) = 1 if x > 0; a if x < 0
string yann::ReluFunction::get_info() const
{
  return "Relu";
}
void yann::ReluFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  YANN_SLOW_CHECK(is_same_size(input, output));

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
  YANN_SLOW_CHECK(is_same_size(input, output));

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
string yann::SigmoidFunction::get_info() const
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
  YANN_SLOW_CHECK(is_same_size(input, output));

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
  YANN_SLOW_CHECK(is_same_size(input, output));
  this->f(input, output);
  output.array() =  output.array() * (1 -  output.array());
}
unique_ptr<ActivationFunction> yann::SigmoidFunction::copy() const
{
  return make_unique<SigmoidFunction>();
}


// Fast sigmoid function:
//  f(x) = 1 / (1 + exp(-x))
yann::FastSigmoidFunction::FastSigmoidFunction(const size_t & table_size, const Value & max_value) :
    _table(table_size + 1),
    _max_value(max_value)
{
  // populates _table with sigmoid(x) values for x in (-max_value, max_value)
  for(size_t ii = 0; ii <= table_size; ++ii) {
    _table[ii] = SigmoidFunction::sigmoid_scalar((2 * ii / (Value)table_size - 1) * max_value);
  }
}

yann::FastSigmoidFunction::FastSigmoidFunction(const SigmoidTable & table, const Value & max_value) :
    _table(table),
    _max_value(max_value)
{
}

string yann::FastSigmoidFunction::get_info() const
{
  ostringstream oss;
  oss << "FastSigmoidFunction["
      << "max_value=" << _max_value
      << ", table_size=" << _table.size()
      << "]"
      ;
  return oss.str();
}

void yann::FastSigmoidFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  YANN_SLOW_CHECK(is_same_size(input, output));

  auto sigmoid = [&](const Value & xx) -> Value {
    if(xx <= -_max_value) {
      return 0;
    } else if(xx >= _max_value) {
      return 1;
    }
    auto index = (size_t)((xx + _max_value) * (_table.size() - 1) / (2 * _max_value));
    YANN_SLOW_CHECK_GE(index, 0);
    YANN_SLOW_CHECK_LT(index, _table.size());
    return _table[index];
  };

  switch(mode) {
  case Operation_Assign:
    for(MatrixSize ii = 0; ii < input.rows(); ++ii) {
      for(MatrixSize jj = 0; jj < input.cols(); ++jj) {
        output(ii, jj) = sigmoid(input(ii, jj));
      }
    }
    break;
  case Operation_PlusEqual:
    for(MatrixSize ii = 0; ii < input.rows(); ++ii) {
      for(MatrixSize jj = 0; jj < input.cols(); ++jj) {
        output(ii, jj) += sigmoid(input(ii, jj));
      }
    }
    break;
  }
}

void yann::FastSigmoidFunction::derivative(const RefConstVectorBatch & input, RefVectorBatch output)
{
  YANN_SLOW_CHECK(is_same_size(input, output));
  this->f(input, output);
  output.array() =  output.array() * (1 -  output.array());
}
unique_ptr<ActivationFunction> yann::FastSigmoidFunction::copy() const
{
  // can't use make_unique<> because this constructor is private
  auto res = new FastSigmoidFunction(_table, _max_value);
  return unique_ptr<ActivationFunction>(res);
}

// tanh function:
//  f(x) = A * tanh(S*x)
//  df/dx = A * S * (1−(tanh(S*x))^2)
string yann::TanhFunction::get_info() const
{
  ostringstream oss;
  oss << "Tanh["
      << "A=" << _AA
      << ", S=" << _SS
      << "]"
      ;
  return oss.str();
}

void yann::TanhFunction::f(const RefConstVectorBatch & input, RefVectorBatch output, enum OperationMode mode)
{
  YANN_SLOW_CHECK(is_same_size(input, output));

  switch(mode) {
  case Operation_Assign:
    output.array() = _AA * tanh(input.array() * _SS);
    break;
  case Operation_PlusEqual:
    output.array() += _AA * tanh(input.array()* _SS);
    break;
  }
}

void yann::TanhFunction::derivative(const RefConstVectorBatch & input, RefVectorBatch output)
{
  YANN_SLOW_CHECK(is_same_size(input, output));
  this->f(input, output);
  output.array() = (1 - tanh(input.array()* _SS).square()) * (_AA * _SS);
}
unique_ptr<ActivationFunction> yann::TanhFunction::copy() const
{
  return make_unique<TanhFunction>(_AA, _SS);
}

// quadratic cost function:
//  f(actual, expected) = sum((actual<i> - expected<i>)^2)
//  d f(actual, expected) / d (actual) = (actual<i> - expected<i>)
string yann::QuadraticCost::get_info() const
{
  return "Quadratic";
}
Value yann::QuadraticCost::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  YANN_SLOW_CHECK(is_same_size(actual, expected));
  return ((actual.array() - expected.array()).square()).sum();
}
void yann::QuadraticCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  YANN_SLOW_CHECK(is_same_size(actual, expected));
  YANN_SLOW_CHECK(is_same_size(actual, output));
  output.noalias() = 2 * (actual - expected);
}
unique_ptr<CostFunction> yann::QuadraticCost::copy() const
{
  return make_unique<QuadraticCost>();
}

// Exponential cost:
//  f(actual, expected) = tau * exp(sum((actual<i> - expected<i>)^2) / tau)
//  d f(actual, expected) / d (actual) = 2 * f(actual, expected) * (actual<i> - expected<i>)
string yann::ExponentialCost::get_info() const
{
  ostringstream oss;
  oss << "Exponential["
      << "tau=" << _tau
      << "]"
      ;
  return oss.str();
}
Value yann::ExponentialCost::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  YANN_SLOW_CHECK(is_same_size(actual, expected));
  const auto val = (actual.array() - expected.array()).square().sum();
  return _tau * exp(val / _tau);
}
void yann::ExponentialCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  YANN_SLOW_CHECK(is_same_size(actual, expected));
  YANN_SLOW_CHECK(is_same_size(actual, output));
  output.noalias() = (2 * f(actual, expected)) * (actual - expected);
}
unique_ptr<CostFunction> yann::ExponentialCost::copy() const
{
  return make_unique<ExponentialCost>(_tau);
}

// cross entropy function:
//  f(actual, expected) = sum(-(expected * ln(actual) + (1 - expected) * ln(1 - actual)))
//  d f(actual, expected) / d (actual) = (actual - expected) / ((1 - actual) * actual)
string yann::CrossEntropyCost::get_info() const
{
  ostringstream oss;
  oss << "CrossEntropy["
      << "epsilon=" << _epsilon
      << "]"
      ;
  return oss.str();
}

Value yann::CrossEntropyCost::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  YANN_SLOW_CHECK(is_same_size(actual, expected));

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
  YANN_SLOW_CHECK(is_same_size(actual, expected));
  YANN_SLOW_CHECK(is_same_size(actual, output));

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
string yann::HellingerDistanceCost::get_info() const
{
  ostringstream oss;
  oss << "HellingerDistance["
      << "epsilon=" << _epsilon
      << "]"
      ;
  return oss.str();
}

Value yann::HellingerDistanceCost::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  YANN_SLOW_CHECK(is_same_size(actual, expected));
  return (((actual.array() + _epsilon).sqrt() - expected.array().sqrt()).square()).sum();
}

void yann::HellingerDistanceCost::derivative(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected, RefVectorBatch output)
{
  YANN_SLOW_CHECK(is_same_size(actual, expected));
  YANN_SLOW_CHECK(is_same_size(actual, output));
  output.array() = (1 - expected.array().sqrt() / (actual.array() + _epsilon).sqrt());
}
unique_ptr<CostFunction> yann::HellingerDistanceCost::copy() const
{
  return make_unique<HellingerDistanceCost>(_epsilon);
}

// Squared Hinge Loss:
//  f(actual, expected) = (max(0, 1 - actual * expected))^2
//  d f(actual, expected) / d (actual(i)) =  if (actual * expected) > 1 then 0; otherwise -2 * (1 - actual * expected) * expected(i)
string yann::SquaredHingeLoss::get_info() const
{
  return "SquaredHingeLoss";
}
Value yann::SquaredHingeLoss::f(const RefConstVectorBatch & actual, const RefConstVectorBatch & expected)
{
  YANN_SLOW_CHECK(is_same_size(actual, expected));

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
  YANN_SLOW_CHECK(is_same_size(actual, expected));
  YANN_SLOW_CHECK(is_same_size(actual, output));

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

