//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "Functions Tests"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/random.h"
#include "core/utils.h"
#include "core/functions.h"

#include "timer.h"
#include "test_utils.h"


using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace boost::numeric;
using namespace yann;
using namespace yann::test;


struct FunctionsTestFixture
{
  const size_t min_steps = 5;

  FunctionsTestFixture()
  {

  }
  ~FunctionsTestFixture()
  {

  }

  void softmax_vector(const RefConstVector & input, RefVector output, const Value & beta)
  {
    YANN_CHECK(is_same_size(input, output));
    YANN_CHECK_EQ(input.rows(), 1); // RowMajor layout, breaks for ColMajor

    Value max = input.maxCoeff(); // adjust the computations to avoid overflowing
    Value sum = exp((input.array() - max) * beta).sum();
    output.array() = (exp((input.array() - max) * beta)) / sum;
  }

  void softmax(const RefConstMatrix & input, RefMatrix output, const Value & beta = 3.0)
  {
    YANN_CHECK(is_same_size(input, output));
    for(MatrixSize ii = 0; ii < input.rows(); ++ii) {
      softmax_vector(input.row(ii), output.row(ii), beta);
    }
  }

  pair<Matrix, Value> test_cost_function(
      const unique_ptr<CostFunction> & cost_function,
      const RefConstMatrix & actual0, const RefConstMatrix & expected, const Value & cost0,
      double learning_rate,
      const size_t & epochs,
      bool use_softmax = false)
  {
    YANN_CHECK(is_same_size(actual0, expected));

    // setup
    Matrix delta, actual;
    delta.resizeLike(expected);
    actual = actual0;
    Value cost = 0;

    // ensure we don't do allocations in eigen
    {
      BlockAllocations block;

      // check activation_function
      cost = cost_function->f(actual0, expected);
      BOOST_CHECK_CLOSE(cost, cost0, TEST_TOLERANCE);

      // check activation_derivative
      size_t progress_step = max(epochs / 10, min_steps);
      for (size_t ii = 0; ii < epochs; ++ii) {
        cost_function->derivative(actual, expected, delta);
        actual -= learning_rate * delta;
        if(use_softmax) {
          softmax(actual, actual);
        }
        cost = cost_function->f(actual, expected);

        if (ii % progress_step == 0) {
          BOOST_TEST_MESSAGE("epoch=" << ii << " out of " << epochs << " cost=" << cost);
        }
      }
    }

    return make_pair(actual, cost);
  }

  pair<Vector, Vector> test_activation_function(
      const unique_ptr<ActivationFunction> & activation_function,
      const unique_ptr<CostFunction> & cost_function,
      const Vector & input0, const Vector & output0, const Vector & expected,
      double learning_rate = 1.0, const size_t & epochs = 10,
      int alloc_check_flags = BlockAllocations::None)
  {
    YANN_CHECK_EQ(input0.size(), output0.size());
    YANN_CHECK_EQ(input0.size(), expected.size());

    // setup
    const size_t size = input0.size();
    Vector output(size);
    Vector delta(size);
    Vector input(input0), gradient(size), cost_derivative(size);

    // ensure we don't do allocations in eigen
    {
      BlockAllocations block(alloc_check_flags);

      // check activation_function
      activation_function->f(input0, output);
      BOOST_CHECK(output0.isApprox(output, TEST_TOLERANCE));

      // check activation_derivative
      size_t progress_step = max(epochs / 10, min_steps);
      for (size_t ii = 0; ii < epochs; ++ii) {
        // calculate gradient
        activation_function->derivative(output, delta);
        cost_function->derivative(output, expected, cost_derivative);
        gradient.array() = cost_derivative.array() * delta.array();

        input -= learning_rate * gradient;

        // next iteration
        activation_function->f(input, output);
        if (ii % progress_step == 0) {
          BOOST_TEST_MESSAGE("epoch=" << ii << " out of " << epochs << " cost=" << cost_function->f(output, expected));
        }
      }
    }

    return make_pair(input, output);
  }

};
// struct FunctionsTestFixture

BOOST_FIXTURE_TEST_SUITE(FunctionsTest, FunctionsTestFixture);

////////////////////////////////////////////////////////////////////////////////////////////////
//
// cost functions tests
//
BOOST_AUTO_TEST_CASE(QuadraticCost_Test)
{
  const size_t size = 2;
  Vector actual0(size);
  Vector expected(size);
  Value cost0;
  const double learning_rate = 0.25;
  size_t epochs = 50;

  actual0 << 11.0, 1;
  expected << 1.0, 6.0;
  cost0 = 125;

  pair<Vector, Value> res = test_cost_function(
      make_unique<QuadraticCost>(),
      actual0, expected, cost0,
      learning_rate, epochs);

  BOOST_TEST_MESSAGE("expected=" << expected << " actual=" << res.first << " cost=" << res.second);
  BOOST_CHECK(expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK_SMALL(res.second, TEST_TOLERANCE);
}

BOOST_AUTO_TEST_CASE(ExponentialCost_Test)
{
  const size_t size = 2;
  Vector actual0(size);
  Vector expected(size);
  Value cost0;
  const double learning_rate = 0.001;
  size_t epochs = 100;

  actual0 << 11.0, 1;
  expected << 1.0, 6.0;
  cost0 = 349.03430;

  pair<Vector, Value> res = test_cost_function(
      make_unique<ExponentialCost>(100.0),
      actual0, expected, cost0,
      learning_rate, epochs);

  BOOST_TEST_MESSAGE("expected=" << expected << " actual=" << res.first << " cost=" << res.second);
  BOOST_CHECK(expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK_CLOSE(100.00, res.second, TEST_TOLERANCE);
}

BOOST_AUTO_TEST_CASE(CrossEntrypyCost_Test)
{
  const size_t size = 2;
  Vector actual0(size);
  Vector expected(size);
  Value cost0;
  const double learning_rate = 0.1;
  size_t epochs = 15;

  // f(actual, expected) = sum(-(expected * ln(actual) + (1 - expected) * ln(1 - actual)))
  actual0 << 0.9, 0.3;
  expected << 0.3, 0.8;
  cost0 = 2.677931;

  pair<Vector, Value> res = test_cost_function(
      make_unique<CrossEntropyCost>(),
      actual0, expected, cost0,
      learning_rate, epochs);

  BOOST_TEST_MESSAGE("expected=" << expected << " actual=" << res.first << " cost=" << res.second);
  BOOST_CHECK(expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK_CLOSE(1.111266, res.second, TEST_TOLERANCE);
}

BOOST_AUTO_TEST_CASE(HellingerDistanceCost_Test)
{
  const size_t size = 2;
  Vector actual0(size);
  Vector expected(size);
  Value cost0;
  const double learning_rate = 0.75;
  size_t epochs = 15;

  actual0 << 0.9, 0.1;
  expected << 0.3, 0.7;
  cost0 = 0.43150;
  pair<Vector, Value> res = test_cost_function(
      make_unique<HellingerDistanceCost>(0.0001),
      actual0, expected, cost0,
      learning_rate, epochs);

  BOOST_TEST_MESSAGE("expected=" << expected << " actual=" << res.first << " cost=" << res.second);
  BOOST_CHECK(expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK_SMALL(res.second, TEST_TOLERANCE);
}

BOOST_AUTO_TEST_CASE(SquaredHingeLoss_Test)
{
  const size_t size = 4;
  Vector actual0(size);
  Vector actual_expected(size);
  Vector expected(size);
  Value cost0;
  const double learning_rate = 0.5;
  size_t epochs = 10;

  actual0   << 0.6, 0.1, 0.3, 0.0;
  expected  << 0.0, 1.0, 0.0, 0.0;
  actual_expected << 0.049, 0.852, 0.049, 0.049;
  cost0  = 0.81;
  pair<Vector, Value> res = test_cost_function(
      make_unique<SquaredHingeLoss>(),
      actual0, expected, cost0,
      learning_rate, epochs, true); // use softmax

  BOOST_TEST_MESSAGE("expected=" << expected << " actual=" << res.first << " cost=" << res.second);
  BOOST_CHECK(actual_expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK_CLOSE(0.02177566296, res.second, TEST_TOLERANCE);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// activations functions tests
//
BOOST_AUTO_TEST_CASE(IdentityFunction_Test)
{
  const size_t size = 2;
  Vector input0(size);
  Vector output0(size);
  Vector input_expected(size);
  Vector expected(size);
  const double learning_rate = 0.75;
  size_t epochs = 1000;

  input0   << -5,  10;
  output0  << -5, 10;
  expected << 20, -5;
  input_expected << 20, -5;
  pair<Vector, Vector> res = test_activation_function(
      make_unique<IdentityFunction>(),
      make_unique<QuadraticCost>(),
      input0, output0, expected,
      learning_rate, epochs);

  BOOST_TEST_MESSAGE("intput expected=" << input_expected << " actual=" << res.first);
  BOOST_TEST_MESSAGE("output expected=" << expected << " actual=" << res.second);
  BOOST_CHECK(input_expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK(expected.isApprox(res.second, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(SigmoidFunction_Test)
{
  const size_t size = 2;
  Vector input0(size);
  Vector output0(size);
  Vector input_expected(size);
  Vector expected(size);
  const double learning_rate = 0.75;
  size_t epochs = 1000;

  input0 << 10, 10;
  output0 << 1.0, 1.0;
  expected << 0.8, 0.1;
  input_expected << 1.3862943611, -2.1972245773; // x = -ln(1/y - 1)
  pair<Vector, Vector> res = test_activation_function(
      make_unique<SigmoidFunction>(),
      make_unique<QuadraticCost>(),
      input0, output0, expected,
      learning_rate, epochs);

  BOOST_TEST_MESSAGE("intput expected=" << input_expected << " actual=" << res.first);
  BOOST_TEST_MESSAGE("output expected=" << expected << " actual=" << res.second);
  BOOST_CHECK(input_expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK(expected.isApprox(res.second, TEST_TOLERANCE));
}


BOOST_AUTO_TEST_CASE(FastSigmoidFunction_Test)
{
  const size_t size = 2;
  Vector input0(size);
  Vector output0(size);
  Vector input_expected(size);
  Vector expected(size);
  const double learning_rate = 0.75;
  size_t epochs = 1000;

  input0 << 10, 10;
  output0 << 1.0, 1.0;
  expected << 0.8, 0.1;
  input_expected << 1.3862943611, -2.1972245773; // x = -ln(1/y - 1)
  pair<Vector, Vector> res = test_activation_function(
      make_unique<FastSigmoidFunction>(10000), // increase internal table size to improve accuracy
      make_unique<QuadraticCost>(),
      input0, output0, expected,
      learning_rate, epochs);

  BOOST_TEST_MESSAGE("intput expected=" << input_expected << " actual=" << res.first);
  BOOST_TEST_MESSAGE("output expected=" << expected << " actual=" << res.second);
  BOOST_CHECK(input_expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK(expected.isApprox(res.second, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(ReluFunction_Test)
{
  const size_t size = 2;
  Vector input0(size);
  Vector output0(size);
  Vector input_expected(size);
  Vector expected(size);
  const double learning_rate = 0.75;
  size_t epochs = 1000;

  input0   << -5,  10;
  output0  << -0.5, 10;
  expected << 20, -5;
  input_expected << 20, -50;
  pair<Vector, Vector> res = test_activation_function(
      make_unique<ReluFunction>(0.1),
      make_unique<QuadraticCost>(),
      input0, output0, expected,
      learning_rate, epochs);

  BOOST_TEST_MESSAGE("intput expected=" << input_expected << " actual=" << res.first);
  BOOST_TEST_MESSAGE("output expected=" << expected << " actual=" << res.second);
  BOOST_CHECK(input_expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK(expected.isApprox(res.second, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(TahhFunction_Test)
{
  const size_t size = 2;
  Vector input0(size);
  Vector output0(size);
  Vector input_expected(size);
  Vector expected(size);
  const double learning_rate = 0.75;
  size_t epochs = 1000;

  input0   << 1,  10;
  output0  << 0.99992, 1.71589;
  expected << 0.1, -0.5;
  input_expected << 0.08753, -0.45018;
  pair<Vector, Vector> res = test_activation_function(
      make_unique<TanhFunction>(1.7159, 0.6666),
      make_unique<QuadraticCost>(),
      input0, output0, expected,
      learning_rate, epochs);

  BOOST_TEST_MESSAGE("intput expected=" << input_expected << " actual=" << res.first);
  BOOST_TEST_MESSAGE("output expected=" << expected << " actual=" << res.second);
  BOOST_CHECK(input_expected.isApprox(res.first, TEST_TOLERANCE));
  BOOST_CHECK(expected.isApprox(res.second, TEST_TOLERANCE));
}


BOOST_AUTO_TEST_SUITE_END()

