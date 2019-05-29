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

#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>

#include "types.h"
#include "random.h"
#include "utils.h"
#include "timer.h"
#include "nn.h"
#include "test_utils.h"
#include "functions.h"

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

  pair<Matrix, Value> test_cost_function(
      const unique_ptr<CostFunction> & cost_function,
      const RefConstMatrix & actual0, const RefConstMatrix & expected, const Value & cost0,
      double learning_rate = 1.0, const size_t & epochs = 10)
  {
    BOOST_VERIFY(is_same_size(actual0, expected));

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
    BOOST_VERIFY(input0.size() == output0.size());
    BOOST_VERIFY(input0.size() == expected.size());

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
// perf functions tests
//
BOOST_AUTO_TEST_CASE(Perf_Test, * disabled())
{
  const size_t size = 1000;
  const size_t data_size = size * size;

  // Eigen::Matrix
  {
      Timer timer("Eigen::Matrix total");
      Matrix aa(size, size), bb(size, size), cc(size, size);
      {
        Timer timer("Eigen::Matrix random generation");
        unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
        gen->generate(aa);
        gen->generate(bb);
      }
      // ensure we don't do allocations in eigen
      {
        BlockAllocations block;
        {
          Timer timer("Eigen::Matrix; cc.noalias() = aa.lazyProduct(bb);");
          cc.noalias() = aa.lazyProduct(bb);
        }
      }
      {
        Timer timer("Eigen::Matrix; cc.noalias() = aa * bb;");
        cc.noalias() = aa * bb;
      }
  }

  // UBLAS::matrix
  {
      Timer timer("UBLAS matrixes total");
      ublas::matrix<Value> aa(size, size), bb(size, size), cc(size, size);
      {
        Timer timer("UBLAS matrixes random generation");
        unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
        gen->generate(aa.data().begin(), aa.data().end());
        gen->generate(bb.data().begin(), bb.data().end());
      }
      {
        Timer timer("UBLAS matrixes noalias(cc) = ublas::prod(aa, bb);");
        noalias(cc) = ublas::prod(aa, bb);
      }
      {
        Timer timer("UBLAS matrixes ublas::axpy_prod(aa, bb, cc, true);");
        ublas::axpy_prod(aa, bb, cc, true);
      }
  }

  // std::vector
  {
      Timer timer("Value[] matrixes total");
      auto aa = make_unique<Value[]>(data_size);
      auto bb = make_unique<Value[]>(data_size);
      auto cc = make_unique<Value[]>(data_size);

      {
        Timer timer("Value[] matrixes random generation");
        unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
        gen->generate(aa.get(), aa.get() + data_size);
        gen->generate(bb.get(), bb.get() + data_size);
      }
      {
        Timer timer("Value[] matrixes ");
        auto aa_ptr = aa.get();
        auto bb_ptr = bb.get();
        auto cc_ptr = cc.get();
        memset(cc_ptr, 0, data_size);

        for(size_t ii = 0; ii < size; ++ii) {
          for(size_t jj = 0; jj < size; ++jj) {
            for(size_t kk = 0; kk < size; ++kk) {
              cc_ptr[ii*size + jj] += aa_ptr[ii*size + kk] * bb_ptr[kk*size + jj];
            }
          }
        }
      }
      {
        Timer timer("Value[] matrixes2 ");
        auto aa_ptr = aa.get();
        auto bb_ptr = bb.get();
        auto cc_ptr = cc.get();
        memset(cc_ptr, 0, data_size);

        for(size_t ii = 0; ii < size; ++ii) {
          for(size_t kk = 0; kk < size; ++kk) {
            for(size_t jj = 0; jj < size; ++jj) {
              cc_ptr[ii*size + jj] += aa_ptr[ii*size + kk] * bb_ptr[kk*size + jj];
            }
          }
        }
      }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// cost functions tests
//
BOOST_AUTO_TEST_CASE(QuadraticCost_Test)
{
  const size_t size = 2;
  Vector actual0(size);
  Vector expected(size);
  Vector actual_expected(size);
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

BOOST_AUTO_TEST_CASE(CrossEntrypyCost_Test)
{
  const size_t size = 2;
  Vector actual0(size);
  Vector expected(size);
  Vector actual_expected(size);
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
  Vector actual_expected(size);
  Value cost0;
  const double learning_rate = 0.75;
  size_t epochs = 15;

  //  f(actual, expected) = sum(sqrt(actual) - sqrt(expected))^2
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

