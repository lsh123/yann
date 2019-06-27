//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "Layer Updaters Tests"
#include <boost/test/unit_test.hpp>

#include <sstream>

#include "core/types.h"
#include "core/utils.h"
#include "core/updaters.h"

#include "test_utils.h"
#include "timer.h"
#include "mnist_test.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;

struct UpdatersTestFixture
{
  UpdatersTestFixture()
  {
  }
  ~UpdatersTestFixture()
  {
  }
}; // struct UpdatersTestFixture

BOOST_FIXTURE_TEST_SUITE(UpdatersTest, UpdatersTestFixture);

BOOST_AUTO_TEST_CASE(Test_GradientDescent_Matrix)
{
  BOOST_TEST_MESSAGE("*** Updater_GradientDescent Matrix test ...");

  const MatrixSize size = 2;

  auto updater = make_unique<Updater_GradientDescent>(0.9, 0.1);
  BOOST_CHECK(updater);
  updater->init(size, size);
  updater->start_epoch();

  Matrix ww(size, size);
  Matrix ww_expected(size, size);
  Matrix delta(size, size);

  // test decay factor
  ww << 1, 2, 3, 4;
  delta << 0.0, 0.0, 0.0, 0.0;
  ww_expected << 0.9, 1.8, 2.7, 3.6;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));

  // test learning + decay factor
  ww << 1, 2, 3, 4;
  delta << 0.1, 0.2, 0.3, 0.4;
  ww_expected << 0.81, 1.62, 2.43, 3.24;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Test_GradientDescent_Value)
{
  BOOST_TEST_MESSAGE("*** Updater_GradientDescent value test ...");

  auto updater = make_unique<Updater_GradientDescent>(0.9, 0.1);
  BOOST_CHECK(updater);
  updater->init(1, 1);
  updater->start_epoch();

  Value ww;
  Value ww_expected;
  Value delta;

  // test decay factor
  ww = 1;
  delta = 0.0;
  ww_expected = 0.9;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);

  // test learning + decay factor
  ww = 1;
  delta = 0.1;
  ww_expected = 0.81;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);
}

BOOST_AUTO_TEST_CASE(Test_GradientDescentWithMomentum_Matrix)
{
  BOOST_TEST_MESSAGE("*** Updater_GradientDescentWithMomentum matrix test ...");

  const MatrixSize size = 2;

  auto updater = make_unique<Updater_GradientDescentWithMomentum>(0.7, 0.9);
  BOOST_CHECK(updater);
  updater->init(size, size);
  updater->start_epoch();

  Matrix ww(size, size);
  Matrix ww_expected(size, size);
  Matrix delta(size, size);

  // first iteration
  // v(0) = 0,0,0,0
  ww << 1, 2, 3, 4;
  delta << 0.1, 0.2, 0.3, 0.4;
  // v(1) = 0.01, 0.02, 0.03, 0.04
  ww_expected << 0.993, 1.986, 2.979, 3.972;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));

  // second iteration
  delta << 0.4, 0.3, 0.2, 0.1;
  // v(2) = 0.049, 0.048, 0.047, 0.046
  ww_expected << 0.9587, 1.9524, 2.9461, 3.9398;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));
}


BOOST_AUTO_TEST_CASE(Test_GradientDescentWithMomentum_Value)
{
  BOOST_TEST_MESSAGE("*** Updater_GradientDescentWithMomentum value test ...");

  auto updater = make_unique<Updater_GradientDescentWithMomentum>(0.7, 0.9);
  BOOST_CHECK(updater);
  updater->init(1, 1);
  updater->start_epoch();

  Value ww;
  Value ww_expected;
  Value delta;

  // first iteration
  // v(0) = 0
  ww = 1;
  delta = 0.1;
  // v(1) = 0.01
  ww_expected = 0.993;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);

  // second iteration
  delta = 0.4;
  // v(2) = 0.049
  ww_expected = 0.9587;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);
}

BOOST_AUTO_TEST_CASE(Test_AdaGrad_Matrix)
{
  BOOST_TEST_MESSAGE("*** Updater_AdaGrad matrix test ...");

  const MatrixSize size = 2;

  auto updater = make_unique<Updater_AdaGrad>(0.01, 1.0e-07);
  BOOST_CHECK(updater);
  updater->init(size, size);
  updater->start_epoch();

  Matrix ww(size, size);
  Matrix ww_expected(size, size);
  Matrix delta(size, size);

  // first iteration
  // s(0) = 0,0,0,0
  ww << 1, 2, 3, 4;
  delta << 0.1, 0.2, 0.3, 0.4;
  // s(1) = 0.01, 0.04, 0.09, 0.16
  ww_expected << 0.99, 1.99, 2.99, 3.99;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));

  // second iteration
  delta << 0.4, 0.3, 0.2, 0.1;
  // s(2) = 0.17, 0.13, 0.13, 0.17
  ww_expected << 0.9803, 1.9817, 2.9845, 3.9876;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Test_AdaGrad_Value)
{
  BOOST_TEST_MESSAGE("*** Updater_AdaGrad value test ...");

  auto updater = make_unique<Updater_AdaGrad>(0.01, 1.0e-07);
  BOOST_CHECK(updater);
  updater->init(1, 1);
  updater->start_epoch();

  Value ww;
  Value ww_expected;
  Value delta;

  // first iteration
  // s(0) = 0
  ww = 1;
  delta = 0.1;
  // s(1) = 0.01
  ww_expected = 0.99;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);

  // second iteration
  delta = 0.4;
  // s(2) = 0.17
  ww_expected = 0.9803;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);
}


BOOST_AUTO_TEST_CASE(Test_RMSprop_Matrix)
{
  BOOST_TEST_MESSAGE("*** Updater_RMSprop matrix test ...");

  const MatrixSize size = 2;

  auto updater = make_unique<Updater_RMSprop>(0.01, 0.9, 1.0e-07);
  BOOST_CHECK(updater);
  updater->init(size, size);
  updater->start_epoch();

  Matrix ww(size, size);
  Matrix ww_expected(size, size);
  Matrix delta(size, size);

  // first iteration
  ww << 1, 2, 3, 4;
  delta << 0.1, 0.2, 0.3, 0.4;
  ww_expected << 0.9683788044, 1.968377619, 2.968377399, 3.968377322;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));

  // second iteration
  delta << 0.4, 0.3, 0.2, 0.1;
  ww_expected << 0.9376096647, 1.941651601, 2.950195656, 3.960319119;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Test_RMSprop_Value)
{
  BOOST_TEST_MESSAGE("*** Updater_RMSprop value test ...");

  auto updater = make_unique<Updater_RMSprop>(0.01, 0.9, 1.0e-07);
  BOOST_CHECK(updater);
  updater->init(1, 1);
  updater->start_epoch();

  Value ww;
  Value ww_expected;
  Value delta;

  // first iteration
  // s(0) = 0
  ww = 1;
  delta = 0.1;
  // s(1) = 0.001
  ww_expected = 0.9683788044;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);

  // second iteration
  delta = 0.4;
  // s(2) = 0.0169
  ww_expected = 0.9376096647;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);
}

BOOST_AUTO_TEST_CASE(Test_AdaDelta_Matrix)
{
  BOOST_TEST_MESSAGE("*** Updater_AdaDelta matrix test ...");

  const MatrixSize size = 2;

  auto updater = make_unique<Updater_AdaDelta>(0.95, 1.0e-07);
  BOOST_CHECK(updater);
  updater->init(size, size);
  updater->start_epoch();

  Matrix ww(size, size);
  Matrix ww_expected(size, size);
  Matrix delta(size, size);

  // first iteration
  // s(0) = 0, 0, 0, 0
  // d(0) = 0, 0, 0, 0
  ww << 1, 2, 3, 4;
  delta << 1, 2, -1, -2;
  // s(1) = 0.05, 0.2, 0.05, 0.2
  // d(1) = 0, 0, 0, 0
  ww_expected << 0.9986, 1.9986, 3.0014, 4.0014;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));

  // second iteration
  delta << 1, 2, -1, -2;
  // s(2) = 0.0975, 0.39, 0.0975, 0.39
  // d(2) = 0.0000000999998, 0.00000009999995, 0.0000000999998, 0.00000009999995
  ww_expected << 0.9971535596, 1.997153557, 3.00284644, 4.002846443;
  updater->update(delta, 1, ww);
  BOOST_CHECK(ww_expected.isApprox(ww, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Test_AdaDelta_Value)
{
  BOOST_TEST_MESSAGE("*** Updater_AdaDelta value test ...");

  auto updater = make_unique<Updater_AdaDelta>(0.95, 1.0e-07);
  BOOST_CHECK(updater);
  updater->init(1, 1);
  updater->start_epoch();

  Value ww;
  Value ww_expected;
  Value delta;

  // first iteration
  // s(0) = 0
  // d(0) = 0
  ww = 1;
  delta = 1;
  // s(1) = 0.05
  // d(1) = 0
  ww_expected = 0.9985857879;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);

  // second iteration
  delta = 1;
  // s(2) = 0.0975
  // d(2) = 0.0000000999998
  ww_expected = 0.9971535596;
  updater->update(delta, 1, ww);
  BOOST_CHECK_CLOSE(ww_expected, ww, TEST_TOLERANCE);
}

BOOST_AUTO_TEST_SUITE_END()
