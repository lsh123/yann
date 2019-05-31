//
// Add --log_level=message to see the messages!
//
#define BOOST_TEST_MODULE "Layers Tests"

#include <boost/test/unit_test.hpp>

#include "utils.h"

#include "timer.h"
#include "test_utils.h"
#include "test_layers.h"

using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace yann;
using namespace yann::test;


////////////////////////////////////////////////////////////////////////////////////////////////
//
// AvgLayer implementation
//
size_t yann::test::AvgLayer::g_counter = 0;

yann::test::AvgLayer::AvgLayer(MatrixSize input_size) :
  _input_size(input_size),
  _value(++g_counter)
{
}
yann::test::AvgLayer:: ~AvgLayer()
{
}

// Layer overwrites
string yann::test::AvgLayer::get_name() const
{
  return "AvgLayer" ;
}
bool yann::test::AvgLayer::is_equal(const Layer& other, double tolerance) const
{
  if(!Base::is_equal(other, tolerance)) {
      return false;
    }
    auto the_other = dynamic_cast<const AvgLayer*>(&other);
    if(the_other == nullptr) {
      return false;
    }
    if(_value != the_other->_value) {
      return false;
    }
    return true;
}

MatrixSize yann::test::AvgLayer::get_input_size() const
{
  return _input_size;
}
MatrixSize yann::test::AvgLayer::get_output_size() const
{
  return _output_size;
}

unique_ptr<Layer::Context> yann::test::AvgLayer::create_context(const MatrixSize & batch_size ) const
{
  return make_unique<Layer::Context>(_output_size, batch_size);
}
unique_ptr<Layer::Context> yann::test::AvgLayer::create_context(const RefVectorBatch & output) const
{
  return make_unique<Context>(output);
}
unique_ptr<Layer::Context> yann::test::AvgLayer::create_training_context(
    const MatrixSize & batch_size,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  return make_unique<Context>(_output_size, batch_size);
}
unique_ptr<Layer::Context> yann::test::AvgLayer::create_training_context(
    const RefVectorBatch & output,
    const std::unique_ptr<Layer::Updater> & updater) const
{
  return make_unique<Context>(output);
}

void yann::test::AvgLayer::feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode) const
{
  YANN_CHECK(context);
  YANN_CHECK_EQ(get_batch_item_size(input), _input_size);
  YANN_CHECK_EQ(get_batch_item_size(context->get_output()), _output_size);

  RefVectorBatch output = context->get_output();
  switch(mode) {
  case Operation_Assign:
    output.setZero();
    break;
  case Operation_PlusEqual:
    // do nothing
    break;
  }
  const auto batch_size = get_batch_size(input);
  for (MatrixSize ii = 0; ii < batch_size; ++ii) {
    get_batch(output, ii)(0) += (get_batch(input, ii).sum() / ((double)_input_size));
  }
}

void yann::test::AvgLayer::backprop(const RefConstVectorBatch & gradient_output, const RefConstVectorBatch & input,
                      boost::optional<RefVectorBatch> gradient_input, Context * /* context */) const
{
  if(gradient_input) {
    const auto batch_size = get_batch_size(input);
    for (MatrixSize ii = 0; ii < batch_size; ++ii) {
      get_batch(*gradient_input, ii).array() = get_batch(gradient_output, ii)(0);
    }
  }
}

void yann::test::AvgLayer::init(enum InitMode mode)
{
  // do nothing
}
void yann::test::AvgLayer::update(Context * context, const size_t & batch_size)
{
  // do nothing
}

void yann::test::AvgLayer::read(istream & is)
{
  Base::read(is);

  char ch;
  if(is >> ch && ch != '(') {
    is.putback(ch);
    is.setstate(ios_base::failbit);
    return;
  }
  if(is >> _value >> ch && ch != ')') {
    is.putback(ch);
    is.setstate(ios_base::failbit);
    return;
  }
}
void yann::test::AvgLayer::write(ostream & os) const
{
  Base::write(os);
  os << "(" << _value << ")";
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Test
//
struct LayersTestFixture
{
  LayersTestFixture()
  {

  }
  ~LayersTestFixture()
  {

  }
};
// struct LayersTestFixture

BOOST_FIXTURE_TEST_SUITE(LayersTest, LayersTestFixture);

BOOST_AUTO_TEST_CASE(Layer_IO_Test)
{
  BOOST_TEST_MESSAGE("*** Layer IO test ...");

  const MatrixSize input_size = 2;
  unique_ptr<AvgLayer> one(new AvgLayer(input_size));
  one->init(InitMode_Random_01);

  BOOST_TEST_MESSAGE("AvgLayer before writing to file: " << "\n" << *one);
  ostringstream oss;
  oss << (*one);
  BOOST_CHECK(!oss.fail());

  unique_ptr<AvgLayer> two(new AvgLayer(input_size));
  std::istringstream iss(oss.str());
  iss >> (*two);
  BOOST_CHECK(!iss.fail());
  BOOST_TEST_MESSAGE("AvgLayer after loading from file: " << "\n" << *two);

  BOOST_CHECK(one->is_equal(*two, TEST_TOLERANCE));
}

BOOST_AUTO_TEST_SUITE_END()

