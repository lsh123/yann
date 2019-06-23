/*
 * updaters.cpp
 *
 */
#include <string>

#include "core/utils.h"
#include "core/layer.h"
#include "core/updaters.h"

using namespace std;
using namespace yann;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// RateProvider_Annealing implementation
//
yann::RateProvider_Annealing::RateProvider_Annealing(
    const double & rate,
    const size_t & search_epochs) :
    _rate(rate),
    _search_epochs(search_epochs),
    _cur_epoch(0)
{
}

std::string yann::RateProvider_Annealing::get_info() const
{
  ostringstream oss;
  oss << "RateProvider_Annealing["
      << "rate=" << _rate
      << ", search_epochs=" << _search_epochs
      << "]";
  return oss.str();
}

void yann::RateProvider_Annealing::start_epoch()
{
  ++_cur_epoch;
}

double yann::RateProvider_Annealing::get_current_rate() const
{
  return (_search_epochs > 0)
      ? _rate / (1 + _cur_epoch / (double)_search_epochs)
      : _rate;
}

double yann::RateProvider_Annealing::get(const size_t & batch_size)
{
  return batch_size > 0 ? (get_current_rate() / batch_size) : get_current_rate();
}

std::unique_ptr<RateProvider> yann::RateProvider_Annealing::copy() const
{
  return std::make_unique<RateProvider_Annealing>(_rate, _search_epochs);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Updater_GradientDescent implementation
//
yann::Updater_GradientDescent::Updater_GradientDescent(
    const double & learning_rate,
    const double & regularization_rate,
    const size_t & search_epochs)
{
  _learning_rate = make_unique<RateProvider_Annealing>(learning_rate, search_epochs);
  _regularization_rate = make_unique<RateProvider_Annealing>(regularization_rate, search_epochs);

  YANN_CHECK(_learning_rate);
  YANN_CHECK(_regularization_rate);
}

yann::Updater_GradientDescent::Updater_GradientDescent(
    const std::unique_ptr<RateProvider> & learning_rate,
    const std::unique_ptr<RateProvider> & regularization_rate) :
    _learning_rate(learning_rate->copy()),
    _regularization_rate(regularization_rate->copy())
{
    YANN_CHECK(_learning_rate);
    YANN_CHECK(_regularization_rate);
}

std::string yann::Updater_GradientDescent::get_info() const
{
  YANN_CHECK(_learning_rate);
  YANN_CHECK(_regularization_rate);

  ostringstream oss;
  oss << "GradientDescent["
      << "learning_rate=" << _learning_rate->get_info()
      << ", regularization_rate=" << _regularization_rate->get_info()
      << "]";
  return oss.str();
}

std::unique_ptr<Layer::Updater> yann::Updater_GradientDescent::copy() const
{
  YANN_CHECK(_learning_rate);
  YANN_CHECK(_regularization_rate);
  return make_unique<Updater_GradientDescent>(_learning_rate, _regularization_rate);
}

void yann::Updater_GradientDescent::init(const MatrixSize & rows, const MatrixSize & cols)
{
  // do nothing
}

void yann::Updater_GradientDescent::start_epoch()
{
  YANN_CHECK(_learning_rate);
  YANN_CHECK(_regularization_rate);
  _learning_rate->start_epoch();
  _regularization_rate->start_epoch();
}

void yann::Updater_GradientDescent::reset()
{
  // do nothing
}

void yann::Updater_GradientDescent::update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value)
{
  YANN_SLOW_CHECK(is_same_size(delta, value));
  YANN_SLOW_CHECK_GT(batch_size, 0);
  YANN_SLOW_CHECK(_learning_rate);
  YANN_SLOW_CHECK(_regularization_rate);

  auto learning_factor = _learning_rate->get(batch_size);
  auto decay_factor = 1 - _regularization_rate->get(batch_size);
  YANN_SLOW_CHECK_GE(learning_factor, 0.0);
  YANN_SLOW_CHECK_GE(decay_factor, 0.0);

  value = decay_factor * value - learning_factor * delta;
}

void yann::Updater_GradientDescent::update(const Value & delta, const size_t & batch_size, Value & value)
{
  YANN_SLOW_CHECK_GT(batch_size, 0);
  YANN_SLOW_CHECK(_learning_rate);
  YANN_SLOW_CHECK(_regularization_rate);

  auto learning_factor = _learning_rate->get(batch_size);
  auto decay_factor = 1 - _regularization_rate->get(batch_size);
  YANN_SLOW_CHECK_GE(learning_factor, 0.0);
  YANN_SLOW_CHECK_GE(decay_factor, 0.0);

  value = decay_factor * value - learning_factor * delta;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Updater_GradientDescentWithMomentum implementation
//
yann::Updater_GradientDescentWithMomentum::Updater_GradientDescentWithMomentum(
    const double & alpha,
    const double & beta,
    const size_t & search_epochs) :
    _alpha(make_unique<RateProvider_Annealing>(alpha, search_epochs)),
    _beta(beta)
{
  YANN_CHECK(_alpha);
}

yann::Updater_GradientDescentWithMomentum::Updater_GradientDescentWithMomentum(
    const std::unique_ptr<RateProvider> & alpha,
    const double  & beta) :
    _alpha(alpha->copy()),
    _beta(beta)
{
    YANN_CHECK(_alpha);
}

std::string yann::Updater_GradientDescentWithMomentum::get_info() const
{
  YANN_CHECK(_alpha);
  ostringstream oss;
  oss << "GradientDescentWithMomentum["
      << "alpha=" << _alpha->get_info()
      << ", beta=" << _beta
      << "]";
  return oss.str();
}

std::unique_ptr<Layer::Updater> yann::Updater_GradientDescentWithMomentum::copy() const
{
  YANN_CHECK(_alpha);
  return make_unique<Updater_GradientDescentWithMomentum>(_alpha, _beta);
}

void yann::Updater_GradientDescentWithMomentum::init(const MatrixSize & rows, const MatrixSize & cols)
{
  _velocity.resize(rows, cols);
  _velocity.setZero();
}

void yann::Updater_GradientDescentWithMomentum::start_epoch()
{
  YANN_CHECK(_alpha);

  _alpha->start_epoch();

  _velocity.setZero();
}

void yann::Updater_GradientDescentWithMomentum::reset()
{
  // do nothing
}

// ww(t+1) = w(t) - alpha * vv(t)
// vv(t) = beta * vv(t-1) + (1-beta) * delta_ww
void yann::Updater_GradientDescentWithMomentum::update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value)
{
  YANN_CHECK(is_same_size(delta, value));
  YANN_CHECK(is_same_size(_velocity, value));
  YANN_SLOW_CHECK_GT(batch_size, 0);
  YANN_SLOW_CHECK(_alpha);

  auto alpha = _alpha->get(batch_size);

  _velocity.array() = _beta * _velocity.array() + (1 - _beta) * delta.array();
  value.array() -= alpha * _velocity.array();
}

void yann::Updater_GradientDescentWithMomentum::update(const Value & delta, const size_t & batch_size, Value & value)
{
  YANN_CHECK_EQ(_velocity.size(), 1);
  YANN_SLOW_CHECK_GT(batch_size, 0);
  YANN_SLOW_CHECK(_alpha);

  auto alpha = _alpha->get(batch_size);
  _velocity(0,0) = _beta * _velocity(0,0) + (1 - _beta) * delta;
  value -= alpha * _velocity(0,0);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Updater_AdaGrad implementation
//
yann::Updater_AdaGrad::Updater_AdaGrad(const double & rate, const double & epsilon) :
    _rate(rate),
    _epsilon(epsilon)
{
  YANN_CHECK_GT(_rate, 0);
  YANN_CHECK_GT(_epsilon, 0);
}

std::string yann::Updater_AdaGrad::get_info() const
{
  ostringstream oss;
  oss << "Updater_AdaGrad["
      << "rate=" << _rate
      << ", epsilon=" << _epsilon
      << "]";
  return oss.str();
}

std::unique_ptr<Layer::Updater> yann::Updater_AdaGrad::copy() const
{
  return make_unique<Updater_AdaGrad>(_rate, _epsilon);
}

void yann::Updater_AdaGrad::init(const MatrixSize & rows, const MatrixSize & cols)
{
  _ss.resize(rows, cols);
  _ss.setZero();
}

void yann::Updater_AdaGrad::start_epoch()
{
  // do nothing
}

void yann::Updater_AdaGrad::reset()
{
  // do nothing
}

// S(t) = S(t-1) + delta^2
// ww(t+1) = w(t) - rate * elem_prod(delta * 1/sqrt(S(t) + epsilon))
void yann::Updater_AdaGrad::update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value)
{
  YANN_CHECK(is_same_size(delta, value));
  YANN_CHECK(is_same_size(_ss, value));
  YANN_SLOW_CHECK_GT(batch_size, 0);

  _ss.array() += delta.array().square();
  value.array() -= _rate * (_ss.array() + _epsilon).rsqrt() * delta.array();
}

void yann::Updater_AdaGrad::update(const Value & delta, const size_t & batch_size, Value & value)
{
  YANN_CHECK_EQ(_ss.size(), 1);
  YANN_SLOW_CHECK_GT(batch_size, 0);

  _ss(0,0) += delta * delta;
  value -= _rate * delta / sqrt(_ss(0,0) + _epsilon);
}

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Updater_AdaDelta implementation
//
yann::Updater_AdaDelta::Updater_AdaDelta(const double & beta, const double & epsilon) :
    _beta(beta),
    _epsilon(epsilon)
{
  YANN_CHECK_GT(_beta, 0);
  YANN_CHECK_LT(_beta, 1);
  YANN_CHECK_GT(_epsilon, 0);
}

std::string yann::Updater_AdaDelta::get_info() const
{
  ostringstream oss;
  oss << "Updater_AdaDelta["
      << "_beta=" << _beta
      << ", epsilon=" << _epsilon
      << "]";
  return oss.str();
}

std::unique_ptr<Layer::Updater> yann::Updater_AdaDelta::copy() const
{
  return make_unique<Updater_AdaDelta>(_beta, _epsilon);
}

void yann::Updater_AdaDelta::init(const MatrixSize & rows, const MatrixSize & cols)
{
  _ss.resize(rows, cols);
  _dd.resize(rows, cols);
  _delta.resize(rows, cols);
  _ss.setZero();
  _dd.setZero();
  _delta.setZero();
}

void yann::Updater_AdaDelta::start_epoch()
{
  // do nothing
}

void yann::Updater_AdaDelta::reset()
{
  // do nothing
}

// S(t) = beta * S(t-1) + (1 - beta) * delta^2
// D(t) = beta * D(t-1) + (1 - beta) * delta_ww(t - 1) ^2
// delta_ww(t) = sqrt(D(t) + epsilon) / sqrt(S(t) + epsilon) * delta
// ww(t+1) = w(t) - delta_ww(t)
void yann::Updater_AdaDelta::update(const RefConstMatrix & delta, const size_t & batch_size, RefMatrix value)
{
  YANN_CHECK(is_same_size(delta, value));
  YANN_CHECK(is_same_size(_ss, value));
  YANN_SLOW_CHECK_GT(batch_size, 0);

  _ss.array() = _beta * _ss.array() + (1 - _beta) * delta.array().square();
  _dd.array() = _beta * _dd.array() + (1 - _beta) * _delta.array().square();
  _delta.array() = delta.array() * (_dd.array() + _epsilon).sqrt() / (_ss.array() + _epsilon).sqrt();
  value.array() -= _delta.array();
}

void yann::Updater_AdaDelta::update(const Value & delta, const size_t & batch_size, Value & value)
{
  YANN_CHECK_EQ(_ss.size(), 1);
  YANN_CHECK_EQ(_dd.size(), 1);
  YANN_CHECK_EQ(_delta.size(), 1);
  YANN_SLOW_CHECK_GT(batch_size, 0);

  _ss(0,0) = _beta * _ss(0,0) + (1 - _beta) * delta * delta;
  _dd(0,0) = _beta * _dd(0,0) + (1 - _beta) * _delta(0,0) * _delta(0,0);
  _delta(0,0) = delta * sqrt(_dd(0,0) + _epsilon) / sqrt(_ss(0,0) + _epsilon);
  value -= _delta(0,0);
}
