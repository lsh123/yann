/*
 * updaters.h
 *
 */
#ifndef UPDATERS_H_
#define UPDATERS_H_

#include "core/layer.h"

namespace yann {

// How do we change rate over time
class RateProvider {
public:
  virtual std::string get_info() const = 0;
  virtual void start_epoch() = 0;
  virtual double get(const size_t & tests_num) = 0;
  virtual std::unique_ptr<RateProvider> copy() const = 0;
}; // RateProvider

//
// If search_epochs > 0
//  cur_rate = _rate / (1 + cur_epoch / search_epochs)
// Else (search_epochs)
//  cur_rate = _rate
//
class RateProvider_Annealing : public RateProvider {
public:
  RateProvider_Annealing(const double & rate, const size_t & search_epochs = 0);

  // RateProvider overwrites
  virtual std::string get_info() const;
  virtual void start_epoch();
  virtual double get(const size_t & tests_num);
  virtual std::unique_ptr<RateProvider> copy() const;

private:
  double get_current_rate() const;

private:
  const double _rate;
  const size_t _search_epochs;
  size_t _cur_epoch;
}; // RateProvider_Annealing

// Updates the values according to the gradient descent with weight decay
class Updater_GradientDescent :
    public Layer::Updater
{
public:
  Updater_GradientDescent(
      const double & learning_rate = 1.0,
      const double & regularization_rate = 0.0,
      const size_t & search_epochs = 0);
  Updater_GradientDescent(
      const std::unique_ptr<RateProvider> & learning_rate,
      const std::unique_ptr<RateProvider> & regularization_rate);

  // Layer::Updater overwrites
  virtual std::string get_info() const;
  virtual std::unique_ptr<Layer::Updater> copy() const;

  virtual void init(const MatrixSize & rows, const MatrixSize & cols);
  virtual void start_epoch();
  virtual void reset();
  virtual void update(const RefConstMatrix & delta, const size_t & tests_num, RefMatrix value);
  virtual void update(const Value & delta, const size_t & tests_num, Value & value);

private:
  std::unique_ptr<RateProvider> _learning_rate;
  std::unique_ptr<RateProvider> _regularization_rate;
}; // Updater_GradientDescent

// Updates the values according to the gradient descent with momentum and weight decay:
//
// ww(t+1) = w(t) - alpha * vv(t)
// vv(t) = beta * vv(t-1) + (1-beta) * delta
//
class Updater_GradientDescentWithMomentum :
    public Layer::Updater
{
public:
  Updater_GradientDescentWithMomentum(
      const double & alpha = 1.0,
      const double & beta = 0.9,
      const size_t & search_epochs = 0);
  Updater_GradientDescentWithMomentum(
      const std::unique_ptr<RateProvider> & alpha,
      const double & beta);

  // Layer::Updater overwrites
  virtual std::string get_info() const;
  virtual std::unique_ptr<Layer::Updater> copy() const;

  virtual void init(const MatrixSize & rows, const MatrixSize & cols);
  virtual void start_epoch();
  virtual void reset();
  virtual void update(const RefConstMatrix & delta, const size_t & tests_num, RefMatrix value);
  virtual void update(const Value & delta, const size_t & tests_num, Value & value);

private:
  std::unique_ptr<RateProvider> _alpha;
  const double _beta;
  Matrix _velocity;
}; // Updater_GradientDescentWithMomentum

//
// S(t) = S(t-1) + delta^2
// ww(t+1) = w(t) - rate * elem_prod(delta * 1/sqrt(S(t) + epsilon))
//
class Updater_AdaGrad :
    public Layer::Updater
{
public:
  Updater_AdaGrad(const double & rate = 0.01, const double & epsilon = 1.0e-07);

  // Layer::Updater overwrites
  virtual std::string get_info() const;
  virtual std::unique_ptr<Layer::Updater> copy() const;

  virtual void init(const MatrixSize & rows, const MatrixSize & cols);
  virtual void start_epoch();
  virtual void reset();
  virtual void update(const RefConstMatrix & delta, const size_t & tests_num, RefMatrix value);
  virtual void update(const Value & delta, const size_t & tests_num, Value & value);

private:
  const double _rate;
  const double _epsilon;
  Matrix _ss;
}; // Updater_AdaGrad


//
// S(t) = beta * S(t-1) + (1 - beta) * delta^2
// D(t) = beta * D(t-1) + (1 - beta) * delta_ww(t - 1) ^2
// delta_ww(t) = sqrt(D(t) + epsilon) / sqrt(S(t) + epsilon) * delta
// ww(t+1) = w(t) - delta_ww(t)
class Updater_AdaDelta :
    public Layer::Updater
{
public:
  Updater_AdaDelta(const double & beta = 0.95, const double & epsilon = 1.0e-07);

  // Layer::Updater overwrites
  virtual std::string get_info() const;
  virtual std::unique_ptr<Layer::Updater> copy() const;

  virtual void init(const MatrixSize & rows, const MatrixSize & cols);
  virtual void start_epoch();
  virtual void reset();
  virtual void update(const RefConstMatrix & delta, const size_t & tests_num, RefMatrix value);
  virtual void update(const Value & delta, const size_t & tests_num, Value & value);

private:
  const double _beta;
  const double _epsilon;
  Matrix _ss;
  Matrix _dd;
  Matrix _delta;
}; // Updater_AdaDelta

}; // namespace yann

#endif /* UPDATERS_H_ */
