/*
 * random.cpp
 *
 */
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>

#include <boost/assert.hpp>

#include "random.h"

using namespace std;
using namespace yann;



namespace yann {
////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::RandomGenerator_NormalDistribution implementation
//
class RandomGenerator_NormalDistribution : public RandomGenerator {
  friend class RandomGenerator;

public:
  RandomGenerator_NormalDistribution(const Value & mean, const Value & stddev) :
    _gen(_rd()),
    _dist(mean, stddev)
  {
  }

  // RandomGenerator overwrites
  Value next() { return _dist(_gen); }

private:
  RandomGenerator_NormalDistribution(const RandomGenerator_NormalDistribution &) = delete;
  RandomGenerator_NormalDistribution& operator=(const RandomGenerator_NormalDistribution &) = delete;

private:
  std::random_device _rd;
  std::mt19937 _gen;
  std::normal_distribution<Value> _dist;
}; // class RandomGenerator_NormalDistribution

}; // RandomGenerator

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::RandomGenerator implementation
//
std::unique_ptr<RandomGenerator> yann::RandomGenerator::normal_distribution(
    const Value & mean, const Value & stddev)
{
  return make_unique<RandomGenerator_NormalDistribution>(mean, stddev);
}

void yann::RandomGenerator::generate(Value & val)
{
  val = next();
}

void yann::RandomGenerator::generate(RefMatrix mm)
{
  for(auto ii = 0; ii < mm.rows(); ++ii) {
    for(auto jj = 0; jj < mm.cols(); ++jj) {
      mm(ii, jj) = next();
    }
  }
}

