/*
 * random.cpp
 *
 */
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>

#include <boost/assert.hpp>

#include "core/random.h"

using namespace std;
using namespace boost;
using namespace yann;


namespace yann {

typedef mt19937 DefaultGenerator;

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::RandomGenerator_NormalDistribution implementation
//
class RandomGenerator_NormalDistribution : public RandomGenerator {
  friend class RandomGenerator;

public:
  RandomGenerator_NormalDistribution(const Value & mean, const Value & stddev, optional<Value> seed) :
    _gen(seed ? (unsigned)(*seed) : _rd()),
    _dist(mean, stddev)
  {
  }

  // RandomGenerator overwrites
  Value next() { return _dist(_gen); }

private:
  RandomGenerator_NormalDistribution(const RandomGenerator_NormalDistribution &) = delete;
  RandomGenerator_NormalDistribution& operator=(const RandomGenerator_NormalDistribution &) = delete;

private:
  random_device _rd;
  DefaultGenerator _gen;
  std::normal_distribution<Value> _dist;
}; // class RandomGenerator_NormalDistribution

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::RandomGenerator_UniformDistribution implementation
//
class RandomGenerator_UniformDistribution : public RandomGenerator {
  friend class RandomGenerator;

public:
  RandomGenerator_UniformDistribution(const Value & aa, const Value & bb, optional<Value> seed) :
    _gen(seed ? (unsigned)(*seed) : _rd()),
    _dist(aa, bb)
  {
  }

  // RandomGenerator overwrites
  Value next() { return _dist(_gen); }

private:
  RandomGenerator_UniformDistribution(const RandomGenerator_UniformDistribution &) = delete;
  RandomGenerator_UniformDistribution& operator=(const RandomGenerator_UniformDistribution &) = delete;

private:
  random_device _rd;
  DefaultGenerator _gen;
  uniform_real_distribution<Value> _dist;
}; // class RandomGenerator_NormalDistribution


}; // RandomGenerator

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::RandomGenerator implementation
//
unique_ptr<RandomGenerator> yann::RandomGenerator::normal_distribution(
    const Value & mean, const Value & stddev, optional<Value> seed)
{
  return make_unique<RandomGenerator_NormalDistribution>(mean, stddev, seed);
}

unique_ptr<RandomGenerator> yann::RandomGenerator::uniform_distribution(
    const Value & aa, const Value & bb, optional<Value> seed)
{
  return make_unique<RandomGenerator_UniformDistribution>(aa, bb, seed);
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

