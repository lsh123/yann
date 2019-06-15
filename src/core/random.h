/*
 * random.h
 *
 */

#ifndef RANDOM_H_
#define RANDOM_H_

#include <algorithm>
#include <memory>

#include <boost/optional.hpp>

#include "core/types.h"

namespace yann {

class RandomGenerator
{
public:
  virtual Value next() = 0;

  template<typename Iterator>
  void generate(const Iterator & begin, const Iterator & end)
  {
    RandomGenerator* gen = this;
    std::generate(begin, end, [gen]()->Value { return gen->next(); });
  }

  void generate(Value & val);
  void generate(RefMatrix mm);

public:
  static std::unique_ptr<RandomGenerator> normal_distribution(
      const Value & mean, const Value & stddev,
      boost::optional<Value> seed = boost::none);
  static std::unique_ptr<RandomGenerator> uniform_distribution(
      const Value & aa, const Value & bb,
      boost::optional<Value> seed = boost::none);
}; // class RandomGenerator

}; // namespace yann

#endif /* RANDOM_H_ */
