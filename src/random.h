/*
 * random.h
 *
 */

#ifndef RANDOM_H_
#define RANDOM_H_

#include <algorithm>
#include <memory>

#include "types.h"

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
  static std::unique_ptr<RandomGenerator> normal_distribution(const Value & mean, const Value & stddev);
}; // class RandomGenerator

}; // namespace yann

#endif /* RANDOM_H_ */
