/*
 * timer.h
 *
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>
#include <iostream>
#include <string>

namespace yann::test {
class Timer {
public:
  Timer(const std::string & message);
  ~Timer();

  bool is_stopped() const;
  void stop();

  std::ostream & write(std::ostream & os);

  static std::string get_time();

private:
  // disable copy constructor and assignment operator
  Timer(const Timer & other) = delete;
  Timer & operator=(const Timer & other) = delete;

private:
  std::string _message;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start;
  std::chrono::time_point<std::chrono::high_resolution_clock> _stop;
};

};// namespace yann::test

// Overwrites from std:: namespace
namespace std {
ostream& operator<<(ostream & os, const yann::test::Timer & timer);
}; // namespace std

#endif /* TIMER_H_ */
