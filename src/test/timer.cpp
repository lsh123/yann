/*
 * timer.cpp
 *
 */
#include <cstdlib>
#include <ctime>

#include <boost/assert.hpp>

#include "timer.h"

using namespace std;
using namespace yann::test;

namespace std {
ostream& operator<<(ostream & os, const yann::test::Timer & timer)
{
  yann::test::Timer & tt = const_cast<yann::test::Timer&>(timer);
  tt.stop();
  return tt.write(os);
}
}; // namespace std

std::string yann::test::Timer::get_time()
{
  time_t rawtime;
  struct tm timeinfo;
  char buffer[80];

  time(&rawtime);
  localtime_r(&rawtime, &timeinfo);
  strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", &timeinfo);
  return buffer;
}

yann::test::Timer::Timer(const std::string & message) :
        _message(message)
{
  _start = _stop = std::chrono::high_resolution_clock::now();
}

yann::test::Timer::~Timer()
{
  if (!is_stopped()) {
    stop();
    write(std::cout) << std::endl;
  }
}

bool yann::test::Timer::is_stopped() const
{
  return _start < _stop;
}

void yann::test::Timer::stop()
{
  _stop = std::chrono::high_resolution_clock::now();
}

std::ostream & yann::test::Timer::write(std::ostream & os)
{
  std::chrono::duration<double, std::milli> duration = (_stop - _start);
  os << _message << " time " << duration.count() << " milliseconds";
  return os;
}
