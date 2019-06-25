/*
 * test_utils.h
 *
 */

#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <fstream>

#include <boost/assert.hpp>

#include "timer.h"

#define TEST_TOLERANCE    0.001
#define TMP_FOLDER  "/tmp/"

namespace yann::test {

// Block allocations in Eigen to test for unwanted object copies.
// Requires -DDEIGEN_RUNTIME_NO_MALLOC (set automatically with
// "./configure --enable-debug")
class BlockAllocations {
public:
  enum Flags {
    None     = 0x0000,
    Disabled = 0x0001,
    Silent   = 0x0002,
  };

public:
  BlockAllocations(int flags = None);
  virtual ~BlockAllocations();

  void block();
  void unblock();

private:
  int _flags;
}; // class BlockAllocations

// defines a range to test
class Range
{
public:
  Range(size_t min, size_t max, size_t step) :
          _min(min),
          _max(max),
          _step(step)
  {
  }

  inline const size_t& min() const { return _min; }
  inline const size_t& max() const { return _max; }
  inline const size_t& step() const { return _step; }

private:
  size_t _min, _max, _step;
}; // class Range

// Progress callback for Trainer::ProgressCallback
void batch_progress_callback(const MatrixSize & cur_pos, const MatrixSize & total, const std::string & message);
void epoch_progress_callback(const MatrixSize & cur_pos, const MatrixSize & total, const std::string & message);

template<typename ObjectType>
void save_to_file(const ObjectType & obj, const std::string & ext)
{
  // save
  std::string filename = TMP_FOLDER + Timer::get_time() + "." + ext;
  std::ofstream ofs(filename, std::ofstream::out | std::ofstream::trunc);
  if(!ofs || ofs.fail()) {
    throw std::runtime_error("can't open file " + filename);
  }
  ofs << obj;
  if(!ofs || ofs.fail()) {
    throw std::runtime_error("can't write to file " + filename);
  }
  ofs.close();
  BOOST_TEST_MESSAGE("*** Saved to file: " << filename);
}
}; // namespace yann::test

#endif /* TEST_UTILS_H_ */
