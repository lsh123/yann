/*
 * test_utils.cpp
 *
 */
#include <boost/test/unit_test.hpp>

#include "core/types.h"
#include "test_utils.h"

using namespace std;
using namespace yann;
using namespace yann::test;

// Overwrites for BOOST_VERIFY()
namespace boost {
void assertion_failed(char const * expr, char const * function,
                      char const * file, long line)
{
  cerr << "ASSERT: " << expr << " in function " << function << " in file "
      << file << " on line " << line << endl;
  exit(1);
}

void assertion_failed_msg(char const * expr, char const * msg, char const * function,
                          char const * file, long line)
{
  cerr << "ASSERT: " << expr << " in function " << function << " in file "
      << file << " on line " << line << ": " << msg << endl;
  exit(1);
}
}; // namespace boost

// Block allocations in Eigen to test for unwanted object copies
yann::test::BlockAllocations::BlockAllocations(int flags) :
    _flags(flags)
{
   // ensure we don't do allocations in eigen
  if(!(_flags & Disabled)) {
    block();
  } else if(!(_flags & Silent)) {
    BOOST_TEST_MESSAGE("EIGEN ALLOCATIONS BLOCK DISABLED");
  }
}

yann::test::BlockAllocations::~BlockAllocations()
{
  // reenable allocations
  if(!(_flags & Disabled)) {
    unblock();
  }
}

void yann::test::BlockAllocations::block() {
#ifdef EIGEN_RUNTIME_NO_MALLOC
    if(!(_flags & Silent)) {
      BOOST_TEST_MESSAGE("EIGEN ALLOCATIONS BLOCKED");
    }
    Eigen::internal::set_is_malloc_allowed(false);
#endif /* EIGEN_RUNTIME_NO_MALLOC */
}

void yann::test::BlockAllocations::unblock() {
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
    if(!(_flags & Silent)) {
      BOOST_TEST_MESSAGE("EIGEN ALLOCATIONS UNBLOCKED");
    }
#endif /* EIGEN_RUNTIME_NO_MALLOC */
}

// Progress callback
void yann::test::batch_progress_callback(const MatrixSize & cur_pos, const MatrixSize & total, const std::string & message)
{
  // we want to print out progress at 1/10 increment
  auto progress_delta = total / 10;
  if(progress_delta > 0 && cur_pos % progress_delta == 0) {
    if(!message.empty()) {
      BOOST_TEST_MESSAGE("  ... at " << cur_pos << " out of " << total << " (" << message << ")");
    } else {
      BOOST_TEST_MESSAGE("  ... at " << cur_pos << " out of " << total);
    }
  }
}

void yann::test::ecpoch_progress_callback(const MatrixSize & cur_pos, const MatrixSize & total, const std::string & message)
{
  if(!message.empty()) {
    BOOST_TEST_MESSAGE("*** Epoch " << cur_pos << " out of " << total << " (" << message << ")");
  } else {
    BOOST_TEST_MESSAGE("*** Epoch " << cur_pos << " out of " << total);
  }
}
