/*
 * test_utils.cpp
 *
 */
#include <boost/test/unit_test.hpp>

#include "types.h"
#include "test_utils.h"

using namespace std;
using namespace yann;
using namespace yann::test;

// Overwrites for YANN_CHECK()
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

