//
// Add --log_level=message to see the messages!
//
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>

#include "core/types.h"
#include "core/random.h"
#include "core/utils.h"

#include "timer.h"
#include "test_utils.h"


using namespace std;
using namespace boost;
using namespace boost::unit_test;
using namespace boost::numeric;
using namespace yann;
using namespace yann::test;


struct TypesTestFixture
{
  TypesTestFixture()
  {
  }
  ~TypesTestFixture()
  {
  }
};

BOOST_FIXTURE_TEST_SUITE(TypesTest, TypesTestFixture);

////////////////////////////////////////////////////////////////////////////////////////////////
//
// perf functions tests
//
BOOST_AUTO_TEST_CASE(Perf_Test, * disabled())
{
  const size_t size = 1000;
  const size_t vector_size = 100000;
  const size_t data_size = size * size;

  // Eigen::Matrix * Vector
  {
      Timer timer("Eigen::Matrix*Vector: total");
      Matrix aa(size, vector_size);
      Vector bb(vector_size), cc(size);
      {
        Timer timer("Eigen::Matrix*Vector: random generation");
        unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
        gen->generate(aa);
        gen->generate(bb);
      }
      // ensure we don't do allocations in eigen
      {
        BlockAllocations block;
        {
          Timer timer("Eigen::Matrix*Vector: cc.noalias() = aa.lazyProduct(bb)");
          cc.noalias() = aa.lazyProduct(bb.transpose());
        }
      }
      {
        Timer timer("Eigen::Matrix*Vector: cc.noalias() = aa * bb;");
        cc.noalias() = aa * bb.transpose();
      }
      // ensure we don't do allocations in eigen
      {
        BlockAllocations block;
        {
          Timer timer("Eigen::Matrix*Vector: cc.noalias() = MatrixFunctions<RefConstMatrix>::product(aa, bb)");
          cc.noalias() = MatrixFunctions<RefConstMatrix>::product(aa, bb.transpose());
        }
      }
  }
  BOOST_TEST_MESSAGE("\n");

  // Eigen::Matrix * Sparse Vector
  {
      Timer timer("Eigen::Matrix*SparseVector: total");
      Matrix aa(vector_size, size);
      SparseMatrix bb(1, vector_size);
      Vector cc(size);
      {
        Timer timer("Eigen::Matrix*SparseVector: random generation");
        unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
        gen->generate(aa);
        bb.insert(1, 10) = 1.0;
      }
      // ensure we don't do allocations in eigen
      {
        BlockAllocations block;
        {
          Timer timer("Eigen::Matrix*SparseVector: cc.noalias() = bb * aa");
          cc.noalias() = bb * aa;
        }
      }
  }
  BOOST_TEST_MESSAGE("\n");

  // Eigen::Matrix * Matrix
  {
      Timer timer("Eigen::Matrix*Matrix: total");
      Matrix aa(size, size), bb(size, size), cc(size, size);
      {
        Timer timer("Eigen::Matrix*Matrix: random generation");
        unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
        gen->generate(aa);
        gen->generate(bb);
      }
      // ensure we don't do allocations in eigen
      {
        BlockAllocations block;
        {
          Timer timer("Eigen::Matrix*Matrix: cc.noalias() = aa.lazyProduct(bb)");
          cc.noalias() = aa.lazyProduct(bb);
        }
      }
      {
        Timer timer("Eigen::Matrix*Matrix: cc.noalias() = aa * bb");
        cc.noalias() = aa * bb;
      }
      // ensure we don't do allocations in eigen
      {
        BlockAllocations block;
        {
          Timer timer("Eigen::Matrix*Matrix: cc.noalias() = MatrixFunctions<RefConstSparseMatrix>::product(aa, bb)");
          cc.noalias() = MatrixFunctions<RefConstSparseMatrix>::product(aa, bb);
        }
      }
  }
  BOOST_TEST_MESSAGE("\n");

  // UBLAS::matrix
  {
      Timer timer("UBLAS Matrix*Matrix: total");
      ublas::matrix<Value> aa(size, size), bb(size, size), cc(size, size);
      {
        Timer timer("UBLAS Matrix*Matrix: random generation");
        unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
        gen->generate(aa.data().begin(), aa.data().end());
        gen->generate(bb.data().begin(), bb.data().end());
      }
      {
        Timer timer("UBLAS Matrix*Matrix: noalias(cc) = ublas::prod(aa, bb)");
        noalias(cc) = ublas::prod(aa, bb);
      }
      {
        Timer timer("UBLAS Matrix*Matrix: ublas::axpy_prod(aa, bb, cc, true)");
        ublas::axpy_prod(aa, bb, cc, true);
      }
  }
  BOOST_TEST_MESSAGE("\n");

  // std::vector
  {
    Timer timer("Value[] Matrix*Matrix: total");
    auto aa = make_unique<Value[]>(data_size);
    auto bb = make_unique<Value[]>(data_size);
    auto cc = make_unique<Value[]>(data_size);

    {
      Timer timer("Value[] Matrix*Matrix: random generation");
      unique_ptr<RandomGenerator> gen = RandomGenerator::normal_distribution(0, 1);
      gen->generate(aa.get(), aa.get() + data_size);
      gen->generate(bb.get(), bb.get() + data_size);
    }
    {
      Timer timer("Value[] Matrix*Matrix: ");
      auto aa_ptr = aa.get();
      auto bb_ptr = bb.get();
      auto cc_ptr = cc.get();
      memset(cc_ptr, 0, data_size);

      for(size_t ii = 0; ii < size; ++ii) {
        for(size_t jj = 0; jj < size; ++jj) {
          for(size_t kk = 0; kk < size; ++kk) {
            cc_ptr[ii*size + jj] += aa_ptr[ii*size + kk] * bb_ptr[kk*size + jj];
          }
        }
      }
    }
    {
      Timer timer("Value[] Matrix*Matrix 2: ");
      auto aa_ptr = aa.get();
      auto bb_ptr = bb.get();
      auto cc_ptr = cc.get();
      memset(cc_ptr, 0, data_size);

      for(size_t ii = 0; ii < size; ++ii) {
        for(size_t kk = 0; kk < size; ++kk) {
          for(size_t jj = 0; jj < size; ++jj) {
            cc_ptr[ii*size + jj] += aa_ptr[ii*size + kk] * bb_ptr[kk*size + jj];
          }
        }
      }
    }
  }
  BOOST_TEST_MESSAGE("\n");
}

BOOST_AUTO_TEST_SUITE_END()

