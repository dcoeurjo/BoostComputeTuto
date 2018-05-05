#include <iostream>
#include <boost/compute/core.hpp>
#include <vector>

#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>
#include <boost/compute/algorithm/count_if.hpp>
#include <boost/compute/function.hpp>

namespace compute = boost::compute;


const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                                                     __kernel void computeDiff(__global const float* positionx,
                                                                               __global const float* positiony,
                                                                               __global float *diff
                                                                              )
                                                     {
                                                       uint gid = get_global_id(0);

                                                       diff[gid]= positionx[gid] - positiony[gid];
                                                     }
                                                     );


int main()
{

  /*
  * Toy example OpenCL kernel in Boost::Compute
  *
  * Given a sequence of pairs {(x_i,y_i)}, the kernel
  * computes diff_i = x_i - y_i.
  *
  */

  // get default device and setup context
  compute::device device = compute::system::default_device();
  compute::context context(device);
  compute::command_queue queue(context, device);

  //CPU Data
  std::vector<float> tabx = { 0.0,1.0,3.0};
  std::vector<float> taby = { 0.0,2.0,40.0};

  //create buffers
  compute::vector<float> m_originalPosX(3, context);
  compute::copy(tabx.begin(), tabx.end(), m_originalPosX.begin(), queue);
  compute::vector<float> m_originalPosY(3, context);
  compute::copy(taby.begin(), taby.end(), m_originalPosY.begin(), queue);

  //diff vector
  compute::vector<float> diff(3, context);
  std::fill(diff.begin(), diff.end(), 0);

  //create and build the kernels
  compute::program m_program;
  m_program = compute::program::create_with_source(source, context);
  try
  {
    m_program.build();
  }
  catch(boost::compute::opencl_error &e)
  {
    // program failed to compile, print out the build log
    std::cout << m_program.build_log() << std::endl;
  }

  //Set up the compute kernel
  compute::kernel diff_kernel;
  diff_kernel = m_program.create_kernel("computeDiff");
  diff_kernel.set_arg(0, m_originalPosX.get_buffer());
  diff_kernel.set_arg(1, m_originalPosY.get_buffer());
  diff_kernel.set_arg(2, diff.get_buffer());

  //Go Go go..
  queue.enqueue_1d_range_kernel(diff_kernel, 0, 3, 0);

  //Get back the diff
  float diff_host[3];
  compute::copy(diff.begin(), diff.end(), diff_host, queue);
  for(auto i=0; i < 3; ++i)
    std::cout<< diff_host[i] <<" -- ";
  std::cout<< std::endl;
}
