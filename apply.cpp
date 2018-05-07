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

//For lambda expressions
using boost::compute::lambda::_1;



//External function to generate integer in [0:100)
int rand_int()
{
    return rand() % 100;
}

//Simple function to add 4 to an integer
BOOST_COMPUTE_FUNCTION(int, add_four, (int x),
{
    return x + 4;
});

// function returing true if the value is less than 50.
BOOST_COMPUTE_FUNCTION(bool, is_less_than_half, (const int value),
                       {
                         return (value < 50);
                       });


int main()
{

  // get default device and setup context
  compute::device device = compute::system::default_device();
  compute::context context(device);
  compute::command_queue queue(context, device);

  // generate random data on the host
  std::vector<int> host_vector(10000); std::generate(host_vector.begin(), host_vector.end(), rand_int);

  // create a vector on the device
  compute::vector<int> device_vector(host_vector.size(), context);

  // transfer data from the host to the device
  compute::copy(host_vector.begin(), host_vector.end(), device_vector.begin(), queue);

  // calculate the square-root of each element in-place
  compute::transform(
                     device_vector.begin(),
                     device_vector.end(),
                     device_vector.begin(),
                     add_four,
                     queue
                     );


  // Substracting 4  using lambda expression
  boost::compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), _1 - 4, queue);


  // count number of random int less than 50
  size_t count = compute::count_if(device_vector.begin(),device_vector.end(), is_less_than_half, queue);

  //double check on CPU
  int count_cpu = std::count_if(host_vector.begin(), host_vector.end(), [](int i){return i < 50;});


  std::cout << "Count= "<<count<< "  CPU count= "<<count_cpu<<std::endl;

}
