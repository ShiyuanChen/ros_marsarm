/**
 *  Plots a ray and the intersections of that ray with obstacles 
 */

#include "ros/ros.h"

#include "calcEntropy.h"
#include "plotRayUtils.h"
#include <sstream>



int main(int argc, char **argv){
  if (argc != 7){
    ROS_INFO("usage: x y z x y z");
    return 1;
  }
  ros::init(argc, argv, "ray_trace_test");


  //Start and end vectors of the ray
  tf::Point start(atof(argv[1]),
		  atof(argv[2]),
		  atof(argv[3]));
		  
  tf::Point end(atof(argv[4]),
		atof(argv[5]),
		atof(argv[6]));

  PlotRayUtils plt;

  plt.plotRay(start, end);

 
  std::vector<double> dist = plt.getDistToParticles(start, end);

  plt.plotIntersections(start, end);

  double entropy = CalcEntropy::calcEntropy(dist);
  ROS_INFO("Entropy is %f", entropy);
  std::stringstream s;
  s << entropy;
  plt.labelRay(start, s.str());

  return 0;
}
  












