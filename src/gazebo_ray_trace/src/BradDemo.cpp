#include <iostream>
// #include <gazebo_ray_trace/plotRayUtils.h>
#include "plotRayUtils.h"

int main(int argc, char **argv){
  std::cout <<"Hello World" << std::endl;
  
  ros::init(argc, argv, "brads_demo_node");
  
  PlotRayUtils plt;

  //Start and end vectors of the ray
  tf::Point start(atof(argv[1]),
		  atof(argv[2]),
		  atof(argv[3]));
		  
  tf::Point end(atof(argv[4]),
		atof(argv[5]),
		atof(argv[6]));



  std::vector<double> dist = plt.getDistToParticles(start, end);

  for(int i=0; i < dist.size(); i++){
      ROS_INFO("Dist is %f", dist[i]);
  }

}
