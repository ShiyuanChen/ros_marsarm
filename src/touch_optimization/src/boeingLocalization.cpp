/**
 *  This is a rosnode that randomly chooses touch points, 
 *    performs simulated measurements, and updates the particle filter.
 *   This node is made to work with a particle filter node, a node that 
 *    publishes visualization messages, and RViz.
 */
#include <iostream>
#include <ros/ros.h>
#include "particle_filter/PFilterInit.h"
#include "particle_filter/AddObservation.h"
#include "geometry_msgs/Point.h"
#include <tf/transform_broadcaster.h>
#include "custom_ray_trace/rayTracePlotter.h"
#include "custom_ray_trace/rayTracer.h"

#define NUM_TOUCHES 20

/**
 * Randomly chooses vectors, gets the Information Gain for each of 
 *  those vectors, and returns the ray (start and end) with the highest information gain
 */
void randomSelection(RayTracePlotter &plt, tf::Point &best_start, tf::Point &best_end)
{
  // tf::Point best_start, best_end;

  double bestIG;
  bestIG = 0;
  std::random_device rd;
  std::uniform_real_distribution<double> rand(-2.0,2.0);


  for(int i=0; i<500; i++){
    tf::Point start(rand(rd), rand(rd), rand(rd));
    start = start.normalize();
    tf::Point end(rand(rd), rand(rd), rand(rd));
    end.normalized();
    double IG = plt.getIG(Ray(start, end), 0.01, 0.002);
    if (IG > bestIG){
      bestIG = IG;
      best_start = start;
      best_end = end;
    }
  }

  // plt.plotCylinder(best_start, best_end, 0.01, 0.002, true);
  ROS_INFO("Ray is: %f, %f, %f.  %f, %f, %f", 
	   best_start.getX(), best_start.getY(), best_start.getZ(),
	   best_end.getX(), best_end.getY(), best_end.getZ());
  
}

bool getIntersection(RayTracePlotter &plt, tf::Point start, tf::Point end, tf::Point &intersection){
  double dist;
  Ray ray(start, end);
  bool intersectionExists = plt.traceRay(ray, dist);
  double radius = 0.001;
  intersection = ray.travelAlongFor(dist);
  intersection = intersection - (end-start).normalize() * radius;
  return intersectionExists;
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "updating_particles");
  ros::NodeHandle n;
  RayTracePlotter plt;

  std::random_device rd;
  std::normal_distribution<double> randn(0.0,0.0005);

  ROS_INFO("Running...");

  ros::Publisher pub_init = 
    n.advertise<particle_filter::PFilterInit>("particle_filter_init", 5);
  ros::ServiceClient srv_add = 
    n.serviceClient<particle_filter::AddObservation>("particle_filter_add");


 
  ros::Duration(2).sleep();

 
  geometry_msgs::Point obs;
  geometry_msgs::Point dir;

  int i = 0;
  //for(int i=0; i<20; i++){
  while (i < NUM_TOUCHES) {
    ros::Duration(2).sleep();
    //tf::Point start(0.95,0,-0.15);
    //tf::Point end(0.95,2,-0.15);
    tf::Point start, end;
    randomSelection(plt, start, end);

    tf::Point intersection;
    if(!getIntersection(plt, start, end, intersection)){
      ROS_INFO("NO INTERSECTION, Skipping");
      continue;
    }
	std::cout << "Intersection at: " << intersection.getX() << "  " << intersection.getY() << "   " << intersection.getZ() << std::endl;
    tf::Point ray_dir(end.x()-start.x(),end.y()-start.y(),end.z()-start.z());
    ray_dir = ray_dir.normalize();
    obs.x=intersection.getX() + randn(rd); 
    obs.y=intersection.getY() + randn(rd); 
    obs.z=intersection.getZ() + randn(rd);
    dir.x=ray_dir.x();
    dir.y=ray_dir.y();
    dir.z=ray_dir.z();
    // obs.x=intersection.getX(); 
    // obs.y=intersection.getY(); 
    // obs.z=intersection.getZ();

    // pub_add.publish(obs);
    

    // plt.plotCylinder(start, end, 0.01, 0.002, true);
    plt.plotRay(Ray(start, end));
    plt.plotIntersection(intersection);
    
    ros::Duration(1).sleep();

    particle_filter::AddObservation pfilter_obs;
    pfilter_obs.request.p = obs;
    pfilter_obs.request.dir = dir;
    if(!srv_add.call(pfilter_obs)){
      ROS_INFO("Failed to call add observation");
    }
    i ++;

    ros::Duration(1).sleep();
    ros::spinOnce();
  }
  
  ROS_INFO("Finished all action");

}
