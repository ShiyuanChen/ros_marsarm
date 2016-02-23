#include <ros/ros.h>
#include "particle_filter/PFilterInit.h"
#include "particle_filter/AddObservation.h"
#include "geometry_msgs/Point.h"
#include <tf/transform_broadcaster.h>
#include "gazebo_ray_trace/plotRayUtils.h"

particle_filter::PFilterInit getInitialPoints(PlotRayUtils &plt)
{
  particle_filter::PFilterInit init_points;

  tf::Point start1(1,1,0);
  tf::Point end1(-1,-1,0);
  tf::Point start2(1,1,.2);
  tf::Point end2(-1,-1,.2);
  tf::Point start3(1.1,0.9,0);
  tf::Point end3(-0.9,-1.1,0);
  
  tf::Point intersection;
  plt.getIntersectionWithPart(start1,end1, intersection);
  tf::pointTFToMsg(intersection,  init_points.p1);
  plt.getIntersectionWithPart(start2,end2, intersection);
  tf::pointTFToMsg(intersection, init_points.p2);
  plt.getIntersectionWithPart(start3,end3, intersection);
  tf::pointTFToMsg(intersection, init_points.p3);
  return init_points;
}


void randomSelection(PlotRayUtils &plt, tf::Point &best_start, tf::Point &best_end)
{
  // tf::Point best_start, best_end;

  double bestIG;
  bestIG = 0;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> rand(0.0,1.0);


  for(int i=0; i<500; i++){
    tf::Point start(rand(gen), rand(gen), 2*rand(gen)-1);
    tf::Point end(rand(gen)-1, rand(gen)-1, 2*rand(gen)-1);
    end.normalized();
    double IG = plt.getIG(start, end, 0.01, 0.002);
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



int main(int argc, char **argv)
{
  ros::init(argc, argv, "updating_particles");
  ros::NodeHandle n;
  PlotRayUtils plt;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> randn(0.0,0.003);

  ROS_INFO("Running...");

  ros::Publisher pub_init = 
    n.advertise<particle_filter::PFilterInit>("/particle_filter_init", 5);
  ros::ServiceClient srv_add = 
    n.serviceClient<particle_filter::AddObservation>("/particle_filter_add");
    // n.advertise<geometry_msgs::Point>("/particle_filter_add", 5);

 
  ros::Duration(2).sleep();
  pub_init.publish(getInitialPoints(plt));
 
  geometry_msgs::Point obs;
 
  for(int i=0; i<10; i++){
    ros::Duration(1).sleep();
    // tf::Point start(1,1.4,.5);
    // tf::Point end(-1,-.6,.5);
    tf::Point start, end;
    randomSelection(plt, start, end);
    tf::Point intersection;
    if(!plt.getIntersectionWithPart(start,end, intersection)){
      ROS_INFO("NO INTERSECTION, Skipping");
      break;
    }
    obs.x=intersection.getX() + randn(gen); 
    obs.y=intersection.getY() + randn(gen); 
    obs.z=intersection.getZ() + randn(gen);
    // pub_add.publish(obs);
    
    plt.plotCylinder(start, end, 0.01, 0.002, true);
    ros::Duration(1).sleep();

    particle_filter::AddObservation pfilter_obs;
    pfilter_obs.request.p = obs;
    if(!srv_add.call(pfilter_obs)){
      ROS_INFO("Failed to call add observation");
    }

  }

  ROS_INFO("Finished all action");

}