/**
 *  This is a rosnode that randomly chooses touch points, 
 *    performs simulated measurements, and updates the particle filter.
 *   This node is made to work with a particle filter node, a node that 
 *    publishes visualization messages, and RViz.
 */

#include <ros/ros.h>
#include "particle_filter/PFilterInit.h"
#include "particle_filter/AddObservation.h"
#include "geometry_msgs/Point.h"
#include "std_msgs/String.h"
#include <tf/transform_broadcaster.h>
#include "custom_ray_trace/plotRayUtils.h"
#include "custom_ray_trace/rayTracer.h"
#include "custom_ray_trace/rayTracePlotter.h"
#include <ros/console.h>
#include "stateMachine.h"
# define M_PI       3.14159265358979323846  /* pi */

/**
 * Gets initial points for the particle filter by shooting
 * rays at the object
 */
// particle_filter::PFilterInit getInitialPoints(RayTracePlotter &pmolt)
// {
//   particle_filter::PFilterInit init_points;

//   tf::Point start1(1,1,0);
//   tf::Point end1(-1,-1,0);
//   tf::Point start2(1,1,.2);
//   tf::Point end2(-1,-1,.2);
//   tf::Point start3(1.1,0.9,0);
//   tf::Point end3(-0.9,-1.1,0);
  
//   tf::Point intersection;
//   plt.getIntersectionWithPart(start1,end1, intersection);
//   tf::pointTFToMsg(intersection,  init_points.p1);
//   plt.getIntersectionWithPart(start2,end2, intersection);
//   tf::pointTFToMsg(intersection, init_points.p2);
//   plt.getIntersectionWithPart(start3,end3, intersection);
//   tf::pointTFToMsg(intersection, init_points.p3);
//   return init_points;
// }

void generateRandomTouchWith(tf::Pose &probePose, double tbX, double tbY, double tbZ, double tbRR, double tbRP, double tbRY, double rotateR, double rotateP, double rotateY, double offX, double offY, double offZ)
{

  // tf::Transform rotate;
  // rotate.setRotation(tf::createQuaternionFromRPY(rotateR, rotateP, rotateY));
  tf::Pose touchBase;
  touchBase.setOrigin(tf::Vector3(tbX, tbY, tbZ));
  // std::cout << "RR " << tbRR << ", RP " << tbRP << ", RY " << tbRY << std::cout;
  tf::Quaternion q;
  q.setRPY(tbRR, tbRP, tbRY);

  touchBase.setRotation(q);

  // tf::Pose offset;
  // offset.setOrigin(tf::Vector3(offX, offY, offZ));
  // offset.setRotation(tf::createQuaternionFromRPY(0,0,0));

  // probePose = rotate*offset*touchBase;
  probePose = touchBase;

}

    // [0.6, 0.6, -0.1, 0, 0, 0]
void generateRandomTouchBottom(std::mt19937 &gen, tf::Pose &probePose)
{
  std::uniform_real_distribution<double> rand(0,1.0);

  if(rand(gen)> 0.5){
    //Generate bottom left touch
    double x_width = 0.6*rand(gen);
    double y_width = 0.45*rand(gen);

    // double y_val = -0.51 + y_width;
    double y_val = 0 + y_width;

    double yaw = -1.5 + 3*y_val;
    yaw = min(-0.8, yaw);
    // std::cout << "yaw: "<< yaw << "\n";
    // generateRandomTouchWith(probePose, 
    // 			  .53 + x_width, .4 + y_width, .687, M_PI, 0, 0, 
    // 			  0,0,0,
    // 			  0,0,0);
    generateRandomTouchWith(probePose, 
			    // 0.8, -0.21, 0.45, M_PI, 0, M_PI, 
			    0.7 + x_width, y_val, 0.45, M_PI, 0, yaw,
			    0,0,0,
			    0,0,0);
  }else{
    //Gereate right touch
    double x_width = 0.6*rand(gen);
    double y_width = 0.35*rand(gen);

    // double y_val = -0.51 + y_width;
    double y_val = -.45 + y_width;

    double yaw = -1.5 + 3*y_val;
    yaw = min(-0.8, yaw);
    yaw = max(-2.0, yaw);

    generateRandomTouchWith(probePose, 
			    // 0.8, -0.21, 0.45, M_PI, 0, M_PI, 
			    0.7 + x_width, y_val, 0.45, M_PI, 0, yaw,
			    0,0,0,
			    0,0,0);
  }
}

void generateRandomTouchFront(std::mt19937 &gen, tf::Pose &probePose)
{
  std::uniform_real_distribution<double> rand(-1.0,1.0);
  double y_width = 0.30*rand(gen);
  double z_width = 0.15*rand(gen);

  double yaw = 0;

  if(y_width < 0){
    yaw = 3*y_width;
  }

  generateRandomTouchWith(probePose, 
			  0.75, -0.2+y_width, 0.61+z_width, M_PI/2, yaw, -M_PI/2,
  			  // 0.812, -0.050, 0.391, -1.396, -2.104, -1.468, 
  			  0,0,0,
  			  0,0,0);
}



// void generateRandomTouchFrontRight(std::mt19937 &gen, tf::Pose &probePose)
// {
//   std::uniform_real_distribution<double> rand(-1.0,1.0);
//   double y_width = 0.10*rand(gen);
//   double z_width = 0.15*rand(gen);

//   generateRandomTouchWith(probePose, 
// 			  .577, -.661+y_width, .260+z_width, -1.5708, 0, -2.39,
//   			  0,0,0,
//   			  0,0,0);
// }


void generateRandomTouchSide(std::mt19937 &gen, tf::Pose &probePose)
{
  std::uniform_real_distribution<double> rand(-1.0,1.0);
  double x_width = 0.15*rand(gen);
  double z_width = 0.15*rand(gen);
  generateRandomTouchWith(probePose, 
			  .58+x_width, .20, .61+z_width, M_PI/2, 0, 0,
			  // .71, .13, .4,  2.724, -1.23, 1.58,
			  0,0,0,
			  0,0,0);

}



void generateRandomRay(std::mt19937 &gen, tf::Pose &probePose, tf::Point &start, tf::Point &end)
{
  std::uniform_real_distribution<double> rand(0.0, 3.0);
  double faceNum = rand(gen);

  // generateRandomTouchSide(gen, probePose);


  // faceNum = 0.5; //HARDCODE FOR TESTING
  
  if(faceNum < 1.0)
    generateRandomTouchBottom(gen, probePose);
  else if(faceNum < 2.0)
    generateRandomTouchFront(gen, probePose);
  else
    generateRandomTouchSide(gen, probePose);
  
  // tf::Transform probeZ;
  // probeZ.setRotation(tf::createQuaternionFromRPY(0,0,0));
  // probeZ.setOrigin(tf::Vector3(0,0,0.1));


  // end = (probePose*probeZ).getOrigin();
  // start = tf::Point(0,0,0);
  // end = tf::Transform(probePose.getRotation()) * tf::Point(0,0,.1);


  start = probePose.getOrigin();
  end = probePose.getOrigin() + 
    tf::Transform(probePose.getRotation()) * tf::Point(0,0,-.25);
}





/**
 * Randomly chooses vectors, gets the Information Gain for each of 
 *  those vectors, and returns the ray (start and end) with the highest information gain
 */
void randomSelection(RayTracePlotter &plt, tf::Pose &probePose)
{
  // tf::Point best_start, best_end;

  double bestIG;
  bestIG = 0;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> rand(-1.0,1.0);

  tf::Point best_start, best_end;


  for(int i=0; i<1000; i++){
    // tf::Point start(rand(gen), rand(gen), rand(gen));
    // start = start.normalize();
    // tf::Point end(rand(gen), rand(gen), rand(gen));
    // end.normalized();
    tf::Point start, end;
    tf::Pose probePoseTmp;

    // ROS_DEBUG("Start of loop");
    generateRandomRay(gen, probePoseTmp, start, end);
    // plt.plotRay(start, end);
    Ray measurement(start, end);
    double IG = plt.getIG(measurement, 0.01, 0.002);
    
    // ROS_DEBUG("Got IG");

    plt.plotRay(measurement, i);
    // ROS_DEBUG("PlottedRay");

    if (IG > bestIG){
      
      bestIG = IG;
      best_start = start;
      best_end = end;
      probePose = probePoseTmp;
    }


    ROS_DEBUG_THROTTLE(10, "Calculating best point based on information gain: %d...", i);
  }
  // ros::Duration(2000).sleep();//DEBUG
  plt.deleteAll();
  plt.plotRay(Ray(best_start, best_end));
  plt.deleteAll();
  plt.plotRay(Ray(best_start, best_end));
  ros::Duration(0.3).sleep();
  plt.plotRay(Ray(best_start, best_end));
  plt.plotIntersections(Ray(best_start, best_end));
  // plt.plotCylinder(best_start, best_end, 0.01, 0.002, true);
  ROS_INFO("Ray is: %f, %f, %f.  %f, %f, %f", 
  	   best_start.getX(), best_start.getY(), best_start.getZ(),
  	   best_end.getX(), best_end.getY(), best_end.getZ());
  ROS_INFO("IG is: %f", bestIG);
  
}


// bool getIntersection(RayTracePlotter &plt, tf::Point start, tf::Point end, tf::Point &intersection){
//   bool intersectionExists = plt.getIntersectionWithPart(start, end, intersection);
//   double radius = 0.005;
//   intersection = intersection - (end-start).normalize() * radius;
//   return intersectionExists;
// }


geometry_msgs::Pose probeAt(tf::Transform rotate, tf::Transform base, double x, double y, double z, double r, double p, double yaw){
  tf::Pose offset;
  offset.setOrigin(tf::Vector3(x,y,z));
  offset.setRotation(tf::createQuaternionFromRPY(r, p, yaw));
  geometry_msgs::Pose probe_msg;
  tf::poseTFToMsg(rotate*offset*base, probe_msg);
  // probe_pub.publish(probe_msg);
  return probe_msg;

}


int main(int argc, char **argv)
{
  if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) ) {
    ros::console::notifyLoggerLevelsChanged();
  }

  ros::init(argc, argv, "updating_particles");
  ros::NodeHandle n;
  RayTracePlotter plt;

  std::random_device rd;
  std::mt19937 gen(rd());


  ROS_INFO("Running...");

  ros::Publisher probe_pub = 
    n.advertise<geometry_msgs::Pose>("/probe_point", 5);


 
  ros::Duration(2).sleep();
  
 

  double dtouch = 0.05;

  tf::Pose probePose;
  geometry_msgs::Pose probe_msg; 

  // tf::Point rStart(0.8, -0.21, 0.45);
  // tf::Point rEnd(0.8, -0.21, 0.35);


  // plt.plotRay(Ray(rStart, rEnd));


  for(int i=0; i<10; i++){
    ROS_INFO("\n------------------------------------------");
    ROS_INFO("Measurement %d", i);
    ROS_INFO("--------------------------------------------");

    randomSelection(plt,  probePose);
    tf::poseTFToMsg(probePose, probe_msg);


    while(!MotionStateMachine::isMotionFinished(n)){
      ROS_INFO_THROTTLE(30, "Waiting for previous movement to finish...");
      ros::spinOnce();
      ros::Duration(.1).sleep();
    }

    probe_pub.publish(probe_msg);
    ros::spinOnce();


    while(!plt.particleHandler.newParticles){
      ROS_INFO_THROTTLE(30, "Waiting for new particles...");
      ros::spinOnce();
      ros::Duration(.1).sleep();
    }
    ros::Duration(2.0).sleep();
  }


  ROS_INFO("Finished all action");

}
