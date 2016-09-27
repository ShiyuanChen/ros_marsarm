#ifndef RAY_TRACE_PLOTTER_H
#define RAY_TRACE_PLOTTER_H

#include "rayTracer.h"
#include <visualization_msgs/MarkerArray.h>

class RayTracePlotter: public RayTracer
{
 private:
  ros::NodeHandle n;
  ros::Publisher marker_pub;
  ros::Publisher marker_pub_array;

  visualization_msgs::Marker createRayMarker(Ray ray, int index);
  visualization_msgs::Marker getIntersectionMarker(tf::Point intersection, int id);
  visualization_msgs::Marker getDeleteMarker(string str, int id);
  void label(tf::Point start, int id, std::string text);


 public:
  RayTracePlotter();
  void plotRay(Ray ray, int index = 0);
  int plotRays(std::vector<Ray> rays, int id = 0);
  int plotIntersections(Ray ray, int id = 0);
  int plotIntersections(const std::vector<double> &dist, Ray ray, int id = 0);
  int plotIntersections(const std::vector<tf::Point> intersections, int id = 0);
  int plotCylinder(Ray ray, double radius, int id = 0);
  void plotIG(Ray ray);
  void plotRayWithIntersections(Ray ray);
  void plotCylinderWithIntersections(Ray ray, double radius);
  void labelRay(tf::Point start, std::string text, int id=0);
  void labelRay(Ray ray, double d, int id=0);

  void deleteAll();
};



#endif



