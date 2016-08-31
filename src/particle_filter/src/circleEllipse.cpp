#include <iostream>
#include <cmath>
#include "circleEllipse.h"
using namespace std;
# define Pi          3.141592653589793238462643383279502884L

circleEllipse::circleEllipse (int num_interactions): numInteractions(num_interactions)
{
    innerPolygonCoef = new double[numInteractions + 1];
    outerPolygonCoef = new double[numInteractions + 1];
    for (int t = 0; t <= numInteractions; t++) {
        int numNodes = 4 << t;
        innerPolygonCoef[t] = 0.5 / cos(2 * Pi / numNodes); // Includes factor 0.5 for averaging
        outerPolygonCoef[t] = 0.5 / (cos(Pi / numNodes)*cos(Pi / numNodes)); // Includes factor 0.5 for averaging
    }
}
bool circleEllipse::circleEllipseIntersection(double r, double x1, double y1, double w, double h)
{
    bool collided = false;
    bool isfinish = false;
    double y = abs(y1);
    double x = abs(x1);
    //cout << "Y: " << y << endl;
    if ((x*x + (h - y)*(h - y) <= r*r) || ((w - x)*(w - x) + y*y <= r*r)) {
        collided = true;
    }
    else if (x*h + y*w + r*sqrt(h*h + w*w) <= w*h) {
        collided = false;
    }
    else if ((x - w)*(x - w) + (y - h)*(y - h) <= r*r || (x <= w && y - r <= h) || (y <= h && x - r <= w)) {
        // Collision within triangle (0, h) (w, h) (0, 0) is possible 
        double c0x = w;
        double c0y = 0;
        double c2x = 0;
        double c2y = h;
        for (int t = 1; t <= numInteractions; t++) {
            double c1x = (c0x + c2x)*innerPolygonCoef[t];
            double c1y = (c0y + c2y)*innerPolygonCoef[t];
            
            
                                         // Collision within triangles c3---c1---c2 and c4---c1---c2 is possible
            double tx = x - c1x; // t indicates a translated coordinate
            double ty = y - c1y;
            if (tx*tx + ty*ty <= r*r) {
                // Collision with t1
                collided = true;
                break;
            }
            double t2x = c2x - c1x;
            double t2y = c2y - c1y;
            double t0x = c0x - c1x;
            double t0y = c0y - c1y;
            if (tx*t2x + ty*t2y <= t2x*t2x + t2y*t2y && tx*t0x + ty*t0y <= t0x*t0x + t0y*t0y && 
                (ty*t2x - tx*t2y >= 0 && r*r*(t2x*t2x + t2y*t2y) < (ty*t2x - tx*t2y)*(ty*t2x - tx*t2y)) && 
                (ty*t0x - tx*t0y <= 0 && r*r*(t0x*t0x + t0y*t0y) < (ty*t0x - tx*t0y)*(ty*t0x - tx*t0y))) {
                // Collision with t1---t0
                collided = false;
                break;
            }
            double c3x = (c0x + c1x)*outerPolygonCoef[t]; // Can be calculated later if no visualization
            double c3y = (c0y + c1y)*outerPolygonCoef[t]; // Can be calculated later if no visualization
            if ((c3x - x)*(c3x - x) + (c3y - y)*(c3y - y) < r*r) {
                // t3 is inside circle
                c2x = c1x;
                c2y = c1y;
                
                continue;
            }
            double c4x = c1x - c3x + c1x; // Can be calculated later if no visualization
            double c4y = c1y - c3y + c1y; // Can be calculated later if no visualization
            if ((c4x - x)*(c4x - x) + (c4y - y)*(c4y - y) < r*r) {
                // t4 is inside circle
                c0x = c1x;
                c0y = c1y;
                
                continue;
            }
            double t3x = c3x - c1x;
            double t3y = c3y - c1y;

            if (ty*t3x - tx*t3y <= 0 || r*r*(t3x*t3x + t3y*t3y) > (ty*t3x - tx*t3y)*(ty*t3x - tx*t3y)) {
                if (tx*t3x + ty*t3y > 0) {
                    if (abs(tx*t3x + ty*t3y) <= t3x*t3x + t3y*t3y || (x - c3x)*(c0x - c3x) + (y - c3y)*(c0y - c3y) >= 0) {
                        // Circle center is inside t0---t1---t3
                        c2x = c1x;
                        c2y = c1y;
                        continue;
                    }
                }
                else if (abs(tx*t3x + ty*t3y) <= t3x*t3x + t3y*t3y || (x - c4x)*(c2x - c4x) + (y - c4y)*(c2y - c4y) >= 0) {
                    // Circle center is inside t1---t2---t4
                    c0x = c1x;
                    c0y = c1y;
                    continue;
                }
            }
            // No collision possible
            break;
        }
    }
    return collided;
}
bool circleEllipse::circleInEllipse(double r, double x1, double y1, double w, double h)
{
    double collided = circleEllipseIntersection(r, x1, y1, w, h);
    if (!collided) {
        double w2 = w*w;
        double h2 = h*h;
        if (x1*x1/w2 + y1*y1/h2 < 1)
            return true;
    }
    return false;
}