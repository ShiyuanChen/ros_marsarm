#ifndef CIRCLEELLIPSE_H
#define CIRCLEELLIPSE_H

/* Circle-Ellipse intersection algorithm */
/* intersect_triangle(double orig[3], double dir[3],
    double vert0[3], double vert1[3], double vert2[3],
    double *t, double *u, double *v) */
/* return 1 if intersects, else return 0 */

class circleEllipse
{
public:
    circleEllipse (int num_interactions);
    bool circleEllipseIntersection(double r, double x1, double y1, double w, double h);
    bool circleInEllipse(double r, double x1, double y1, double w, double h);
private:
    int numInteractions;
    double *innerPolygonCoef;
    double *outerPolygonCoef;

};

#endif // CIRCLEELLIPSE_H