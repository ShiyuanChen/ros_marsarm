#include <string.h>
#include <iostream>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <chrono>
#include "tribox.h"
#include "raytri.h"
#include "circleEllipse.h"
#include "distanceTransformNew.h"
#include "particleFilter.h"
#include "matrix.h"
#include "Node.h"

#define COMBINE_RAYCASTING
#define ADAPTIVE_NUMBER
#define ADAPTIVE_BANDWIDTH

#define max3(a,b,c) ((a>b?a:b)>c?(a>b?a:b):c)
#define max2(a,b) (a>b?a:b)
#define min3(a,b,c) ((a<b?a:b)<c?(a<b?a:b):c)
#define min2(a,b) (a<b?a:b)

using namespace std;

Node::Node(int n_particles, particleFilter::cspace b_init[2]):numParticles(n_particles), type(0) {
  maxNumParticles = numParticles;

  particles.resize(numParticles);
  particlesPrev.resize(numParticles);

  createParticles(particlesPrev, b_init, numParticles);
  particles = particlesPrev;

#ifdef ADAPTIVE_BANDWIDTH
  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)particlesPrev.data(), cdim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
  cout << cov_mat << endl;
#endif
}

Node::Node(int n_particles, std::vector<Parent *> &p, int t):numParticles(n_particles), type(t) {
  maxNumParticles = numParticles;

  particles.resize(numParticles);
  particlesPrev.resize(numParticles);
  int size = p.size();
  for (int i = 0; i < size; i ++) {
    parent.push_back(p[i]);
    p[i]->node->child.push_back(this);
  }

  createParticles(p);
  particles = particlesPrev;

#ifdef ADAPTIVE_BANDWIDTH
  Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>((double *)particlesPrev.data(), cdim, numParticles);
  Eigen::MatrixXd mat_centered = mat.colwise() - mat.rowwise().mean();
  cov_mat = (mat_centered * mat_centered.adjoint()) / double(max2(mat.cols() - 1, 1));
  cout << cov_mat << endl;
#endif
}

/*
 * Create initial particles at start
 * Input: particles
 *        b_Xprior: prior belief
 *        n_partcles: number of particles
 * output: none
 */
void Node::createParticles(Particles &particles_dest, particleFilter::cspace b_Xprior[2],
                   int n_particles)
{
  random_device rd;
  normal_distribution<double> dist(0, 1);
  int cdim = sizeof(particleFilter::cspace) / sizeof(double);
  for (int i = 0; i < n_particles; i++) {
    for (int j = 0; j < cdim; j++) {
      particles_dest[i][j] = b_Xprior[0][j] + b_Xprior[1][j] * (dist(rd));
    }
  }
}

/*
 * Create initial particles at start
 * Input: particles
 *        b_Xprior: prior belief
 *        n_partcles: number of particles
 * output: none
 */
void Node::createParticles(std::vector<Parent *> &p)
{
  int size = p.size();
  if (size == 0) return;
  if (type == 1) {
    for (int i = 0; i < numParticles; i++) {
      for (int j = 0; j < cdim; j++) {
        particlesPrev[i][j] = 0;
      }
    }
    //particlesPrev = parent[0]->node->particles;
  } else if (type == 2) {
    Parent *plane = parent[0];
    Parent *edge = parent[1];
    
    double *edgeTol = edge->tol;
    double *edgeOffSet = edge->offset;
    std::random_device rd;
    std::uniform_int_distribution<> distEdge(0, edge->node->numParticles - 1);
    std::uniform_int_distribution<> distPlane(0, plane->node->numParticles - 1);
    std::uniform_real_distribution<double> dist(-1, 1);
    for (int i = 0; i < numParticles; i ++) {
      int indexEdge = int(distEdge(rd));
      particleFilter::cspace configEdge = edge->node->particles[indexEdge];
      Eigen::Vector3d pa, pb; 
      pa << configEdge[0], configEdge[1], configEdge[2];
      pb << configEdge[3], configEdge[4], configEdge[5];
      int indexPlane = int(distPlane(rd));
      particleFilter::cspace configPlane = plane->node->particles[indexPlane];
      Eigen::Vector3d pa_prime, pb_prime;
      inverseTransform(pa, configPlane, pa_prime);
      inverseTransform(pb, configPlane, pb_prime);
      double tx = dist(rd) * edgeTol[0];
      double tz = dist(rd) * edgeTol[2];
      pa_prime(1) = 0;
      pb_prime(1) = 0;
      pa_prime(0) += edgeOffSet[0] + tx;
      pb_prime(0) += edgeOffSet[0] + tx;
      pa_prime(2) += edgeOffSet[2] + tz;
      pb_prime(2) += edgeOffSet[2] + tz;
      Transform(pa_prime, configPlane, pa);
      Transform(pb_prime, configPlane, pb);
      particlesPrev[i][0] = pa(0);
      particlesPrev[i][1] = pa(1);
      particlesPrev[i][2] = pa(2);
      particlesPrev[i][3] = pb(0);
      particlesPrev[i][4] = pb(1);
      particlesPrev[i][5] = pb(2);
    }
  } else {
    Parent *plane = parent[0];
    Parent *edge1 = parent[1];
    Parent *edge2 = parent[2];

    double *edgeTol1 = edge1->tol;
    double *edgeOffSet1 = edge1->offset;
    double *edgeTol2 = edge2->tol;
    double *edgeOffSet2 = edge2->offset;

    std::random_device rd;
    std::uniform_int_distribution<> distEdge1(0, edge1->node->numParticles - 1);
    std::uniform_int_distribution<> distEdge2(0, edge2->node->numParticles - 1);
    std::uniform_int_distribution<> distPlane(0, plane->node->numParticles - 1);
    std::uniform_real_distribution<double> dist(-1, 1);
    for (int i = 0; i < numParticles; i ++) {
      int indexEdge1 = int(distEdge1(rd));
      int indexEdge2 = int(distEdge2(rd));
      particleFilter::cspace configEdge1 = edge1->node->particles[indexEdge1];
      particleFilter::cspace configEdge2 = edge2->node->particles[indexEdge2];
      Eigen::Vector3d pa1, pb1, pa2, pb2;
      pa1 << configEdge1[0], configEdge1[1], configEdge1[2];
      pb1 << configEdge1[3], configEdge1[4], configEdge1[5];
      pa2 << configEdge2[0], configEdge2[1], configEdge2[2];
      pb2 << configEdge2[3], configEdge2[4], configEdge2[5];
      int indexPlane = int(distPlane(rd));
      particleFilter::cspace configPlane = plane->node->particles[indexPlane];
      Eigen::Vector3d pa1_prime, pb1_prime, pa2_prime, pb2_prime;
      inverseTransform(pa1, configPlane, pa1_prime);
      inverseTransform(pb1, configPlane, pb1_prime);
      inverseTransform(pa2, configPlane, pa2_prime);
      inverseTransform(pb2, configPlane, pb2_prime);
      double tx = dist(rd) * edgeTol1[0];
      double tz = dist(rd) * edgeTol1[2];
      pa1_prime(1) = 0;
      pb1_prime(1) = 0;
      pa2_prime(1) = 0;
      pb2_prime(1) = 0;
      pa1_prime(0) += edgeOffSet1[0] + tx;
      pb1_prime(0) += edgeOffSet1[0] + tx;
      pa1_prime(2) += edgeOffSet1[2] + tz;
      pb1_prime(2) += edgeOffSet1[2] + tz;
      tx = dist(rd) * edgeTol2[0];
      tz = dist(rd) * edgeTol2[2];
      pa2_prime(0) += edgeOffSet2[0] + tx;
      pb2_prime(0) += edgeOffSet2[0] + tx;
      pa2_prime(2) += edgeOffSet2[2] + tz;
      pb2_prime(2) += edgeOffSet2[2] + tz;

      Eigen::Matrix2d divisor, dividend; 
      divisor << pa1_prime(0) - pb1_prime(0), pa1_prime(2) - pb1_prime(2),
                 pa2_prime(0) - pb2_prime(0), pa2_prime(2) - pb2_prime(2);
      dividend << pa1_prime(0)*pb1_prime(2) - pa1_prime(2)*pb1_prime(0), pa1_prime(0) - pb1_prime(0),
                  pa2_prime(0)*pb2_prime(2) - pa2_prime(2)*pb2_prime(0), pa2_prime(0) - pb2_prime(0);
      Eigen::Vector3d pi_prime, pi, dir_prime, origin_prime, dir, origin;
      pi_prime(0) = dividend.determinant() / divisor.determinant();
      dividend << pa1_prime(0)*pb1_prime(2) - pa1_prime(2)*pb1_prime(0), pa1_prime(2) - pb1_prime(2),
                  pa2_prime(0)*pb2_prime(2) - pa2_prime(2)*pb2_prime(0), pa2_prime(2) - pb2_prime(2);
      pi_prime(1) = 0;
      pi_prime(2) = dividend.determinant() / divisor.determinant();
      dir_prime << 0, 1, 0; origin_prime << 0, 0, 0;
      Transform(pi_prime, configPlane, pi);
      Transform(dir_prime, configPlane, dir);
      Transform(origin_prime, configPlane, origin);
      dir -= origin;
      particlesPrev[i][0] = pi(0);
      particlesPrev[i][1] = pi(1);
      particlesPrev[i][2] = pi(2);
      particlesPrev[i][3] = dir(0);
      particlesPrev[i][4] = dir(1);
      particlesPrev[i][5] = dir(2);
    }
  }
}