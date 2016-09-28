#ifndef NODE_H
#define NODE_H
#include <vector>
#include <array>
#include <cstring>
#include <unordered_set>
#include <Eigen/Dense>
#include "distanceTransformNew.h"
#include "particleFilter.h"

using namespace std;
typedef array<array<float, 3>, 4> vec4x3;
class Parent;

// class particleFilter;
class Node
{
  friend class particleFilter;
  friend class Parent;
 public:
  static const int cdim = 6;
  //typedef std::array<double,cdim> cspace; // configuration space of the particles
  typedef std::vector<particleFilter::cspace> Particles;
  int numParticles; // number of particles
  int maxNumParticles;

  Node (int n_particles, particleFilter::cspace b_init[2]);
  Node (int n_particles, std::vector<Parent *> &p, int type);
  // void addObservation (double obs[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, bool miss = false);
  void estimateGaussian(particleFilter::cspace &x_mean, particleFilter::cspace &x_est_stat);
  void getAllParticles(Particles &particles_dest);

 protected:
  // Parameters of Node
  int type; // 0: root; 1: plane; 2. edge; 3. hole
  // double Xstd_ob; // observation measurement error
  // double R; // probe radius

  // internal variables
  particleFilter::cspace b_Xprior[2]; // Initial distribution (mean and variance)
  //cspace b_Xpre[2];   // Previous (estimated) distribution (mean and variance)
  std::vector<Node*> child;
  std::vector<Parent*> parent;
  Particles particles;  // Current set of particles
  Particles particlesPrev; // Previous set of particles
  // Particles particles_1; // Previous previous set of particles
  
  Eigen::MatrixXd cov_mat;

  // Local functions
  void createParticles(Particles &particles, particleFilter::cspace b_Xprior[2], int n_particles);
  void createParticles(std::vector<Parent *> &p);
  bool updateParticles(double cur_M[2][3], vector<vec4x3> &mesh, distanceTransform *dist_transform, double Xstd_ob, double R, bool miss);


};

class Parent
{
  friend class particleFilter;
  friend class Node;
public:
  Parent(Node *p, double x, double y, double z, double tx, double ty, double tz) {
    node = p;
    type = p->type;
    offset[0] = x;
    offset[1] = y;
    offset[2] = z;
    tol[0] = tx;
    tol[1] = ty;
    tol[2] = tz;
  }
protected:
  Node *node;
  int type;
  double tol[3];
  double offset[3];

};

#endif // NODE_H

