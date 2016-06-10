#include <cstring>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <Eigen/Dense>
#include <array>
#include "stlParser.h"


using namespace std;

// typedef array<array<float, 3>, 4> vec4x3;

/*
* Import binary STL file to arrays.
* Input: filename: STL file name
* Output: Triangle mesh vector.
*/
pluckerMesh StlParser::importSTL(string filename)
{
	ifstream stlFile;
	char title[80];
	unsigned int num_triangles;
	short attribute;
	stlFile.open(filename, ios::binary | ios::in);
	if (!stlFile) {
	  cerr << "Cant open " << filename << endl;
	}
	stlFile.read((char *)&title, 80 * sizeof(char));
	stlFile.read((char *)&num_triangles, sizeof(num_triangles));
	cout << title << endl;
	cout << num_triangles << endl;
	stlMesh mesh(num_triangles);
	Eigen::Vector3f normal_vec, edge1, edge2;
	Eigen::Vector3f vet[4];

	for (unsigned int i = 0; i < num_triangles; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			stlFile.read((char *)&mesh[i][j][0], sizeof(float));
			stlFile.read((char *)&mesh[i][j][1], sizeof(float));
			stlFile.read((char *)&mesh[i][j][2], sizeof(float));
			mesh[i][j][0] *= 0.0254f;
			mesh[i][j][1] *= 0.0254f;
			mesh[i][j][2] *= 0.0254f;
			vet[j] << mesh[i][j][0], mesh[i][j][1], mesh[i][j][2];
		}
		edge1 = vet[2] - vet[1];
		edge2 = vet[3] - vet[1];
		vet[0] = edge1.cross(edge2) / edge1.cross(edge2).norm();
		mesh[i][0][0] = vet[0][0];
		mesh[i][0][1] = vet[0][1];
		mesh[i][0][2] = vet[0][2];
		stlFile.read((char *)&attribute, sizeof(short));
	}
	return meshToPluckerMesh(mesh);
}

void setMeshPoint(array<float,3> &point, float* value)
{
  for(int i=0; i<3; i++){
    point[i] = value[i];
  }
}

void setMeshFace(vec4x3 &face, float boundX[2], 
		 float boundY[2], float boundZ[2],
		 vector<vector<int>> faceBool)
{
  for(int i=0; i<3; i++){
    float point[3] = {boundX[faceBool[i][0]], 
		      boundY[faceBool[i][1]], 
		      boundZ[faceBool[i][2]]};

    setMeshPoint(face[i+1], point);
    // for(int j=0; j<3; j++){
    //   face[i+1][j] = point[j];
    // }
    // cout << point[0] << ", " << point[1] << ", " << point[2] << endl; 
  }
}


vector<vector<vector<int>>> getFaceIndices()
{
  vector<vector<vector<int>>> faceIndices;
  faceIndices = {{{0,0,0}, {0,0,1}, {0,1,0}},
		 {{0,0,0}, {0,0,1}, {1,0,0}},
		 {{0,0,0}, {0,1,0}, {1,0,0}}, 
		 {{0,1,1}, {0,0,1}, {0,1,0}},
		 {{0,1,1}, {0,0,1}, {1,1,1}},
		 {{0,1,1}, {0,1,0}, {1,1,1}},
		 {{1,0,1}, {0,0,1}, {1,0,0}},
		 {{1,0,1}, {0,0,1}, {1,1,1}},
		 {{1,0,1}, {1,0,0}, {1,1,1}},
		 {{1,1,0}, {0,1,0}, {1,0,0}},
		 {{1,1,0}, {0,1,0}, {1,1,1}},
		 {{1,1,0}, {1,0,0}, {1,1,1}}};

  return faceIndices;
}

/*
 * Returns a mesh of the smallest axis-aligned box 
 * that surrounds the full mesh
 */
pluckerMesh StlParser::getSurroundingBox(pluckerMesh pMesh){

  stlMesh fullMesh = pMesh.stl;
  stlMesh surroundingBox(12);
  


  float boundX[2] = {fullMesh[0][1][0], fullMesh[0][1][0]};
  float boundY[2] = {fullMesh[0][1][1], fullMesh[0][1][1]};
  float boundZ[2] = {fullMesh[0][1][2], fullMesh[0][1][2]};

  

  for(int i=0; i<fullMesh.size(); i++){
    for(int j=1; j<4; j++){
      boundX[0] = std::min(boundX[0], fullMesh[i][j][0]);
      boundX[1] = std::max(boundX[1], fullMesh[i][j][0]);
      boundY[0] = std::min(boundY[0], fullMesh[i][j][1]);
      boundY[1] = std::max(boundY[1], fullMesh[i][j][1]);
      boundZ[0] = std::min(boundZ[0], fullMesh[i][j][2]);
      boundZ[1] = std::max(boundZ[1], fullMesh[i][j][2]);
    }
  }

  int faceIndex = 0;

  vector<vector<vector<int>>> faceIndBool = getFaceIndices();


  for(vector<vector<int>> faceBool: faceIndBool){
    setMeshFace(surroundingBox[faceIndex++], boundX, boundY, boundZ,
    		faceBool);
  }

  
  // int n = 0;
  // for(vec4x3 tri: surroundingBox){

  //   cout << endl;
  //   cout << n++ << endl;
  //   for(array<float,3> point: tri){
  //     cout << endl;

  //     for(float coord: point){
  // 	cout << coord << ", ";
  //     }
  //   }

  // }

  return meshToPluckerMesh(surroundingBox);

}


pluckerMesh StlParser::meshToPluckerMesh(stlMesh sMesh){
  pluckerMesh pMesh;
  pMesh.stl = sMesh;

  for(vec4x3 stlTri:sMesh){
    array<float,3> p1 = stlTri[1];
    array<float,3> p2 = stlTri[2];
    array<float,3> p3 = stlTri[3];
    pMesh.plucker.push_back(pointsToPluckerTri(p1, p2, p3));
  }
  return pMesh;
}
