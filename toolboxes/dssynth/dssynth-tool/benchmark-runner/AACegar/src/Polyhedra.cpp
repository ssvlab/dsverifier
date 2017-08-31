//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include <streambuf>
#include <iostream>
#include <fstream>
#include <math.h>

#include <boost/timer.hpp>

#include "Polyhedra.h"
#include "VertexEnumerator.h"

namespace abstract{

using std::max;

template <class scalar>
typename Tableau<scalar>::MatrixS Polyhedra<scalar>::ms_emptyMatrix(0,0);

template <class scalar>
Eigen::JacobiSVD<typename Tableau<scalar>::MatrixS> Polyhedra<scalar>::ms_svd;

template <class scalar>
JordanMatrix<scalar>* Polyhedra<scalar>::ms_pJordan(NULL);

template <class scalar>
traceVertices_t Polyhedra<scalar>::ms_trace_vertices=eTraceNoVertex;

template <class scalar>
traceDynamics_t Polyhedra<scalar>::ms_trace_dynamics=eTraceNoDynamics;

template <class scalar>
bool Polyhedra<scalar>::ms_auto_make_vertices=true;

/// Constructs an empty buffer
template <class scalar>
Polyhedra<scalar>::Polyhedra(int dimension) :
  DualSimplex<scalar>(0,dimension),
  m_isCentralised(false),
  m_vertices(0,dimension),
  m_tag(0),
  m_loadTime(0),
  m_transformTime(0),
  m_enumerationTime(0),
  m_calculationTime(0)
{}

/// Constructs transformed polyhedra
template <class scalar>
Polyhedra<scalar>::Polyhedra(const Polyhedra &source,const MatrixS &transform,const MatrixS &inverse) :
  DualSimplex<scalar>(source.m_size,source.getDimension()),
  m_isCentralised(false),
  m_vertices(0,getDimension()),
  m_tag(0),
  m_loadTime(0),
  m_transformTime(0),
  m_enumerationTime(0),
  m_calculationTime(0)
{
  copy(source);
  if (transform.rows()>0) this->transform(transform,inverse);
}

/// Calculates the convex hull of a point cloud
template <class scalar>
bool Polyhedra<scalar>::convexHull(const MatrixS &points,const MatrixS &vectors)
{
  MatrixS supports=points*vectors;
  m_faces=vectors.transpose();
  m_supports=supports.transpose().rowwise().maxCoeff();
  return load(m_faces,m_supports);
}

/// Loads a polyhedral description from file
template <class scalar>
int Polyhedra<scalar>::loadData(const std::string &data,size_t pos,const bool vertices)
{
  size_t result=0;
  boost::timer timer;
  this->m_isNormalised=false;
  m_vertices.resize(0,m_vertices.cols());
  if ((result=data.find("cube<",pos)) >= 0) {
    int dimension=getDimension();
    m_faces.resize(2*dimension,dimension);
    m_faces.block(0,0,dimension,dimension)=MatrixS::Identity(dimension,dimension);
    m_faces.block(dimension,0,dimension,dimension)=-MatrixS::Identity(dimension,dimension);
    m_supports.resize(m_faces.rows(),1);
    result+=5;
    m_supports.coeffRef(0,0)=ms_logger.getNumber(data,result);
    for (int row=1;row<m_faces.rows();row++) {
      m_supports.coeffRef(row,0)=m_supports.coeff(row-1,0);
    }
    load(m_faces,m_supports);
    m_loadTime=timer.elapsed()*1000;
    return result;
  }
  int lines=MatToStr<scalar>::ms_defaultLogger.lines(data,pos);
  if (lines==0) {
    result=ms_logger.StringToMat(m_faces,data,pos);
    clear();
    return result;
  }
  if (vertices) {
    result=ms_logger.StringToMat(m_vertices,data,pos);
    if (result>0) makeFaces();
    m_loadTime=timer.elapsed()*1000;
    return result;
  }
  if (getDimension()==0) {
    int cols=MatToStr<scalar>::ms_defaultLogger.cols(data,pos);
    changeDimension(cols);
  }
  m_supports.resize(lines,1);
  m_faces.resize(lines,getDimension());
  result=ms_logger.StringToMat(m_faces,m_supports,data,pos);
  if (result>0) {
    load(m_faces,m_supports);
    m_loadTime=timer.elapsed()*1000;
    if (ms_trace_tableau>=eTraceTableau) logTableau("loaded ");
    if (ms_auto_make_vertices && (m_faces.rows()>m_dimension)) makeVertices();
  }
  return result;
}

/// Loads a polyhedral description
template <class scalar>
bool Polyhedra<scalar>::loadVertices(const MatrixS &vertices)
{
  changeDimension(vertices.cols());
  m_vertices.conservativeResize(vertices.rows(),vertices.cols());
  m_vertices.block(0,0,vertices.rows(),vertices.cols())=vertices;
  if (makeFaces()) return true;
  if (ms_trace_errors) ms_logger.logData("Failed to load vertices");
  return false;
}

/// Changes the dimensionality of the polyhedra
template <class scalar>
void Polyhedra<scalar>::changeDimension(const int dimension,const bool keep)
{
  int cols=m_faces.cols();
  int rows=keep ? m_faces.rows() : 0;
  bool extend=keep && (dimension>cols);
  int newRows=extend ? 2*(dimension-cols) : 0;
  if (dimension!=cols) {
    m_faces.conservativeResize(rows+newRows,dimension);
    m_supports.conservativeResize(rows+newRows,1);
    if (extend) {
      m_faces.block(0,cols,rows,dimension-cols)=MatrixS::Zero(rows,dimension-cols);
      m_faces.block(rows,0,newRows,cols)=MatrixS::Zero(newRows,cols);
      m_faces.block(rows,cols,dimension-cols,dimension-cols)=MatrixS::Identity(dimension-cols,dimension-cols);
      m_faces.block(rows+dimension-cols,cols,dimension-cols,dimension-cols)=-MatrixS::Identity(dimension-cols,dimension-cols);
      m_supports.block(rows,0,newRows,1)=MatrixS::Zero(newRows,1);
    }
    load(m_faces,m_supports);
  }
}

/// Loads a polyhedral description from file
template <class scalar>
bool Polyhedra<scalar>::loadFromFile(const std::string &fileName)
{
  std::stringstream buffer;
  std::ifstream file;
  file.open(fileName.data());
  if (!file.is_open()) return false;
  buffer << file.rdbuf();
  file.close();
  std::string str=buffer.str();
  return loadData(str);
}

/// Returns a description of the polyhedra
template <class scalar>
std::string Polyhedra<scalar>::getDescription(displayType_t displayType,bool interval,bool useBrackets,MatrixS& templates)
{
  if (displayType==eVertices) return getVertices(",",interval,useBrackets);
  return getFaces(displayType==eNormalised,interval,useBrackets,templates);
}

/// Returns a description of the polyhedra
template <class scalar>
std::string Polyhedra<scalar>::getFaces(bool normalised,bool interval,bool useBrackets,MatrixS& templates)
{
  std::string result;
  MatrixS faces=m_faces;
  MatrixS supports=m_supports;
  if (templates.cols()>0) {
    faces=templates.transpose();
    maximiseAll(templates,supports);
  }
  if (supports.rows()!=faces.rows()) {
    int row=supports.rows();
    supports.conservativeResize(faces.cols(),1);
    for (;row<supports.rows();row++) supports.coeffRef(row,0)=func::ms_nan;
  }

  if (useBrackets) result+=ms_logger.IneToString(faces,supports,interval,normalised);
  else          result+=ms_decoder.IneToString(faces,supports,interval,normalised);
  if (m_isCentralised) {
    result+="\nc=";
    if (useBrackets)   result+=ms_logger.MatToString(m_centre,interval);
    else            result+=ms_decoder.MatToString(m_centre,interval);
  }
  return result;
}

/// Returns a description of the vertices of the polyhedra
template <class scalar>
std::string Polyhedra<scalar>::getVertices(std::string separator,bool interval,bool useBrackets)
{
  makeVertices();
  std::string result=ms_decoder.MatToString(m_vertices,interval);
  if (m_isCentralised) {
    result+="\nc=";
    result+=ms_decoder.MatToString(m_centre,interval);
  }
  return result;
}

/// Copies the polyhedra from another source
template <class scalar>
bool Polyhedra<scalar>::copy(const Polyhedra &source)
{
  if (!load(source.m_faces,source.m_supports)) return false;
  this->m_isNormalised=source.m_isNormalised;
  m_centre=source.m_centre;
  m_isCentralised=source.m_isCentralised;
  m_vertices.resize(source.m_vertices.rows(),source.m_vertices.cols());
  m_vertices.block(0,0,m_vertices.rows(),m_vertices.cols())=source.m_vertices;
  return true;
}

template <class scalar>
bool Polyhedra<scalar>::load(const MatrixS &faces,const MatrixS &supports,const bool transpose)
{
  boost::timer timer;
  this->m_dimension=faces.cols()+1;
  m_vertices.resize(0,getDimension());
  if (!DualSimplex<scalar>::load(faces,supports,transpose)) return false;
  if (&faces!=&m_faces) {
    m_centre=MatrixS::Zero(1,getDimension());
    m_isCentralised=false;
  }
  m_loadTime=timer.elapsed()*1000;
  return true;
}

/// Indicates if the referenced polyhedra is contained inside this one
template <class scalar>
bool Polyhedra<scalar>::contains(Polyhedra<scalar>& polyhedra)
{
  MatrixS supports(m_faces.rows(),1);
  MatrixS matrix=m_faces.transpose();
  polyhedra.maximiseAll(matrix,supports,this->eUnderAprox);
  supports-=m_supports;
  for (int i=0;i<supports.rows();i++) {
    if (func::isPositive(supports.coeff(i,0))) {
      return false;
    }
  }
  return true;
}

/// Intersects the polyhedra with another polyhedra
template <class scalar>
bool Polyhedra<scalar>::pseudoIntersect(const Polyhedra &polyhedra)
{
  if (polyhedra.m_dimension!=this->m_dimension) return false;
  this->m_isNormalised=false;
  int count=m_supports.rows();
  int remoteCount=polyhedra.m_supports.rows();
  m_faces.conservativeResize(count+remoteCount,m_faces.cols());
  m_faces.block(count,0,remoteCount,m_faces.cols())=polyhedra.m_faces;
  m_supports.conservativeResize(count+remoteCount,1);
  m_supports.block(count,0,remoteCount,1)=polyhedra.m_supports;
  load(m_faces,m_supports);
  return true;
}

/// Calculates the pseudoinverse of a matrix
template <class scalar>
typename Tableau<scalar>::MatrixS Polyhedra<scalar>::pseudoInverseEigen(const MatrixS &matrix,bool &hasInverse)
{
  if (matrix.rows()!=matrix.cols()) return pseudoInverseSVD(matrix,hasInverse);
  // We only care about the directions, so errors are acceptable (they only cause wrapping
  MatrixR refMatrix(matrix.rows(),matrix.cols());
  for (int row=0;row<matrix.rows();row++) {
    for (int col=0;col<matrix.cols();col++) {
      refMatrix.coeffRef(row,col)=func::toCentre(matrix.coeff(row,col));
    }
  }
  Eigen::EigenSolver<MatrixR> eigenSpace(refMatrix);
  if (eigenSpace.info()!=Eigen::Success) return this->getJordanSolver()->getSVDpseudoInverse(matrix,hasInverse);
  MatrixRC eigenvalues=eigenSpace.eigenvalues().asDiagonal();
  for (int row=0;row<matrix.rows();row++)
  {
    if (func::isZero(func::norm2(eigenvalues.coeff(row,row)),func::ms_weakZero)) hasInverse=false;
    else eigenvalues.coeffRef(row,row)=refScalar(1)/eigenvalues.coeff(row,row);
  }
  if (hasInverse) return matrix.inverse();
  MatrixRC eigenVectors=eigenSpace.eigenvectors();
  eigenvalues=eigenVectors.inverse()*eigenvalues*eigenVectors;
  MatrixS result(matrix.rows(),matrix.cols());
  for (int row=0;row<matrix.rows();row++) {
    for (int col=0;col<matrix.cols();col++) {
      result.coeffRef(row,col)=eigenvalues.coeff(row,col).real();
    }
  }
  if (ms_trace_dynamics>=eTraceDynamics) {
    ms_logger.logData(matrix,"Matrix:");
    ms_logger.logData(result,"Inverse:");
  }
  return result;
}

/// Calculates the pseudoinverse of a matrix
template <class scalar>
typename Tableau<scalar>::MatrixS Polyhedra<scalar>::pseudoInverseSVD(const MatrixS &matrix,bool &hasInverse)
{
  // We only care about the directions, so errors are acceptable (they only cause wrapping
  MatrixR refMatrix(matrix.rows(),matrix.cols());
  for (int row=0;row<matrix.rows();row++) {
    for (int col=0;col<matrix.cols();col++) {
      refMatrix.coeffRef(row,col)=func::toCentre(matrix.coeff(row,col));
    }
  }
  return this->getJordanSolver()->getSVDpseudoInverse(matrix,hasInverse);
}

/// Calculates the pseudoinverse of a matrix
template <class scalar>
typename Tableau<scalar>::MatrixS Polyhedra<scalar>::pseudoInverseJordan(const MatrixS &matrix,bool &hasInverse)
{
  return this->getJordanSolver()->getPseudoInverse(matrix,hasInverse);
}

/// Intersects the polyhedra with another polyhedra
template <class scalar>
bool Polyhedra<scalar>::intersect(const Polyhedra &polyhedra,const bool over)
{
  if (polyhedra.isEmpty()) return true;
  if (pseudoIntersect(polyhedra)) {
    MatrixS supports(m_faces.rows(),1);
    bool redundant[supports.rows()];
    MatrixS matrix=m_faces.transpose();
    maximiseAll(matrix,supports);
    for (int i=0;i<supports.rows();i++) {
      scalar dif=m_supports.coeff(i,0)-supports.coeff(i,0);
      char sign=func::hardSign(dif);
      redundant[i]=(sign>0) || (over && (sign==0));
    }
    int pos=0;
    for (int i=0;i<supports.rows();i++) {
      if (!redundant[i]) {
        m_faces.row(pos)=m_faces.row(i);
        m_supports.coeffRef(pos++,0)=m_supports.coeff(i,0);
      }
    }
    m_faces.conservativeResize(pos,m_faces.cols());
    m_supports.conservativeResize(pos,1);
    load(m_faces,m_supports);
    return true;
  }
  return false;
}

/// Performs the union of another polyhedra with this one
template <class scalar>
bool Polyhedra<scalar>::merge(Polyhedra &polyhedra,const bool extend)
{
  boost::timer timer;
  if (polyhedra.isEmpty()) return true;
  MatrixS supports2(m_faces.rows(),1);
  MatrixS faceVectors2=m_faces.transpose();
  polyhedra.maximiseAll(faceVectors2,supports2);
  if (this->ms_trace_tableau>=eTraceTransforms) {
    ms_logger.logData(m_faces,m_supports,"Orig Set");
    ms_logger.logData(m_faces,supports2,"Merge Set");
  }
  if (extend) {
    MatrixS supports(polyhedra.m_faces.rows(),1);
    MatrixS faceVectors=polyhedra.m_faces.transpose();
    maximiseAll(faceVectors,supports);
    if (!pseudoIntersect(polyhedra)) return false;
    for (int i=0;i<supports2.rows();i++) m_supports.coeffRef(i,0)=max(m_supports.coeff(i,0),supports2.coeff(i,0));
    for (int i=0;i<supports.rows();i++) m_supports.coeffRef(i+supports2.rows(),0)=max(polyhedra.m_supports.coeff(i,0),supports.coeff(i,0));
    this->removeRedundancies();
  }
  else {
    for (int i=0;i<supports2.rows();i++) m_supports.coeffRef(i,0)=max(m_supports.coeff(i,0),supports2.coeff(i,0));
  }
  if (this->ms_trace_tableau>=eTraceTransforms) {
    ms_logger.logData(m_faces,m_supports,"Merged Set");
  }
  if (this->ms_trace_time) {
    int elapsed=timer.elapsed()*1000;
    ms_logger.logData(elapsed," Merge time",true);
  }
  return true;
}

template <class scalar>
bool Polyhedra<scalar>::concatenate(Polyhedra &polyhedra)
{
  return concatenate(polyhedra.m_faces,polyhedra.m_supports);
}

template <class scalar>
bool Polyhedra<scalar>::concatenate(MatrixS &faces,MatrixS &supports)
{
  int oldRows=m_faces.rows();
  int oldCols=m_faces.cols();
  m_faces.conservativeResize(oldRows+faces.rows(),oldCols+faces.cols());
  m_supports.conservativeResize(oldRows+supports.rows(),1);
  m_faces.block(0,oldCols,oldRows,faces.cols())=MatrixS::Zero(oldRows,faces.cols());
  m_faces.block(oldRows,0,faces.rows(),oldCols)=MatrixS::Zero(faces.rows(),oldCols);
  m_faces.block(oldRows,oldCols,faces.rows(),faces.cols())=faces;
  m_supports.block(oldRows,0,supports.rows(),1)=supports;
  load(m_faces,m_supports);
  return true;
}

/// Performs the Minkowski sum of this polyhedra to another
template <class scalar>
bool Polyhedra<scalar>::add(Polyhedra &polyhedra,const bool extended)
{
  if (polyhedra.isEmpty()) return true;
  MatrixS supports2(m_faces.rows(),1);
  MatrixS faceVectors2=m_faces.transpose();
  polyhedra.maximiseAll(faceVectors2,supports2);
  if (!extended) {
    for (int i=0;i<supports2.rows();i++) {
      m_supports.coeffRef(i,0)=m_supports.coeff(i,0)+supports2.coeff(i,0);
    }
    return true;
  }
  MatrixS supports(polyhedra.m_faces.rows(),1);
  MatrixS faceVectors=polyhedra.m_faces.transpose();
  maximiseAll(faceVectors,supports);
  if (!pseudoIntersect(polyhedra)) return false;
  for (int i=0;i<supports2.rows();i++) m_supports.coeffRef(i,0)=m_supports.coeff(i,0)+supports2.coeff(i,0);
  int j=0;
  for (int i=supports2.rows();i<m_supports.rows();i++,j++) m_supports.coeffRef(i,0)=m_supports.coeff(i,0)+supports.coeff(j,0);
  this->removeRedundancies();
  return true;
}

/// Performs the Minkowski difference of this polyhedra with another
template <class scalar>
bool Polyhedra<scalar>::erode(Polyhedra &polyhedra)
{
  MatrixS supports(polyhedra.m_faces.rows(),1);
  MatrixS supports2(m_faces.rows(),1);
  MatrixS faceVectors=polyhedra.m_faces.transpose();
  maximiseAll(faceVectors,supports);
  MatrixS faceVectors2=m_faces.transpose();
  polyhedra.maximiseAll(faceVectors2,supports2);
  if (!pseudoIntersect(polyhedra)) return false;
  for (int i=0;i<supports2.rows();i++) m_supports.coeffRef(i,0)=m_supports.coeff(i,0)-supports2.coeff(i,0);
  int j=0;
  for (int i=supports2.rows();i<m_supports.rows();i++,j++) m_supports.coeffRef(i,0)=m_supports.coeff(i,0)-supports.coeff(j,0);
  this->removeRedundancies();
  return true;
}

/// Adds a number of directions to the template of the polhedra
template <class scalar>
bool Polyhedra<scalar>::addDirection(const MatrixS &directions)
{
  MatrixS supports(directions.cols(),1);
  if (directions.rows()!=getDimension()) return false;
  maximiseAll(directions,supports);
  return addDirection(directions,supports);
}

/// Adds a number of directions to the template of the polhedra
template <class scalar>
bool Polyhedra<scalar>::addDirection(const MatrixS &directions,MatrixS &supports)
{
  if (directions.rows()!=getDimension()) return false;
  this->m_isNormalised=false;
  int count=m_faces.rows();
  m_faces.conservativeResize(count+directions.cols(),getDimension());
  m_faces.block(count,0,directions.cols(),m_faces.cols())=directions.transpose();
  m_supports.conservativeResize(count+directions.cols(),1);
  m_supports.block(count,0,directions.cols(),1)=supports;  
  this->removeRedundancies();
  m_centre.resize(0,m_centre.cols());
  m_vertices.resize(0,m_vertices.cols());
  return true;
}

/// Retrieves a copy of this polyhedra transformed by the given matrix
template <class scalar>
Polyhedra<scalar>& Polyhedra<scalar>::getTransformedPolyhedra(Polyhedra& polyhedra,const MatrixS& transform,const MatrixS& inverse,const MatrixS &templates)
{
  if (!polyhedra.copy(*this)) {
    ms_logger.logData(m_name,false);
    ms_logger.logData(" loading Error");
    throw loadError;
  }
  if ((transform.rows()>0) || (inverse.rows()>0)) polyhedra.transform(transform,inverse);
  if (templates.cols()>0) polyhedra.retemplate(templates);
  return polyhedra;
}

/// linearly transofrm the polyhedra (rotate, translate, stretch)
template <class scalar>
bool Polyhedra<scalar>::transform(const MatrixS &transform,const MatrixS& inverse)
{
  bool hasInverse=true;
  if (this->isEmpty()) return true;
  boost::timer timer;
  if ((transform.rows()<=0) && (inverse.rows()>0)) {
    this->m_isNormalised=false;
    if (m_faces.cols()==inverse.rows()) m_faces*=inverse;
    else m_faces*=inverse.transpose();
    m_centre.resize(0,m_centre.cols());
    m_vertices.resize(0,m_vertices.cols());
  }
  else {
    if (transform.cols()!=getDimension()) return false;
    if (transform.rows()>transform.cols()) {
      int rows=transform.rows();
      int cols=transform.cols();
      MatrixS newTransform(rows,rows);
      newTransform.block(0,0,rows,cols)=transform;
      newTransform.block(0,cols,rows,rows-cols)=MatrixS::Zero(rows,rows-cols);
      changeDimension(rows,true);
      return this->transform(newTransform);
    }
    if (m_centre.rows()>0) {
      if (m_centre.cols()==transform.cols()) {
        m_centre=m_centre*transform.transpose();
        if (m_isCentralised) m_isCentralised=(func::isPositive(m_centre.norm()));
      }
    }
    if (m_vertices.rows()>0) m_vertices*=transform.transpose();
    this->m_isNormalised=false;
    if (inverse.rows()>0) m_faces*=inverse;
    else {
      MatrixS matrix=pseudoInverseEigen(transform,hasInverse);
      if (ms_trace_dynamics>=eTraceDynamics) {
        ms_logger.logData(transform,"Transform:");
        ms_logger.logData(matrix,"Pseudo Inverse:");
      }
      if (hasInverse) m_faces*=matrix;// The normal gets multiplied by A^-1^T
      else {
        MatrixS faces(2*m_faces.rows(),getDimension());
        faces.block(0,0,m_faces.rows(),getDimension())=m_faces;
        faces.block(m_faces.rows(),0,m_faces.rows(),getDimension())=m_faces*matrix;
        MatrixS supports;
        MatrixS vectors=transform.transpose()*faces.transpose();
        if (this->ms_trace_tableau>=eTraceTableau) this->logTableau();
        maximiseAll(vectors,supports);
        m_faces=faces;
        m_supports=supports;
        return this->removeRedundancies();
      }
    }
  }
  bool result=DualSimplex<scalar>::load(m_faces,m_supports);
  m_transformTime=timer.elapsed()*1000;
  if (this->ms_trace_time) {
    if (m_transformTime>1) {
      ms_logger.logData(m_name,false);
      ms_logger.logData(m_transformTime," Transform:",true);
    }
  }
  return result;
}

/// templates the polyhedra in the given directions
template <class scalar>
bool Polyhedra<scalar>::retemplate(const MatrixS& templates,refScalar aprox)
{
  MatrixS supports(templates.cols(),1);
  decentralize();
  this->m_isNormalised=false;
  if (aprox<0) {
    refScalar threshold=1+aprox;
    this->removeRedundancies();
    if (!makeVertices()) return false;
    for (int dir=0;dir<templates.cols();dir++) {
      scalar support=-func::ms_infinity;
      supports.coeffRef(dir,0)=support;
      for (int point=0;point<m_vertices.rows();point++) {
        MatrixS thisSupport=m_vertices.row(point)*templates.block(0,dir,m_vertices.cols(),1);
        if (thisSupport.coeff(0,0)>support) support=thisSupport.coeff(0,0);
      }
      supports.coeffRef(dir,0)=support;
      for (int point=0;point<m_vertices.rows();point++) {
        MatrixS thisSupport=m_vertices.row(point)*templates.block(0,dir,m_vertices.cols(),1);
        if ((thisSupport.coeff(0,0)>support*threshold) && (thisSupport.coeff(0,0)<supports.coeffRef(dir,0)))
        {
          supports.coeffRef(dir,0)=thisSupport.coeff(0,0);
        }
      }
    }
    addDirection(templates,supports);
    logTableau();//templog
    this->removeRedundancies();
    //m_vertices.resize(0,m_vertices.cols());
    makeVertices(true);
    if (!makeFaces()) return false;
    logTableau();//templog
    for (int i=0;i<supports.rows();i++) supports.coeffRef(i,0)=func::toLower(supports.coeffRef(i,0));
    return true;
  }
  maximiseAll(templates,supports);
  m_faces=templates.transpose();
  m_supports=supports;
  return load(m_faces,m_supports);
}

/// linearly transofrm the polyhedra (rotate, translate, stretch)
template <class scalar>
void Polyhedra<scalar>::transform(const scalar &coefficient)
{
  this->m_isNormalised=false;
  decentralize();
  if (m_vertices.rows()>0) m_vertices*=coefficient;
  m_supports*=coefficient;
  m_centre*=coefficient;
  m_isCentralised=(m_centre.norm()>this->m_zero);
  load(m_faces,m_supports);
}

/// linearly transofrm the polyhedra through vertex enumeration
template <class scalar>
bool Polyhedra<scalar>::vertexTransform(const MatrixS &transform,const MatrixS &templates)
{
  if (!makeVertices()) return false;
  MatrixS vertices=m_vertices*transform.transpose();
  return convexHull(vertices,templates);
}

/// finds the inequalities of the polyhedra and stores them in a matrix
template <class scalar>
bool Polyhedra<scalar>::makeFaces()
{
  if (m_vertices.rows()<=0) return true;
  decentralize();
  MatrixS supports=MatrixS::Ones(m_vertices.rows(),1);
  MatrixS centre=m_vertices.colwise().sum()/m_vertices.rows();
  for (int row=0;row<m_vertices.rows();row++) m_vertices.row(row)-=centre;
  VertexEnumerator<scalar> enumerator(m_vertices.rows(),m_vertices.cols());
  typename VertexEnumerator<scalar>::RayList& rayList=enumerator.findVertices(m_vertices,supports,true);
  int numFaces=rayList.size();
  if (numFaces==0) return false;
  m_faces.resize(numFaces,getDimension());
  m_supports.resize(numFaces,1);
  int row=0;
  for (typename VertexEnumerator<scalar>::RayList::iterator it=rayList.begin();it!=rayList.end();it++,row++) {
    m_faces.row(row)=it->data.block(0,1,1,m_faces.cols());
    m_supports.coeffRef(row,0)=it->data.coeff(0,0);
  }
  centre=-centre;
  m_supports-=m_faces*centre.transpose();
  load(m_faces,m_supports);
  return true;
}

template <class scalar>
void Polyhedra<scalar>::logVertices(bool force)
{
  if ((ms_trace_vertices>=eTraceVertices) || force) {
    makeVertices(force);
    ms_logger.logData(m_name,false);
    ms_logger.logData(m_vertices," Vertices:");
  }
}

template <class scalar>
void Polyhedra<scalar>::logPolyhedra(std::string parameters)
{
  if (parameters.length()>0) {
    std::stringstream stream;
    stream << getName();
    stream << ": " << parameters;
    ms_logger.logData(m_faces,m_supports,stream.str());
  }
  else ms_logger.logData(m_faces,m_supports,getName());
  if (ms_trace_vertices>=eTraceVertices) {
    makeVertices();
    ms_logger.logData(m_vertices," Vertices:");
  }
}

/// finds the vertices of the polyhedra and stores them in a matrix
template <class scalar>
bool Polyhedra<scalar>::makeVertices(bool force)
{
  if (!force && (m_vertices.rows()>0)) return true;
  if (m_faces.rows()==0) {
    return false;
  }
  boost::timer timer;
  this->normalise();
  VertexEnumerator<scalar> enumerator(m_faces.rows(),m_faces.cols());
  typename VertexEnumerator<scalar>::RayList& rayList=enumerator.findVertices(m_faces,m_supports,false);
  enumerator.logRays();

  int numVertices=rayList.size();
  if (numVertices==0) {
    if (this->ms_trace_time) {
      ms_logger.logData(m_name,false);
      ms_logger.logData(timer.elapsed()*1000," Make Vertices Time:",true);
    }
    if (ms_trace_errors) {
      ms_logger.logData(m_name, false);
      ms_logger.logData(": Failed to make vertices");
      logTableau();
    }
    return false;
  }
  m_vertices.resize(numVertices,getDimension());

  int row=0;
  for (typename VertexEnumerator<scalar>::RayList::iterator it=rayList.begin();it!=rayList.end();it++,row++) {
    scalar scale = abs(it->data.coeff(0,0));
    if (func::isZero(scale)) scale=1;
    for (int col=0;col<m_vertices.cols();col++) m_vertices.coeffRef(row,col)=it->data.coeff(0,col+1)/scale;
  }
  m_enumerationTime=timer.elapsed()*1000;
  if (ms_trace_vertices>=eTraceVertices) {
    ms_logger.logData(m_name,false);
    ms_logger.logData(" Find Vertices:");
    this->logTableau();
  }
  if (this->ms_trace_time) {
    ms_logger.logData(m_name,false);
    ms_logger.logData(m_enumerationTime," Make Vertices Time:",true);
  }
  return true;
}

/// Removes any existing faces
template <class scalar>
void Polyhedra<scalar>::clear()
{
  m_faces.resize(0,getDimension());
  m_supports.resize(0,1);
  load(m_faces,m_supports);
}

/// Retrieves the (hyper-rectangular) center of the polyhedra
template <class scalar>
typename Tableau<scalar>::MatrixS& Polyhedra<scalar>::getCentre()
{
  if (!m_isCentralised) {
    MatrixS positiveDirections=MatrixS::Identity(getDimension(),getDimension());
    MatrixS positiveSupports;
    maximiseAll(positiveDirections,positiveSupports);
    MatrixS negativeDirections=-positiveDirections;
    MatrixS negativeSupports;
    maximiseAll(negativeDirections,negativeSupports);
    m_centre=(positiveSupports-negativeSupports).transpose()/2;
  }
  return m_centre;
}

/// Finds the (hyper-rectangular) center of the polyhedra, and moves it to the origin
template <class scalar>
void Polyhedra<scalar>::centralize()
{
  if (m_isCentralised) return;
  m_isCentralised=false;
  translate(getCentre());
}

/// Translates the polyhedra in the direction of vector
template <class scalar>
void Polyhedra<scalar>::translate(const MatrixS &vector,const bool storeOffset)
{
  if (storeOffset) {
    if (m_isCentralised)  m_centre+=vector;
    else                  m_centre=vector;
    m_isCentralised=(func::toLower(m_centre.norm())>this->m_zero);
  }
  for (int row=0;row<m_vertices.rows();row++) m_vertices.row(row)-=vector;
  MatrixS matrix=m_faces*vector.transpose();
  m_supports-=matrix;
}

/// Moves the polyhedra by the vector indicated in centre.
template <class scalar>
void Polyhedra<scalar>::decentralize()
{
  if (!m_isCentralised) return;
  MatrixS minCentre=-m_centre;
  translate(minCentre);
  m_isCentralised=false;
}

/// Indicates if there is a selected non-zero central point
template <class scalar>
bool Polyhedra<scalar>::isCentralized()
{
  return m_isCentralised;//(m_centre.norm()>this->m_zero);
}

/// Indicates if a set of points is inside the Polyhedra
template <class scalar>
bool Polyhedra<scalar>::isInside(const MatrixS &points)
{
  if (points.cols()!=getDimension()) return false;
  MatrixS supports=m_faces*points.transpose();
  for (int col=0;col<supports.cols();col++) {
    supports.col(col)-=m_supports;
    for (int row=0;row<supports.rows();row++) {
      if (func::toUpper(supports.coeff(row,col))>0) {
        /*ms_logger.logData(points,"Points");
        ms_logger.logData(supports,"Supports");
        logTableau("True Supports",true);*/
        return false;
      }
    }
  }
  return true;
}


template <class scalar>
void Polyhedra<scalar>::ComputeVertexOrderVector()
{
  m_vertices.QuickAngleSort();
}

template <class scalar>
typename Tableau<scalar>::MatrixS Polyhedra<scalar>::vertexMaximize(const MatrixS &vectors,const bool all)
{
  if (!makeVertices()) return MatrixS(0,0);
  if (m_vertices.cols()!=vectors.cols()) return MatrixS::Zero(vectors.cols(),1);
  MatrixS result=(m_vertices*vectors).transpose();
  if (all) return result;
  MatrixS supports=result.rowwise().maxCoeff();
  return supports;
}

template <class scalar>
typename Tableau<scalar>::MatrixS Polyhedra<scalar>::boundingHyperBox()
{
  int dimension=getDimension();
  MatrixS hyperbox(2*dimension,2);
  makeVertices();
  for (int i=0;i<dimension;i++) {
    hyperbox.coeffRef(i,0)=m_vertices.coeff(0,i);
    hyperbox.coeffRef(i,1)=m_vertices.coeff(0,i);
  }
  for  (int row=1;row<m_vertices.rows();row++) {
    for (int i=0;i<dimension;i++) {
      if (m_vertices.coeff(row,i)<hyperbox.coeff(i,0)) hyperbox.coeffRef(i,0)=m_vertices.coeff(0,i);
      if (m_vertices.coeff(row,i)>hyperbox.coeff(i,1)) hyperbox.coeffRef(i,1)=m_vertices.coeff(0,i);
    }
  }
  return hyperbox;
}

template <class scalar>
JordanMatrix<scalar>* Polyhedra<scalar>::getJordanSolver()
{
  if (!ms_pJordan) ms_pJordan=new JordanMatrix<scalar>(0);
  return ms_pJordan;
}

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class Polyhedra<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class Polyhedra<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class Polyhedra<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class Polyhedra<mpinterval>;
  #endif
#endif

}
