//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include "EigenPolyhedra.h"
#include <Eigen/Eigenvalues>

namespace abstract{

/// Constructs an empty buffer
template <class scalar>
EigenPolyhedra<scalar>::EigenPolyhedra(const std::string name,const int dimension) :
    m_name(name),
    m_dimension(dimension),
    m_polyhedra(dimension),
    m_eigenPolyhedra(dimension),
    m_singularPolyhedra(dimension),
    m_pEigenVectors(&Polyhedra<scalar>::ms_emptyMatrix),
    m_pInverseEigenVectors(&Polyhedra<scalar>::ms_emptyMatrix)
{
  m_polyhedra.setName(name);
  m_eigenPolyhedra.setName(name+"-eigen");
  m_singularPolyhedra.setName(name+"-singular");
}

/// Constructs an empty buffer
template <class scalar>
EigenPolyhedra<scalar>::EigenPolyhedra(const std::string name,const int dimension,AbstractPolyhedra<scalar> &source,MatrixS &transform,MatrixS &inverse) :
    m_name(name),
    m_dimension(dimension),
    m_polyhedra(source,transform,inverse),
    m_eigenPolyhedra(dimension),
    m_singularPolyhedra(dimension),
    m_pEigenVectors(&transform),
    m_pInverseEigenVectors(&inverse)
{
  m_polyhedra.setName(name);
  m_eigenPolyhedra.setName(name+"-eigen");
  m_singularPolyhedra.setName(name+"-singular");
  if (transform.rows()>0) getEigenPolyhedra();
}

template <class scalar>
EigenPolyhedra<scalar>::~EigenPolyhedra()
{
}

template <class scalar>
void EigenPolyhedra<scalar>::setName(const std::string name)
{
  m_name=name;
  m_polyhedra.setName(name);
  m_eigenPolyhedra.setName(name+"-eigen");
  m_singularPolyhedra.setName(name+"-singular");
}

/// Changes the dimensionality of the polyhedra
template <class scalar>
void EigenPolyhedra<scalar>::changeDimension(const int dimension)
{
  m_polyhedra.changeDimension(dimension);
  m_eigenPolyhedra.changeDimension(dimension);
  m_singularPolyhedra.changeDimension(dimension);
}

/// Loads a full dynamic description
template <class scalar>
int EigenPolyhedra<scalar>::load(const std::string &data,size_t pos,const bool vertices)
{
  int result=m_polyhedra.loadData(data,pos,vertices);
  m_eigenPolyhedra.clear();
  m_singularPolyhedra.clear();
  if (result>0) getEigenPolyhedra();
  return result;
}

/// Loads a known polyhedra
template <class scalar>
AbstractPolyhedra<scalar>& EigenPolyhedra<scalar>::load(AbstractPolyhedra<scalar> &source,const MatrixS &transform,const MatrixS &inverse,space_t space)
{
  if (space==eNormalSpace) {
    source.getTransformedPolyhedra(m_polyhedra,transform,inverse);
    m_polyhedra.getTransformedPolyhedra(m_eigenPolyhedra,*m_pInverseEigenVectors,*m_pEigenVectors);
    m_singularPolyhedra.clear();
  }
  else {
    source.getTransformedPolyhedra(m_eigenPolyhedra,transform,inverse);
    m_eigenPolyhedra.getTransformedPolyhedra(m_polyhedra,*m_pEigenVectors,*m_pInverseEigenVectors);
    return m_eigenPolyhedra;
  }
  return m_polyhedra;
}

/// Loads a system description from file
template <class scalar>
bool EigenPolyhedra<scalar>::loadFromFile(const std::string &fileName)
{
  if (!m_polyhedra.loadFromFile(fileName)) return false;
  m_eigenPolyhedra.clear();
  m_singularPolyhedra.clear();
  getEigenPolyhedra();
  return true;
}

/// Loads a system description from file
template <class scalar>
AbstractPolyhedra<scalar>& EigenPolyhedra<scalar>::load(MatrixS &faces,MatrixS &supports,space_t space)
{
  m_singularPolyhedra.clear();
  if (space==eNormalSpace) {
    m_polyhedra.load(faces,supports);
    getEigenPolyhedra();
    return m_polyhedra;
  }
  else {
    m_eigenPolyhedra.load(faces,supports);
    m_eigenPolyhedra.getTransformedPolyhedra(m_polyhedra,*m_pEigenVectors,*m_pInverseEigenVectors);
  }
  return getPolyhedra(space);
}

/// Loads a system description from file
template <class scalar>
AbstractPolyhedra<scalar>& EigenPolyhedra<scalar>::mergeLoad(AbstractPolyhedra<scalar> &source,MatrixS &faces,MatrixS &supports,space_t space)
{
  m_singularPolyhedra.clear();
  if (space==eNormalSpace) {
    m_polyhedra.load(faces,supports);
    m_polyhedra.merge(source,false);
    getEigenPolyhedra();
    return m_polyhedra;
  }
  else {
    m_eigenPolyhedra.load(faces,supports);
    m_eigenPolyhedra.merge(source,false);
    m_eigenPolyhedra.getTransformedPolyhedra(m_polyhedra,*m_pEigenVectors,*m_pInverseEigenVectors);
  }
  return getPolyhedra(space);
}

template <class scalar>
void EigenPolyhedra<scalar>::clear()
{
  m_polyhedra.clear();
  m_eigenPolyhedra.clear();
  m_singularPolyhedra.clear();
}

/// Retrieves a copy of this polyhedra transformed by the given matrix
template <class scalar>
AbstractPolyhedra<scalar>& EigenPolyhedra<scalar>::getPolyhedra(space_t space,bool force)
{
  if (space==eNormalSpace) return m_polyhedra;
  return (space==eEigenSpace) ? getEigenPolyhedra(force) : m_singularPolyhedra;
}

template <class scalar>
AbstractPolyhedra<scalar>& EigenPolyhedra<scalar>::getPolyhedra(space_t space,const std::vector<int> &rotations,const std::vector<int> &dilations)
{
  if (space==eNormalSpace) return m_polyhedra;
  return (space==eEigenSpace) ? getEigenPolyhedra() : getSingularPolyhedra(rotations,dilations);
}

/// Retrieves a copy of this polyhedra transformed by the given matrix
template <class scalar>
AbstractPolyhedra<scalar>& EigenPolyhedra<scalar>::getTransformedPolyhedra(Polyhedra<scalar>& polyhedra,space_t space,const MatrixS& transform,const MatrixS& inverse,const MatrixS &templates)
{
  return getPolyhedra(space).getTransformedPolyhedra(polyhedra,transform,inverse,templates);
}

/// Sets the eigenvector transformation matrix
template <class scalar>
void EigenPolyhedra<scalar>::setEigenSpace(MatrixS &transform,MatrixS &inverse)
{
  m_pEigenVectors=&transform;
  m_pInverseEigenVectors=&inverse;
  m_eigenPolyhedra.clear();
  m_polyhedra.getTransformedPolyhedra(m_eigenPolyhedra,*m_pInverseEigenVectors,*m_pEigenVectors);
  m_singularPolyhedra.clear();
}

/// Centralizes the polyhedra and reflects the change in the converse space
template <class scalar>
void EigenPolyhedra<scalar>::centralize(const bool inEigenSpace)
{
  if (inEigenSpace) {
    getEigenPolyhedra();
    m_eigenPolyhedra.decentralize();
    m_eigenPolyhedra.centralize();
    m_eigenCentre=m_eigenPolyhedra.getCentre();
    m_centre=m_eigenCentre*m_pInverseEigenVectors->transpose();
    m_polyhedra.translate(m_centre);
  }
  else {
    m_polyhedra.decentralize();
    m_polyhedra.centralize();
    m_centre=m_polyhedra.getCentre();
    m_eigenCentre=m_eigenCentre*m_pEigenVectors->transpose();
    m_eigenPolyhedra.translate(m_eigenCentre);
  }
}

/// Retrieve the centre of the polyhedra
template <class scalar>
typename Tableau<scalar>::MatrixS EigenPolyhedra<scalar>::getCentre(const bool fromEigenSpace,const bool inEigenSpace)
{
  centralize(fromEigenSpace);
  return inEigenSpace ? m_eigenCentre : m_centre;
}

/// Creates m_pEigenPolyhedra if not existent
template <class scalar>
AbstractPolyhedra<scalar>& EigenPolyhedra<scalar>::getEigenPolyhedra(bool force)
{
  if (force || m_eigenPolyhedra.isEmpty()) {
    m_polyhedra.getTransformedPolyhedra(m_eigenPolyhedra,*m_pInverseEigenVectors,*m_pEigenVectors);
  }
  return m_eigenPolyhedra;
}

template <class scalar>
AbstractPolyhedra<scalar>& EigenPolyhedra<scalar>::getSingularPolyhedra(const std::vector<int> &rotations,const std::vector<int> &dilations)
{
  if (m_singularPolyhedra.isEmpty()) {
    AbstractPolyhedra<scalar> source(getEigenPolyhedra());
    source.getRounded(rotations,dilations,m_singularPolyhedra);
  }
  return m_singularPolyhedra;
}

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class EigenPolyhedra<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class EigenPolyhedra<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class EigenPolyhedra<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class EigenPolyhedra<mpinterval>;
  #endif
#endif

}
