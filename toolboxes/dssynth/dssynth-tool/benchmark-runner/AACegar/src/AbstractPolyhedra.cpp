//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include "AbstractPolyhedra.h"

namespace abstract{

template <class scalar>
bool AbstractPolyhedra<scalar>::ms_trace_abstraction=false;

/// Constructs an empty buffer
template <class scalar>
AbstractPolyhedra<scalar>::AbstractPolyhedra(int dimension) :
    Polyhedra<scalar>(dimension)
{}

/// Constructs transformed polyhedra
template <class scalar>
AbstractPolyhedra<scalar>::AbstractPolyhedra(const AbstractPolyhedra &source,const MatrixS &transform,const MatrixS &inverse) :
    Polyhedra<scalar>(source,transform,inverse)
{
}

/// Retrieves an abstraction that has circular faces in the rotating axis and jordan blocks
template <class scalar>
typename Tableau<scalar>::MatrixS AbstractPolyhedra<scalar>::getRoundedDirections(const MatrixS &vectors,const std::vector<int> &rotations,const std::vector<int> &dilations)
{
  if (rotations.size()<vectors.rows() && dilations.size()<vectors.rows()) {
    return MatrixS(0,0);
  }
  MatrixS result(getDimension(),vectors.cols());
  int pos=0;
  for (int row=0;row<vectors.rows();row++)
  {
    int mult=(rotations[row]<0) ? 1 : 2;
    if ((rotations[row]<0) && (dilations[row+mult]<=0)) {
      result.row(pos)=vectors.row(row);
    }
    else {
      for (int col=0;col<vectors.cols();col++) {
        scalar value=func::squared(vectors.coeff(row,col));//*vectors.coeff(row,col);
        for (int row2=row+1;(rotations[row2]>=0) || (dilations[row2]>0);row2++) {
          value+=func::squared(vectors.coeff(row2,col));
//          func::madd(value,vectors.coeff(row2,col),vectors.coeff(row2,col));
        }
        result.coeffRef(pos,col)=sqrt(value);
      }
      row+=mult;
      while (dilations[row]>0) row+=mult;
      row--;
    }
    pos++;
  }
  return result;
}

/// Retrieves an abstraction that has circular faces in the rotating axis and jordan blocks
template <class scalar>
AbstractPolyhedra<scalar>& AbstractPolyhedra<scalar>::getRounded(const std::vector<int> &rotations,const std::vector<int> &dilations,AbstractPolyhedra<scalar> &roundAbstraction)
{
  int dimension=getDimension();
  int roundDimension=0;
  for (int col=0;col<m_faces.cols();col++)
  {
    if ((dilations[col+1]>0) || (rotations[col]>col)) dimension--;
    else if ((dilations[col]>0) || (rotations[col]>=0)) roundDimension++;
  }
  this->centralize();
  if (dimension==getDimension()) {
    roundAbstraction.copy(*this);
    this->decentralize();
    return roundAbstraction;
  }
  if (!this->makeVertices()) {
    roundAbstraction.clear();
    this->decentralize();
    return roundAbstraction;
  }
  roundAbstraction.changeDimension(dimension);
  roundAbstraction.m_vertices.resize(this->m_vertices.rows(),dimension);
  roundAbstraction.m_faces.resize(m_faces.rows()+roundDimension,dimension);
  roundAbstraction.m_faces.block(m_faces.rows(),0,roundDimension,dimension)=MatrixS::Zero(roundDimension,dimension);
  int pos=0;
  roundDimension=m_faces.rows();
  for (int col=0;col<m_faces.cols();col++)
  {
    int mult=(rotations[col]>col) ? 2 : 1;
    int subDim=mult;
    while(dilations[col+subDim]>0) subDim+=mult;
    if (subDim>1) {
      for (int row=0;row<this->m_vertices.rows();row++) {
        roundAbstraction.m_vertices.coeffRef(row,pos)=this->m_vertices.block(row,col,1,subDim).norm();
      }
      for (int row=0;row<m_faces.rows();row++) {
        roundAbstraction.m_faces.coeffRef(row,pos)=m_faces.block(row,col,1,subDim).norm();
      }
      roundAbstraction.m_faces.coeffRef(roundDimension++,pos)=-1;
      col+=subDim-1;
    }
    else {
      roundAbstraction.m_vertices.col(pos)=this->m_vertices.col(col);
      roundAbstraction.m_faces.block(0,pos,m_faces.rows(),1)=m_faces.col(col);
    }
    pos++;
  }
  this->decentralize();
  //roundAbstraction.m_supports.resize(roundAbstraction.m_faces.rows()+roundDimension,1);
  MatrixS matrix=roundAbstraction.m_faces*roundAbstraction.m_vertices.transpose();
  roundAbstraction.m_supports=matrix.col(0);
  for (int row=0;row<matrix.rows();row++) {
    for (int col=1;col<matrix.cols();col++) {
      if (func::toUpper(matrix.coeff(row,col))>func::toUpper(roundAbstraction.m_supports.coeff(row,0))) {
        roundAbstraction.m_supports.coeffRef(row,0)=matrix.coeff(row,col);
      }
    }
  }
  //roundAbstraction.m_supports=matrix.rowwise().maxCoeff();
  roundAbstraction.m_supports.block(m_faces.rows(),0,roundDimension-m_faces.rows(),1)=MatrixS::Zero(roundDimension-m_faces.rows(),1);
  roundAbstraction.Tableau<scalar>::load(roundAbstraction.m_faces,roundAbstraction.m_supports);
  roundAbstraction.removeRedundancies();
  return roundAbstraction;
  }

/// Performs the Minkowski sum of this polyhedra to another
template <class scalar>
bool AbstractPolyhedra<scalar>::addRounded(Polyhedra<scalar> &polyhedra,const std::vector<int> &rotations,const std::vector<int> &dilations)
{
  if (rotations.size()<m_faces.cols() && dilations.size()<m_faces.cols()) return false;
  MatrixS vectors(polyhedra.getDimension(),m_faces.rows());
  int col2=0;
  for (int row=0;row<m_faces.rows();row++) {
    for (int col=m_faces.cols()-1;col>=0;col--) {
      if ((dilations[col]>0) || rotations[col]>=0) {
        scalar coeff=0;
        while (dilations[col]>0) {
          func::madd(coeff,m_faces.coeff(row,col),m_faces.coeff(row,col));
          col--;
        }
        if (rotations[col]==col-1) {
          func::madd(coeff,m_faces.coeff(row,col),m_faces.coeff(row,col));
          col--;
        }
        func::madd(coeff,m_faces.coeff(row,col),m_faces.coeff(row,col));
        vectors.coeffRef(col2++,row)=sqrt(coeff);
      }
      else vectors.coeffRef(col2++,row)=m_faces.coeff(row,col);
    }
  }
  MatrixS supports(m_supports.rows(),1);
  if (polyhedra.maximiseAll(vectors,supports)) {
    m_supports+=supports;
    return true;
  }
  return false;
}

/// Retrives the vertices mapped into an abstract domain of a linear matrix
template <class scalar>
typename Tableau<scalar>::MatrixS AbstractPolyhedra<scalar>::getAbstractVertices(MatrixS &vector,const int row,const std::vector<int> &rotations,const std::vector<int> &dilations,const MatrixS& vertices) {
  /// sup<v.xy>=sup(sum_i(v_ix_iy_i))=sup<vx.y>
  /// vector is after transformation thus v1(ax1+bx2)+v2(ax2-bx1)=a(v1x1+v2x2)+b(v1x2-v2x1)
  /// for real jordan =a(v1x1+v2x2)+b(v1x2) on the previous col (series=a(sum(vi xi))+b(sum(v{i-1} xi))+c(sum(v{i-2} xi))...)
  /// for complex jordan

  int rows=vertices.rows();
  MatrixS result(vertices.rows(),vertices.cols());
  if ((rotations.size()<vertices.cols()) || (dilations.size()<vertices.cols())) {
    ms_logger.logData("Dimension error");
    throw dimensionMismatch;
  }
  for (int col=0;col<vertices.cols();col++){
    result.col(col)=vertices.col(col)*vector.coeff(row,col);
    if (rotations[col]>col) {
      col++;
      result.col(col-1)+=vertices.col(col)*vector.coeff(row,col);
      result.col(col)=vertices.col(col)*vector.coeff(row,col-1)-vertices.col(col-1)*vector.coeff(row,col);
      if (dilations[col]>0) {
        result.block(0,col-2*dilations[col]-1,rows,2)+=result.block(0,col-1,rows,2);
        result.col(col-1)=vertices.col(col-1)*vector.coeff(row,2*dilations[col]-1)+vector.coeff(row,col-2*dilations[col])*vertices.col(col);
        result.col(col)=vector.coeff(row,col-2*dilations[col]-1)*vertices.col(col)-vector.coeff(row,col-2*dilations[col])*vertices.col(col-1);
        for (int offset=1;offset<dilations[col];offset++) {
          result.col(col-2*offset-1)+=vector.coeff(row,col-2*(dilations[col]-offset)-1)*vertices.col(col-1)+vector.coeff(row,col-2*(dilations[col]-offset))*vertices.block(0,col,rows,1);
          result.col(col-2*offset  )+=vector.coeff(row,col-2*(dilations[col]-offset)-1)*vertices.col(col  )-vector.coeff(row,col-2*(dilations[col]-offset))*vertices.block(0,col-1,rows,1);
        }
      }
    }
    else if (dilations[col]>0) {
      result.col(col-dilations[col])+=result.col(col);
      result.col(col)=vector.coeff(row,col-dilations[col])*vertices.col(col);
      for (int offset=1;offset<dilations[col];offset++) {
        result.col(col-offset)+=vector.coeff(row,col-dilations[col]+offset)*vertices.col(col);
      }
    }
  }
  return result;
}

/// Retrives the vertices mapped into an abstract domain of a linear matrix
template <class scalar>
typename Tableau<scalar>::MatrixS AbstractPolyhedra<scalar>::getAbstractVertices(const MatrixS &vectors,const std::vector<int> &rotations,const std::vector<int> &dilations,const MatrixS &vertices)
{
  /// sup<v.xy>=sup(sum_i(v_ix_iy_i))=sup<vx.y>
  /// vector is after transformation thus v1(ax1+bx2)+v2(ax2-bx1)=a(v1x1+v2x2)+b(v1x2-v2x1)
  /// for real jordan =a(v1x1+v2x2)+b(v1x2) on the previous col (series=a(sum(vi xi))+b(sum(v{i-1} xi))+c(sum(v{i-2} xi))...)
  /// for complex jordan (series=a(sum(vi xi))+b(sum(v{2j-2} x{2j+1}-v{2j} x{2j+1}))+c(sum(v{i-2} xi))...)
  MatrixS result(vectors.cols()*vertices.rows(),vertices.cols());
  if (rotations.size()<vertices.cols() || dilations.size()<vertices.cols()) {
    ms_logger.logData("Dimension error");
    throw dimensionMismatch;
  }
  int rows=vertices.rows();
  for (int vectorNum=0;vectorNum<vectors.cols();vectorNum++) {
    int blockNum=vectorNum*rows;
    for (int col=0;col<vertices.cols();col++){
      result.block(blockNum,col,rows,1)=vertices.block(0,col,rows,1)*vectors.coeff(col,vectorNum);
      if (rotations[col]>col) {
        col++;
        result.block(blockNum,col-1,rows,1)+=vertices.col(col)*vectors.coeff(col  ,vectorNum);
        result.block(blockNum,col,rows,1)=   vertices.col(col)*vectors.coeff(col-1,vectorNum)-vertices.col(col-1)*vectors.coeff(col,vectorNum);
        if (dilations[col]>0) {
          result.block(blockNum,col-2*dilations[col]-1,rows,2)+=result.block(blockNum,col-1,rows,2);
          result.block(blockNum,col-1,rows,1)=vectors.coeff(col-2*dilations[col]-1,vectorNum)*vertices.col(col-1)+vectors.coeff(col-2*dilations[col],vectorNum)*vertices.col(col);
          result.block(blockNum,col  ,rows,1)=vectors.coeff(col-2*dilations[col]-1,vectorNum)*vertices.col(col  )-vectors.coeff(col-2*dilations[col],vectorNum)*vertices.col(col-1);
          for (int offset=1;offset<dilations[col];offset++) {
            int voff=2*(dilations[col]-offset);
            result.block(blockNum,col-2*offset-1,rows,1)+=vectors.coeff(col-voff-1,vectorNum)*vertices.col(col-1)+vectors.coeff(col-voff,vectorNum)*vertices.col(col);
            result.block(blockNum,col-2*offset  ,rows,1)+=vectors.coeff(col-voff-1,vectorNum)*vertices.col(col  )-vectors.coeff(col-voff,vectorNum)*vertices.col(col-1);
          }
        }
      }
      else if (dilations[col]>0) {
        result.block(blockNum,col-dilations[col],rows,1)+=result.block(blockNum,col,rows,1);
        result.block(blockNum,col,rows,1)=vectors.coeff(col-dilations[col],vectorNum)*vertices.col(col);
        for (int offset=1;offset<dilations[col];offset++) {
          result.block(blockNum,col-offset,rows,1)+=vectors.coeff(col-dilations[col]+offset,vectorNum)*vertices.col(col);
        }
      }
    }
  }
  if (this->ms_trace_abstraction) {
    ms_logger.logData(this->m_name);
    ms_logger.logData(vertices,"Vertices:");
    ms_logger.logData(vectors,"Vectors:");
    ms_logger.logData(result,"Abstract Vertices:");
  }
  return result.transpose();
}

template<class scalar>
typename Tableau<scalar>::MatrixS AbstractPolyhedra<scalar>::getSynthVertices(const MatrixS &vectors,const std::vector<int> &rotations,const std::vector<int> &dilations,const MatrixS &vertices)
{
  /// sup<v.xy>=sup(sum_i(v_ix_iy_i))=sup<vx.y>
  /// vector is after transformation thus v1(ax1+bx2)+v2(ax2-bx1)=x1(av1-bv2)+x2(av2+bv1)
  /// for real jordan =x1(av1)+x2(av2+bv1)+... xn(avn+bv{n-1}...+nv{1})
  /// for complex jordan
  int cols=vertices.cols();
  if (vectors.rows()<cols) cols=vectors.rows();
  MatrixS result(vectors.cols()*vertices.rows(),cols);
  int rows=vertices.rows();
  for (int vectorNum=0;vectorNum<vectors.cols();vectorNum++) {
    int blockNum=vectorNum*rows;
    for (int col=0;col<cols;col++) {
      result.block(blockNum,col,rows,1)=vectors.coeff(col,vectorNum)*vertices.col(col);
      if (rotations[col]>=0) {
        col++;
        result.block(blockNum,col-1,rows,1)-=vectors.coeff(col,vectorNum)*vertices.col(col);
        result.block(blockNum,col,rows,1)=vectors.coeff(col-1,vectorNum)*vertices.col(col)+vectors.coeff(col,vectorNum)*vertices.col(col-1);
        if (dilations[col]>0) {
          result.block(blockNum,col,rows,1)=MatrixS::Zero(rows,1);
          if (rotations[col]>col) {
            for (int offset=-1;offset<2*dilations[col];offset++) {
              if (offset&1) result.block(blockNum,col,rows,1)-=vectors.coeff(col-2*dilations[col]+offset,vectorNum)*vertices.col(col-offset);
              else result.block(blockNum,col,rows,1)+=vectors.coeff(col-2*dilations[col]+offset,vectorNum)*vertices.col(col-offset);
            }
          }
          else {
            for (int offset=0;offset<2*dilations[col];offset++) {
              result.block(blockNum,col,rows,1)+=vectors.coeff(col-2*dilations[col]+offset,vectorNum)*vertices.col(col-offset);
            }
          }
        }
      }
      else if (dilations[col]>0) {
        result.block(blockNum,col,rows,1)=MatrixS::Zero(rows,1);
        for (int offset=0;offset<dilations[col];offset++) {
          result.block(blockNum,col,rows,1)+=vectors.coeff(col-dilations[col]+offset,vectorNum)*vertices.col(col-offset);
        }
      }
    }
  }
  if (this->ms_trace_abstraction) {
    ms_logger.logData(this->m_name);
    ms_logger.logData(vertices,"Vertices:");
    ms_logger.logData(vectors,"Vectors:");
    ms_logger.logData(result,"Synth Vertices:");
  }
  return result;
}

/// Retrives the vertices mapped into an abstract domain of a linear matrix
template <class scalar>
typename Tableau<scalar>::MatrixS AbstractPolyhedra<scalar>::getAbstractVertices(MatrixS &vectors)
{
  const MatrixS &vertices=this->getVertices();
  int rows=vertices.rows();
  MatrixS result(vectors.cols()*rows,vertices.cols());
  for (int vectorNum=0;vectorNum<vectors.cols();vectorNum++) {
    for (int col=0;col<vertices.cols();col++){
      result.block(vectorNum*rows,col,rows,1)=vertices.col(col)*vectors.coeff(col,vectorNum);
    }
  }
  return result.transpose();
}

/// Retrieves a copy of this polyhedra transformed by the given matrix
template <class scalar>
AbstractPolyhedra<scalar>& AbstractPolyhedra<scalar>::getTransformedPolyhedra(Polyhedra<scalar> &polyhedra,const MatrixS &transform,const MatrixS &inverse,const MatrixS &templates)
{
  return dynamic_cast<AbstractPolyhedra&>(Polyhedra<scalar>::getTransformedPolyhedra(polyhedra,transform,inverse,templates));
}

#ifdef USE_LDOUBLE
  #ifdef USE_SINGLES
    template class AbstractPolyhedra<long double>;
  #endif
  #ifdef USE_INTERVALS
    template class AbstractPolyhedra<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  #ifdef USE_SINGLES
    template class AbstractPolyhedra<mpfr::mpreal>;
  #endif
  #ifdef USE_INTERVALS
    template class AbstractPolyhedra<mpinterval>;
  #endif
#endif

}
