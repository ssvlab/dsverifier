//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef SORTEDMATRIX_H
#define SORTEDMATRIX_H

#include "Scalar.h"
#include "Set.h"
#include <vector>

namespace abstract{
#define RHSCol 0    //Right Hand Side Column
typedef enum { MaxIndex, MinIndex, LexMin, LexMax, LexAngle } OrderType;

template <class scalar> class SortedMatrix : public Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic> {
public:
    typedef interval_def<scalar> type;
    typedef typename type::scalar_type refScalar;
    typedef typename type::MatrixS  MatrixS;
    typedef functions<refScalar> func;

    using Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic>::coeff;
    using Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic>::rows;
    using Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic>::cols;

    SortedMatrix(int rows,int cols);

    void ComputeRowOrderVector(const OrderType &type);

    void UpdateRowOrderVector(const Set &priorityRows);

protected:
    /// Checks if row1<row2 ordered by adjacent column angle
    /// @param row1 row to check for
    /// @param row2 row to check against
    /// @return true if row1 is smaller than row2 by angle
    bool SmallerAngleRow(const int row1,const int row2);

    /// Checks if row1>row2 ordered by adjacent column angle
    /// @param row1 row to check for
    /// @param row2 row to check against
    /// @return true if row1 is larger than row2 by angle
    bool LargerAngleRow(const int row1,const int row2);

    /// Checks if row1<row2 ordered by column
    /// @param row1 row to check for
    /// @param row2 row to check against
    /// @return true if row1 is smaller than row2
    bool SmallerRow(const int row1,const int row2);

    /// Checks if row1>row2 ordered by column
    /// @param row1 row to check for
    /// @param row2 row to check against
    /// @return true if row1 is larger than row2
    bool LargerRow(const int row1,const int row2);

    long subPartition(std::vector<int> &order, const long p, const long r);

    long anglePartition(std::vector<int> &order, const long p, const long r);

    void QuickMatrixSort(std::vector<int> &order, const long p, const long r);

public:

    void QuickAngleSort();

    inline int order(const int row)       { return m_order[row]; }
    inline int zeroOrder(const int row)   { return m_order[row]-1; }//TODO: zero index
protected:
    std::vector<int>            m_order;
    static scalar ms_one;
};
}
#endif
