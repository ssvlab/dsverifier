//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford

#include <time.h>
#include "DualSimplex.h"

#ifndef GENERAL_SIMPLEX_H
#define GENERAL_SIMPLEX_H

namespace abstract{
typedef enum {eCrissCross, eDualSimplex} SolverType_t;

template <class scalar> class GeneralSimplex : public DualSimplex<scalar> {
public:
    using typename Tableau<scalar>::refScalar;
    using typename Tableau<scalar>::func;
    using typename Tableau<scalar>::MatrixS;

    GeneralSimplex(int size,int dimension);

    void SetSolutions(scalar* sol, scalar* dsol);
    scalar solve(ConversionType_t lpconv, SolverType_t solver,int UseExisting, scalar *sol, scalar *dsol);

protected:
    /// Selects a pivot row and column based on the first available non-zero entry
    /// Fast version: assumes no ordering (ie Lmin)
    /// @param rowmax maximum available row for pivot
    /// @param noPivotRow set of restricted rows that cannot pivot
    /// @param noPivotCol set of restricted columns that cannot pivot
    /// @param pivot return row and column to pivot on
    bool FastSelectPivot(const long rowmax, const Set &noPivotRow, const Set &noPivotCol, pivot_t &pivot);

    /// Finds a feasible basis to start the maximise process
    int FindFeasLPBasis();

    /// Selects a pivot for the primal solution (find an optimal solution whilst remaining feasible)
    /// @param pivot return row and column to pivot on
    bool SelectPrimalSimplexPivot(pivot_t &pivot);

    int FindPrimalFeasibleBasis();

    /// finds the maximum for the selected problem (tableau)
    virtual scalar processMaximize(int UseExisting);

public:
  };
}
#endif//GENERAL_SIMPLEX_H
