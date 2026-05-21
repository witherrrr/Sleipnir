// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace slp {

/// Solver iteration information exposed to an iteration callback.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IterationInfo {
  /// The solver iteration.
  int iteration;

  /// The decision variables.
  const Eigen::Vector<Scalar, Eigen::Dynamic>& x;

  /// The inequality constraint slack variables.
  const Eigen::Vector<Scalar, Eigen::Dynamic>& s;

  /// The equality constraint dual variables.
  const Eigen::Vector<Scalar, Eigen::Dynamic>& y;

  /// The inequality constraint dual variables.
  const Eigen::Vector<Scalar, Eigen::Dynamic>& z;

  /// The gradient of the cost function.
  const Eigen::SparseVector<Scalar>& g;

  /// The Hessian of the Lagrangian.
  const Eigen::SparseMatrix<Scalar>& H;

  /// The equality constraint Jacobian.
  const Eigen::SparseMatrix<Scalar>& A_e;

  /// The inequality constraint Jacobian.
  const Eigen::SparseMatrix<Scalar>& A_i;

  /// Whether the solver is in the feasibility restoration phase. When true,
  /// f/g/H/A_e/A_i describe the feasibility restoration sub-problem rather
  /// than the original problem.
  bool in_feasibility_restoration = false;
};

}  // namespace slp
