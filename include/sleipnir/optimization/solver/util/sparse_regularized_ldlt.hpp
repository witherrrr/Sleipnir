// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/util/inertia.hpp"
#include "sleipnir/util/print.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/// Solves sparse systems of linear equations using a regularized LDLT
/// factorization.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class SparseRegularizedLDLT {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  /// Constructs a SparseRegularizedLDLT instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  SparseRegularizedLDLT(int num_decision_variables,
                        int num_equality_constraints)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints} {}

  /// Constructs a SparseRegularizedLDLT instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  /// @param γ_min The minimum constraint regularization.
  SparseRegularizedLDLT(int num_decision_variables,
                        int num_equality_constraints, Scalar γ_min)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints},
        m_γ_min{γ_min} {}

  /// Reports whether previous computation was successful.
  ///
  /// @return Whether previous computation was successful.
  Eigen::ComputationInfo info() const { return m_info; }

  /// Computes the regularized LDLT factorization of a matrix.
  ///
  /// The matrix's symbolic decomposition is reused in subsequent calls, so
  /// subsequent calls must be given a matrix with the same sparsity pattern.
  ///
  /// @param lhs Left-hand side of the system.
  /// @return The factorization.
  SparseRegularizedLDLT& compute(const SparseMatrix& lhs) {
    // The regularization procedure is based on algorithm B.1 of [1]

    // Regularization with zeros ensures the pattern analysis in the sparse
    // solver is reused by all factorizations
    SparseMatrix unregularized_lhs = lhs + regularization(Scalar(0), Scalar(0));

    if (!m_analyzed_pattern) {
      m_solver.analyzePattern(unregularized_lhs);
      m_analyzed_pattern = true;
    }

    m_solver.factorize(unregularized_lhs);
    m_info = m_solver.info();

    if (m_info == Eigen::Success) {
      auto D = m_solver.vectorD();

      // If the inertia is ideal and D from LDLT is sufficiently far from zero,
      // don't regularize the system
      if (Inertia{D} == ideal_inertia &&
          (D.cwiseAbs().array() >= Scalar(1e-4)).all()) {
        m_prev_δ = Scalar(0);
        m_prev_γ = Scalar(0);
        return *this;
      }
    }

    // Also regularize the Hessian. If the Hessian wasn't regularized in a
    // previous run of compute(), start at small values of δ and γ. Otherwise,
    // attempt a δ and γ half as big as the previous run so δ and γ can trend
    // downwards over time.
    Scalar δ = m_prev_δ == Scalar(0) ? Scalar(1e-4) : m_prev_δ / Scalar(2);
    Scalar γ = m_prev_γ == Scalar(0) ? m_γ_min : m_prev_γ / Scalar(2);

    slp::println(
        "[reg-LDLT] enter loop: ideal=({}+ {}, {}0), start δ={:.3e} γ={:.3e}",
        ideal_inertia.positive, ideal_inertia.negative, ideal_inertia.zero,
        δ, γ);

    int attempt = 0;
    while (true) {
      m_solver.factorize(lhs + regularization(δ, γ));
      m_info = m_solver.info();

      if (m_info == Eigen::Success) {
        Inertia inertia{m_solver.vectorD()};

        slp::println(
            "[reg-LDLT]   attempt {}: factorize OK, inertia=({}+ {}, {}0), "
            "δ={:.3e} γ={:.3e}",
            attempt, inertia.positive, inertia.negative, inertia.zero, δ,
            γ);

        if (inertia == ideal_inertia) {
          // If the inertia is ideal, report success
          slp::println(
              "[reg-LDLT]   -> ideal inertia hit, return (δ={:.3e}, γ={:.3e})",
              δ, γ);
          m_prev_δ = δ;
          m_prev_γ = γ;
          return *this;
        } else if (inertia.zero > 0) {
          // If there's zero eigenvalues, check which type of inertia we need
          slp::println("[reg-LDLT]   -> branch: ZERO eigenvalues ({}, {}-, {}+), need {} positive, {} negative",
                       inertia.zero, inertia.negative, inertia.positive, ideal_inertia.positive,
                       ideal_inertia.negative);
          if (inertia.negative < ideal_inertia.negative &&
              inertia.positive < ideal_inertia.positive) {
            slp::println(
                "[reg-LDLT]   -> subbranch: need more NEGATIVE and POSITIVE");
            // If we need more negative and positive eigenvalues, increase both
            // δ and γ by an order of magnitude and try again
            δ *= Scalar(10);
            if (γ == Scalar(0)) {
              γ = Scalar(1e-10);
            } else {
              γ *= Scalar(10);
            }
          } else if (inertia.negative < ideal_inertia.negative) {
            slp::println(
                "[reg-LDLT]   -> subbranch: need more NEGATIVE eigenvalues");
            // If we need more negative eigenvalues, increase γ by an order of
            // magnitude and try again
            if (γ == Scalar(0)) {
              γ = Scalar(1e-10);
            } else {
              γ *= Scalar(10);
            }
          } else if (inertia.positive < ideal_inertia.positive) {
            slp::println(
                "[reg-LDLT]   -> subbranch: need more POSITIVE eigenvalues");
            // If we need more positive eigenvalues, increase δ by an order of
            // magnitude and try again
            δ *= Scalar(10);
          } else {
            slp::println(
                "[reg-LDLT]   -> subbranch: zero eigenvalues only, bump both");
            // If we have the right number of negative and positive eigenvalues,
            // but some are zero, increase both δ and γ by an order of magnitude
            // and try again
            δ *= Scalar(10);
            if (γ == Scalar(0)) {
              γ = Scalar(1e-10);
            } else {
              γ *= Scalar(10);
            }
          }
        } else if (inertia.negative > ideal_inertia.negative) {
          // If there's too many negative eigenvalues, increase δ
          slp::println(
              "[reg-LDLT]   -> branch: too many NEGATIVE ({} > {}), bump δ ×10",
              inertia.negative, ideal_inertia.negative);
          δ *= Scalar(10);
        } else if (inertia.positive > ideal_inertia.positive) {
          // If there's too many positive eigenvalues, increase γ
          slp::println(
              "[reg-LDLT]   -> branch: too many POSITIVE ({} > {}), bump γ ×10",
              inertia.positive, ideal_inertia.positive);
          γ = γ == Scalar(0) ? Scalar(1e-10) : γ * Scalar(10);
        }
      } else {
        // If the decomposition failed, increase δ and reset γ
        slp::println(
            "[reg-LDLT]   attempt {}: factorize FAILED (info={}), δ={:.3e} "
            "γ={:.3e} -> branch: bump δ ×10 γ ×10",
            attempt, static_cast<int>(m_info), δ, γ);
        δ *= Scalar(10);
        γ = Scalar(1e-10);
      }
      ++attempt;
    }
  }

  /// Solves the system of equations using a regularized LDLT factorization.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  template <typename Rhs>
  DenseVector solve(const Eigen::MatrixBase<Rhs>& rhs) const {
    return m_solver.solve(rhs);
  }

  /// Solves the system of equations using a regularized LDLT factorization.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  template <typename Rhs>
  DenseVector solve(const Eigen::SparseMatrixBase<Rhs>& rhs) const {
    return m_solver.solve(rhs);
  }

  /// Returns the Hessian regularization factor.
  ///
  /// @return Hessian regularization factor.
  Scalar hessian_regularization() const { return m_prev_δ; }

  /// Returns the constraint Jacobian regularization factor.
  ///
  /// @return Constraint Jacobian regularization factor.
  Scalar constraint_jacobian_regularization() const { return m_prev_γ; }

 private:
  using Solver = Eigen::SimplicialLDLT<SparseMatrix>;

  Solver m_solver;
  bool m_analyzed_pattern = false;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  int m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  int m_num_equality_constraints = 0;

  /// The minimum constraint regularization.
  Scalar m_γ_min{1e-10};

  /// The ideal system inertia.
  Inertia ideal_inertia{m_num_decision_variables, m_num_equality_constraints,
                        0};

  /// The value of δ from the previous run of compute().
  Scalar m_prev_δ{0};

  /// The value of γ from the previous run of compute().
  Scalar m_prev_γ{0};

  /// Returns regularization matrix.
  ///
  ///   [δI    0]
  ///   [ 0  −γI]
  ///
  /// @param δ The Hessian regularization factor.
  /// @param γ The equality constraint Jacobian regularization factor.
  /// @return Regularization matrix.
  SparseMatrix regularization(Scalar δ, Scalar γ) const {
    DenseVector vec{m_num_decision_variables + m_num_equality_constraints};
    vec.segment(0, m_num_decision_variables).setConstant(δ);
    vec.segment(m_num_decision_variables, m_num_equality_constraints)
        .setConstant(-γ);

    return SparseMatrix{vec.asDiagonal()};
  }
};

}  // namespace slp
