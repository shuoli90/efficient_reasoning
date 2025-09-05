import Mathlib.Topology.Basic

import Mathlib

open Nat

/-- Prove that the fraction $\\frac{21n+4}{14n+3}$ is irreducible for every natural number $n$.-/

theorem imo_1959_p1
  (n : ℕ)
  (h₀ : 0 < n) :
  Nat.gcd (21*n + 4) (14*n + 3) = 1 := by
  have h₁: Nat.gcd (21*n + 4) (14*n + 3) = Nat.gcd (7*n + 1) (14*n + 3) := by
    have g₀: (21 * n + 4) = (7*n + 1) + 1 * (14 * n + 3) := by linarith
    rw [g₀]
    exact gcd_add_mul_right_left (14 * n + 3) (7 * n + 1) 1
  have h₂: Nat.gcd (7*n + 1) (14*n + 3) = Nat.gcd (7*n + 1) (1) := by
    have g₁: 14 * n + 3 = (7 * n + 1) * 2 + 1 := by linarith
    rw [g₁]
    exact gcd_mul_left_add_right (7 * n + 1) 1 2
  have h₃: Nat.gcd (7*n + 1) (1) = 1 := by
    exact Nat.gcd_one_right (7 * n + 1)
  linarith