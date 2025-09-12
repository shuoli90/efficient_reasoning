import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.NumberTheory.Basic
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology

theorem mathd_numbertheory_543 :
  (∑ k in (Nat.divisors (30^4)), 1) - 2 = 123 := by
ring
  {horner { output_le ''{a => 1, b => -3, c => 4, d => 3}:oid ~ untrans ${a/b$num iterable} /= None} Proof_String("(1,0,0,0)+(1,1,0,0)+(1,0,1,0)+(1,1,1,0)+(1,0,0,1)+(1,1,0,1)+(1,1,1,1)+(1,0,1,1)+(1,1,1,2)+(1,0,2,1)+(1,1,2,2)+(1,0,1,2)+(1,2,0,1)+(1,2,1,2)+(1,1,2,1)+(1,2,1,3)+(1,2,2,2)+(1,1,2,3)+(1,2,2,1)+(2,0,0,1)+(2,1,0,2)+(2,0,1,2)+(4,0,1,2)+(1,2,1,2)+(1,2,1,1)+(2,0,1,1)+(2,1,1,1)+(2,2,0,2)+(4,0,2,1)+(4,1,0,1)+(8,0,1,1)+(6,0,1,2)+(6,1,1,3)+(9,0,2,3)+(10,0,1,3)+(3,0,2,1)+(6,2,1,2)+(3,0,1,3)+(3,1,2,1)+(4,0,2,2)+(4,2,0,4)+(8,0,2,2)+(16,0,2,2)+(1,0,2,0)+(1,0,2,2)+(1,0,4,0)+(2,0,1,0)+(2,0,2,2)+(4,1,0,2)+(8,0,1,4)+(6,0,0,2)+(4,1,0,4)+(10,0,0,2)+(16,1,0,2)+(20,0,0,2)+(5,3,0,7)+(5,0,1,2)+(5,1,0,7)+(1,2,1,4)+(5,2,0,8)+(10,0,2,7)+(16,0,2,5)+(6,1,0,3)+(6,0,1,3)+(1,1,3,9)+(5,0,3,3)+(9,1,1,3)+(1+e 4, 2 * e 5, 0 * e 4, 1  p)).trans_fun_etale 3 round }
end mathd_numbertheory_543