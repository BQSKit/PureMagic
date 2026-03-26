use std::error::Error;
use std::fmt;

/// Quantum gate types in the circuit.
/// T gates require magic state distillation; S/SX/CX are Cliffords and repeat multiple times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum GateType {
    T,
    S,
    SX,
    CX,
    M,
    Z,
    X,
}

impl GateType {
    /// Returns true if this is a T gate.
    pub(crate) fn is_t(&self) -> bool {
        matches!(self, GateType::T)
    }

    /// Returns true if this is an S gate.
    pub(crate) fn is_s(&self) -> bool {
        matches!(self, GateType::S)
    }

    /// Returns true if this is an SX gate.
    pub(crate) fn is_sx(&self) -> bool {
        matches!(self, GateType::SX)
    }

    /// Returns true if this is a CX (CNOT) gate.
    pub(crate) fn is_cx(&self) -> bool {
        matches!(self, GateType::CX)
    }

    /// Returns true if this is a measurement gate.
    pub(crate) fn is_m(&self) -> bool {
        matches!(self, GateType::M)
    }

    /// Returns true if this is a Pauli X gate.
    pub(crate) fn is_x(&self) -> bool {
        matches!(self, GateType::X)
    }

    /// Returns true if this is a Pauli Z gate.
    pub(crate) fn is_z(&self) -> bool {
        matches!(self, GateType::Z)
    }

    /// Returns true if this is a Clifford gate (CX, S, or SX).
    pub(crate) fn is_clifford(&self) -> bool {
        self.is_cx() || self.is_s() || self.is_sx()
    }
}

impl fmt::Display for GateType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A single Pauli operator (X, Y, or Z) applied to a specific qubit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Operator {
    pub qubit: u16,
    pub basis: char,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.qubit, self.basis)
    }
}

/// A quantum gate represented as a Pauli product with dependency tracking.
/// Weight is the sum of operator costs (Y counts as 2, others as 1).
#[derive(Debug, Clone)]
pub(crate) struct PauliProduct {
    pub operators: Vec<Operator>,
    pub parents: Vec<i32>,
    pub children: Vec<i32>,
    pub max_qubit: u16,
    pub id: i32,
    pub gate_type: GateType,
}

impl Default for PauliProduct {
    fn default() -> Self {
        PauliProduct {
            operators: Vec::new(),
            max_qubit: 0,
            parents: Vec::new(),
            children: Vec::new(),
            id: -1,
            gate_type: GateType::T,
        }
    }
}

impl PauliProduct {
    /// Creates a new empty Pauli product.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Parses a circuit format string into this Pauli product.
    /// Format: `[±][X/Y/Z operators][<gate_type>]` where _ denotes identity on a qubit.
    pub(crate) fn set_from_str(&mut self, product_id: i32, s: &str) -> Result<(), Box<dyn Error>> {
        self.id = product_id;

        for (i, c) in s.chars().enumerate() {
            if i == 0 {
                continue;
            }

            match c {
                '_' => continue,
                'X' | 'Z' | 'Y' => {
                    self.operators.push(Operator { qubit: (i - 1) as u16, basis: c });
                }
                '<' => {
                    let gate_type = &s[i..];
                    match gate_type {
                        "<M>" => self.gate_type = GateType::M,
                        "<T>" => self.gate_type = GateType::T,
                        "<CX>" => self.gate_type = GateType::CX,
                        "<S>" | "<Sdg>" => self.gate_type = GateType::S,
                        "<SX>" | "<SXdg>" => self.gate_type = GateType::SX,
                        "<Z>" => self.gate_type = GateType::Z,
                        "<X>" => self.gate_type = GateType::X,
                        _ => {
                            return Err(format!("Unknown gate {} in {}", gate_type, s).into());
                        }
                    }
                    break;
                }
                _ => {
                    return Err(format!(
                        "Illegal character {} at position {} in product {}",
                        c, i, s
                    )
                    .into());
                }
            }
        }
        if self.gate_type.is_cx() {
            assert_eq!(self.operators.len(), 2);
        } else if self.gate_type.is_s() || self.gate_type.is_sx() {
            assert_eq!(self.operators.len(), 1, "Should have max 1 qubit: {}", self);
        }
        self.max_qubit = self.operators.iter().map(|op| op.qubit).max().unwrap_or(0);
        Ok(())
    }

    /// Returns a string representation of the operators (without sign).
    pub(crate) fn to_operator_str(&self) -> String {
        let ops = self.operators.iter().map(|op| op.to_string()).collect::<String>();
        format!("{}<{:?}>", ops, self.gate_type)
    }

    /// Returns sorted list of qubits on which this product operates.
    pub(crate) fn get_qubits(&self) -> Vec<u16> {
        self.operators.iter().map(|op| op.qubit).collect()
    }
}

impl fmt::Display for PauliProduct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ops = self.operators.iter().map(|op| op.to_string()).collect::<String>();
        write!(
            f,
            "{} {} <{:?}> children {:?} parents {:?}",
            self.id, ops, self.gate_type, self.children, self.parents
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GateType tests ────────────────────────────────────────────────────────

    #[test]
    fn gate_type_is_t() {
        assert!(GateType::T.is_t());
        assert!(!GateType::S.is_t());
        assert!(!GateType::SX.is_t());
        assert!(!GateType::CX.is_t());
        assert!(!GateType::M.is_t());
        assert!(!GateType::Z.is_t());
        assert!(!GateType::X.is_t());
    }

    #[test]
    fn gate_type_is_s() {
        assert!(GateType::S.is_s());
        assert!(!GateType::T.is_s());
        assert!(!GateType::SX.is_s());
    }

    #[test]
    fn gate_type_is_sx() {
        assert!(GateType::SX.is_sx());
        assert!(!GateType::S.is_sx());
        assert!(!GateType::T.is_sx());
    }

    #[test]
    fn gate_type_is_cx() {
        assert!(GateType::CX.is_cx());
        assert!(!GateType::T.is_cx());
        assert!(!GateType::S.is_cx());
    }

    #[test]
    fn gate_type_is_m() {
        assert!(GateType::M.is_m());
        assert!(!GateType::T.is_m());
    }

    #[test]
    fn gate_type_is_x() {
        assert!(GateType::X.is_x());
        assert!(!GateType::T.is_x());
        assert!(!GateType::Z.is_x());
    }

    #[test]
    fn gate_type_is_z() {
        assert!(GateType::Z.is_z());
        assert!(!GateType::T.is_z());
        assert!(!GateType::X.is_z());
    }

    #[test]
    fn gate_type_is_clifford() {
        assert!(GateType::CX.is_clifford());
        assert!(GateType::S.is_clifford());
        assert!(GateType::SX.is_clifford());
        assert!(!GateType::T.is_clifford());
        assert!(!GateType::M.is_clifford());
        assert!(!GateType::Z.is_clifford());
        assert!(!GateType::X.is_clifford());
    }

    #[test]
    fn gate_type_display() {
        assert_eq!(format!("{}", GateType::T), "T");
        assert_eq!(format!("{}", GateType::S), "S");
        assert_eq!(format!("{}", GateType::CX), "CX");
        assert_eq!(format!("{}", GateType::M), "M");
    }

    // ── Operator tests ────────────────────────────────────────────────────────

    #[test]
    fn operator_display() {
        let op = Operator { qubit: 3, basis: 'X' };
        assert_eq!(format!("{}", op), "3X");
    }

    #[test]
    fn operator_display_z_basis() {
        let op = Operator { qubit: 0, basis: 'Z' };
        assert_eq!(format!("{}", op), "0Z");
    }

    // ── PauliProduct::new / default ───────────────────────────────────────────

    #[test]
    fn pauli_product_new_defaults() {
        let pp = PauliProduct::new();
        assert_eq!(pp.id, -1);
        assert!(pp.operators.is_empty());
        assert!(pp.parents.is_empty());
        assert!(pp.children.is_empty());
        assert_eq!(pp.max_qubit, 0);
        assert!(pp.gate_type.is_t());
    }

    // ── PauliProduct::set_from_str ────────────────────────────────────────────

    #[test]
    fn set_from_str_t_gate_single_x() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(0, "+_X______<T>").unwrap();
        assert_eq!(pp.id, 0);
        assert!(pp.gate_type.is_t());
        assert_eq!(pp.operators.len(), 1);
        assert_eq!(pp.operators[0].qubit, 1);
        assert_eq!(pp.operators[0].basis, 'X');
        assert_eq!(pp.max_qubit, 1);
    }

    #[test]
    fn set_from_str_t_gate_multi_operator() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(1, "-ZY______<T>").unwrap();
        assert_eq!(pp.id, 1);
        assert!(pp.gate_type.is_t());
        assert_eq!(pp.operators.len(), 2);
        assert_eq!(pp.operators[0].qubit, 0);
        assert_eq!(pp.operators[0].basis, 'Z');
        assert_eq!(pp.operators[1].qubit, 1);
        assert_eq!(pp.operators[1].basis, 'Y');
        assert_eq!(pp.max_qubit, 1);
    }

    #[test]
    fn set_from_str_m_gate() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(2, "+X_<M>").unwrap();
        assert!(pp.gate_type.is_m());
        assert_eq!(pp.operators.len(), 1);
        assert_eq!(pp.operators[0].qubit, 0);
    }

    #[test]
    fn set_from_str_s_gate() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(3, "+_Z<S>").unwrap();
        assert!(pp.gate_type.is_s());
        assert_eq!(pp.operators.len(), 1);
    }

    #[test]
    fn set_from_str_sdg_gate() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(4, "+_Z<Sdg>").unwrap();
        assert!(pp.gate_type.is_s(), "Sdg should map to S gate type");
    }

    #[test]
    fn set_from_str_sx_gate() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(5, "+X_<SX>").unwrap();
        assert!(pp.gate_type.is_sx());
    }

    #[test]
    fn set_from_str_sxdg_gate() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(6, "+X_<SXdg>").unwrap();
        assert!(pp.gate_type.is_sx(), "SXdg should map to SX gate type");
    }

    #[test]
    fn set_from_str_cx_gate() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(7, "+XZ<CX>").unwrap();
        assert!(pp.gate_type.is_cx());
        assert_eq!(pp.operators.len(), 2);
    }

    #[test]
    fn set_from_str_z_gate() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(8, "+Z_<Z>").unwrap();
        assert!(pp.gate_type.is_z());
    }

    #[test]
    fn set_from_str_x_gate() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(9, "+X_<X>").unwrap();
        assert!(pp.gate_type.is_x());
    }

    #[test]
    fn set_from_str_unknown_gate_returns_error() {
        let mut pp = PauliProduct::new();
        let result = pp.set_from_str(0, "+X_<UNKNOWN>");
        assert!(result.is_err());
    }

    #[test]
    fn set_from_str_illegal_char_returns_error() {
        let mut pp = PauliProduct::new();
        let result = pp.set_from_str(0, "+A_<T>");
        assert!(result.is_err());
    }

    #[test]
    fn set_from_str_max_qubit_computed_correctly() {
        let mut pp = PauliProduct::new();
        // operators at positions 2, 5, 7 (0-indexed after sign char)
        pp.set_from_str(0, "+__X__Z_Z<T>").unwrap();
        assert_eq!(pp.max_qubit, 7);
    }

    // ── PauliProduct::get_qubits ──────────────────────────────────────────────

    #[test]
    fn get_qubits_returns_all_qubit_indices() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(0, "+XZ_<T>").unwrap();
        let qubits = pp.get_qubits();
        assert_eq!(qubits, vec![0, 1]);
    }

    #[test]
    fn get_qubits_empty_when_no_operators() {
        let pp = PauliProduct::new();
        assert!(pp.get_qubits().is_empty());
    }

    // ── PauliProduct::to_operator_str ─────────────────────────────────────────

    #[test]
    fn to_operator_str_format() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(0, "+X_<T>").unwrap();
        let s = pp.to_operator_str();
        assert!(s.contains("0X"), "should contain qubit-basis pair");
        assert!(s.contains("T"), "should contain gate type");
    }

    // ── PauliProduct Display ──────────────────────────────────────────────────

    #[test]
    fn pauli_product_display_contains_id_and_gate() {
        let mut pp = PauliProduct::new();
        pp.set_from_str(42, "+X_<M>").unwrap();
        let s = format!("{}", pp);
        assert!(s.contains("42"));
        assert!(s.contains("M"));
    }
}
