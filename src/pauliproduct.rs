use rand::Rng;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    T,
    S,
    SX,
    CX,
    M,
    Z,
    X,
}

impl GateType {
    pub fn is_t(&self) -> bool {
        matches!(self, GateType::T)
    }

    pub fn is_s(&self) -> bool {
        matches!(self, GateType::S)
    }

    pub fn is_sx(&self) -> bool {
        matches!(self, GateType::SX)
    }

    pub fn is_cx(&self) -> bool {
        matches!(self, GateType::CX)
    }

    pub fn is_m(&self) -> bool {
        matches!(self, GateType::M)
    }

    pub fn is_x(&self) -> bool {
        matches!(self, GateType::X)
    }

    pub fn is_z(&self) -> bool {
        matches!(self, GateType::Z)
    }

    pub fn is_clifford(&self) -> bool {
        matches!(self, GateType::S | GateType::SX | GateType::CX | GateType::M)
    }
}

#[derive(Debug, Clone)]
pub struct Operator {
    pub qubit: usize,
    pub basis: char,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.qubit, self.basis)
    }
}

#[derive(Debug, Clone)]
pub struct PauliProduct {
    pub operators: Vec<Operator>,
    pub parents: Vec<i32>,
    pub children: Vec<i32>,
    pub max_qubit: usize,
    pub id: i32,
    pub gate_type: GateType,
}

impl Default for PauliProduct {
    fn default() -> Self {
        PauliProduct { operators: Vec::new(),
                       max_qubit: 0,
                       parents: Vec::new(),
                       children: Vec::new(),
                       id: -1,
                       gate_type: GateType::T }
    }
}

impl PauliProduct {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_from_str(&mut self, product_id: i32, s: &str) -> Result<(), Box<dyn Error>> {
        self.id = product_id;

        for (i, c) in s.chars().enumerate() {
            if i == 0 {
                continue;
            }

            match c {
                '_' => continue,
                'X' | 'Z' | 'Y' => {
                    self.operators.push(Operator { qubit: i - 1, basis: c });
                }
                '<' => {
                    let gate_type = &s[i..];
                    match gate_type {
                        "<M>" => self.gate_type = GateType::M,
                        "<pi/8>" => self.gate_type = GateType::T,
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
                    return Err(format!("Illegal character {} at position {} in product {}",
                                       c, i, s).into());
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

    pub fn get_product_str(&self) -> String {
        let ops = self.operators.iter().map(|op| op.to_string()).collect::<String>();
        format!("{}<{:?}>", ops, self.gate_type)
    }

    pub fn get_qubits(&self) -> Vec<usize> {
        self.operators.iter().map(|op| op.qubit).collect()
    }

    pub fn gen_rnd_t(product_id: i32, num_qubits: usize, spread_probability: f64,
                     decay_factor: f64)
                     -> Self {
        let mut rng = rand::thread_rng();
        let mut operators = Vec::new();
        // Choose initial random location
        let center_qubit = rng.gen_range(0..num_qubits);
        // Set operator at center location with random basis
        let center_basis = ['X', 'Y', 'Z'][rng.gen_range(0..3)];
        operators.push(Operator { qubit: center_qubit, basis: center_basis });
        // Spread left from center
        let mut current_prob = spread_probability;
        for distance in 1..=center_qubit {
            if rng.gen_range(0.0..1.0) < current_prob {
                let qubit = center_qubit - distance;
                let basis = ['X', 'Y', 'Z'][rng.gen_range(0..3)];
                operators.push(Operator { qubit, basis });
            }
            current_prob *= decay_factor;
            if current_prob < 0.001 {
                break;
            } // Stop if probability becomes negligible
        }
        // Spread right from center
        current_prob = spread_probability;
        for distance in 1..(num_qubits - center_qubit) {
            if rng.gen_range(0.0..1.0) < current_prob {
                let qubit = center_qubit + distance;
                let basis = ['X', 'Y', 'Z'][rng.gen_range(0..3)];
                operators.push(Operator { qubit, basis });
            }
            current_prob *= decay_factor;
            if current_prob < 0.001 {
                break;
            } // Stop if probability becomes negligible
        }
        // Sort operators by qubit index for consistency
        operators.sort_by_key(|op| op.qubit);
        // Determine max qubit
        let max_qubit = operators.iter().map(|op| op.qubit).max().unwrap_or(0);

        PauliProduct { operators,
                       parents: Vec::new(),
                       children: Vec::new(),
                       max_qubit,
                       id: product_id,
                       gate_type: GateType::T }
    }

    pub fn to_circuit_format(&self, num_qubits: usize) -> String {
        let mut rng = rand::thread_rng();
        let sign = if rng.gen_bool(0.5) { "+" } else { "-" };
        let mut pauli_string = vec!['_'; num_qubits];
        for op in &self.operators {
            pauli_string[op.qubit] = op.basis;
        }
        format!("{}{}<{:?}>", sign, pauli_string.iter().collect::<String>(), self.gate_type)
    }

    pub fn count_weighted_terms(&self) -> usize {
        self.operators.iter().map(|op| if op.basis == 'Y' { 2 } else { 1 }).sum()
    }
}

impl fmt::Display for PauliProduct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ops = self.operators.iter().map(|op| op.to_string()).collect::<String>();
        write!(f,
               "{} {} <{:?}> children {:?} parents {:?}",
               self.id, ops, self.gate_type, self.children, self.parents)
    }
}
