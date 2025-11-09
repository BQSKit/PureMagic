use rand::Rng;
use std::error::Error;
use std::fmt;

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
    pub num_ys: usize,
    pub need_estabilizer: bool,
    pub need_ancilla: bool,
    pub is_clifford: bool,
}

impl Default for PauliProduct {
    fn default() -> Self {
        PauliProduct { operators: Vec::new(),
                       max_qubit: 0,
                       parents: Vec::new(),
                       children: Vec::new(),
                       id: -1,
                       num_ys: 0,
                       need_estabilizer: false,
                       need_ancilla: false,
                       is_clifford: false }
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
                    if c == 'Y' {
                        self.num_ys += 1;
                    }
                }
                '<' => {
                    let angle = &s[i..];
                    match angle {
                        "<M>" => self.is_clifford = true,
                        "<pi/8>" => self.is_clifford = false,
                        _ => {
                            return Err(format!("Unknown angle {} in product {}", angle, s).into());
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
        //if self.num_ys % 2 == 1 {
        //    self.need_ancilla = true;
        //}
        self.max_qubit = self.operators.iter().map(|op| op.qubit).max().unwrap_or(0);
        Ok(())
    }

    pub fn get_product_str(&self) -> String {
        let ops = self.operators.iter().map(|op| op.to_string()).collect::<String>();
        let angle = if self.is_clifford { "<M>" } else { "<T>" };
        format!("{}{}", ops, angle)
    }

    pub fn get_qubits(&self) -> Vec<usize> {
        self.operators.iter().map(|op| op.qubit).collect()
    }

    pub fn generate_random(product_id: i32, num_qubits: usize, spread_probability: f64,
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
        // Count Y operators
        let num_ys = operators.iter().filter(|op| op.basis == 'Y').count();
        // Determine max qubit
        let max_qubit = operators.iter().map(|op| op.qubit).max().unwrap_or(0);

        PauliProduct { operators,
                       parents: Vec::new(),
                       children: Vec::new(),
                       max_qubit,
                       id: product_id,
                       num_ys,
                       need_estabilizer: false,
                       need_ancilla: false,
                       is_clifford: false }
    }

    pub fn to_circuit_format(&self, num_qubits: usize) -> String {
        let mut rng = rand::thread_rng();
        let sign = if rng.gen_bool(0.5) { "+" } else { "-" };
        let mut pauli_string = vec!['_'; num_qubits];

        for op in &self.operators {
            pauli_string[op.qubit] = op.basis;
        }

        let angle = if self.is_clifford { "<M>" } else { "<pi/8>" };
        format!("{}{}{}", sign, pauli_string.iter().collect::<String>(), angle)
    }
}

impl fmt::Display for PauliProduct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ancilla_str = if self.num_ys % 2 == 1 { "A" } else { "-" };
        let es_str = if self.need_estabilizer { "E" } else { "-" };
        let clifford_str = if self.is_clifford { "clifford" } else { "non-clifford" };
        let ops = self.operators.iter().map(|op| op.to_string()).collect::<String>();

        write!(f,
               "{} {} {} {} {} {:?} {:?}",
               self.id, ops, ancilla_str, es_str, clifford_str, self.children, self.parents)
    }
}
