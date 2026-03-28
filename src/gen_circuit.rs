#![allow(dead_code)]
use clap::Parser;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs::File;
use std::io::{self, Write};

mod circuit;
mod pauliproduct;
#[macro_use]
mod utils;

use circuit::Circuit;
use pauliproduct::{GateType, Operator, PauliProduct};

impl Circuit {
    /// Generates a random T-gate circuit with spatial locality.
    /// Each product is generated with Pauli operators spreading from a center qubit.
    /// `spread_probability` controls spreading to adjacent qubits, decaying with `decay_factor`.
    /// `rng` is a caller-supplied seeded RNG so that results are reproducible.
    pub(crate) fn generate_random(
        &mut self, num_products: usize, num_qubits: usize, spread_probability: f64,
        decay_factor: f64, rng: &mut StdRng,
    ) {
        self.products.extend((0..num_products).map(|product_id| {
            PauliProduct::gen_rnd_t(
                product_id as i32,
                num_qubits,
                spread_probability,
                decay_factor,
                rng,
            )
        }));
        self.num_qubits =
            self.products.iter().map(|pp| pp.max_qubit as usize).max().unwrap_or(0) + 1;
        println!(
            "Generated random circuit with {} products and {} qubits",
            self.products.len(),
            self.num_qubits
        );
        self.generate_dependencies();
    }

    /// Writes all products to a circuit file in standard format.
    /// `rng` is used to assign a random sign (`+`/`-`) to each product line.
    pub(crate) fn save_circuit_to_file(
        &self, circuit_fname: String, rng: &mut StdRng,
    ) -> io::Result<()> {
        let _timer = fn_timer!();
        let mut file = File::create(&circuit_fname)?;
        for product in &self.products {
            let circuit_line = product.to_circuit_format(self.num_qubits, rng);
            writeln!(file, "{}", circuit_line)?;
        }
        println!("Saved circuit to {}", circuit_fname);
        Ok(())
    }
}

impl PauliProduct {
    /// Generates a random T-gate product with spatial locality.
    /// Starts at a random qubit and spreads to neighbors with decaying probability.
    /// `rng` is a caller-supplied seeded RNG so that results are reproducible.
    pub(crate) fn gen_rnd_t(
        product_id: i32, num_qubits: usize, spread_probability: f64, decay_factor: f64,
        rng: &mut StdRng,
    ) -> Self {
        let mut operators = Vec::new();
        let center_qubit = rng.gen_range(0..num_qubits);
        let center_basis = ['X', 'Y', 'Z'][rng.gen_range(0..3)];
        operators.push(Operator { qubit: center_qubit as u16, basis: center_basis });
        let mut current_prob = spread_probability;
        for distance in 1..=center_qubit {
            if rng.gen_range(0.0..1.0) < current_prob {
                let qubit = (center_qubit - distance) as u16;
                let basis = ['X', 'Y', 'Z'][rng.gen_range(0..3)];
                operators.push(Operator { qubit, basis });
            }
            current_prob *= decay_factor;
            if current_prob < 0.001 {
                break;
            }
        }
        current_prob = spread_probability;
        for distance in 1..(num_qubits - center_qubit) {
            if rng.gen_range(0.0..1.0) < current_prob {
                let qubit = (center_qubit + distance) as u16;
                let basis = ['X', 'Y', 'Z'][rng.gen_range(0..3)];
                operators.push(Operator { qubit, basis });
            }
            current_prob *= decay_factor;
            if current_prob < 0.001 {
                break;
            }
        }
        operators.sort_by_key(|op| op.qubit);
        let max_qubit = operators.iter().map(|op| op.qubit).max().unwrap_or(0);
        PauliProduct {
            operators,
            parents: Vec::new(),
            children: Vec::new(),
            max_qubit,
            id: product_id,
            gate_type: GateType::T,
        }
    }

    /// Converts this product to circuit file format with a random sign drawn from `rng`.
    pub(crate) fn to_circuit_format(&self, num_qubits: usize, rng: &mut StdRng) -> String {
        let sign = if rng.gen_bool(0.5) { "+" } else { "-" };
        let mut pauli_string = vec!['_'; num_qubits];
        for op in &self.operators {
            pauli_string[op.qubit as usize] = op.basis;
        }
        format!("{}{}<{:?}>", sign, pauli_string.iter().collect::<String>(), self.gate_type)
    }
}

/// Command-line arguments for random circuit generation.
#[derive(Parser, Debug)]
#[command(author, version, about = "Generate random T-gate circuits for PureMagic", long_about = None)]
struct Args {
    /// Random seed for reproducible results.
    #[arg(short, long, default_value = "29")]
    rseed: u32,
    /// Number of qubits for random circuit generation.
    #[arg(short = 'q', long, default_value = "64")]
    random_qubits: usize,
    /// Number of products for random circuit generation.
    #[arg(short = 'n', long, default_value = "1000")]
    random_products: usize,
    /// Spread probability: probability of adding operators to adjacent qubits (0.0–1.0).
    #[arg(short = 's', long, default_value = "0.4")]
    spread_probability: f64,
    /// Decay factor: how much probability decreases with distance (0.0–1.0).
    #[arg(short = 'd', long, default_value = "0.7")]
    decay_factor: f64,
    /// Output filename (default: auto-generated from parameters).
    #[arg(short, long)]
    output: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.spread_probability < 0.0 || args.spread_probability > 1.0 {
        eprintln!("Error: spread_probability must be between 0.0 and 1.0");
        std::process::exit(1);
    }
    if args.decay_factor < 0.0 || args.decay_factor > 1.0 {
        eprintln!("Error: decay_factor must be between 0.0 and 1.0");
        std::process::exit(1);
    }
    println!(
        "Generating random circuit with {} products on {} qubits",
        args.random_products, args.random_qubits
    );
    println!("  spread_probability: {}", args.spread_probability);
    println!("  decay_factor: {}", args.decay_factor);
    let spread_str = args.spread_probability.to_string().replace(".", "_");
    let decay_str = args.decay_factor.to_string().replace(".", "_");
    let fname = format!("random_circuit-{}-{}_n{}", spread_str, decay_str, args.random_qubits);
    let mut rng = StdRng::seed_from_u64(args.rseed as u64);
    let mut circuit = Circuit::new(&fname);
    circuit.generate_random(
        args.random_products,
        args.random_qubits,
        args.spread_probability,
        args.decay_factor,
        &mut rng,
    );
    let save_fname = args.output.unwrap_or_else(|| format!("{}.generated.txt", fname));
    circuit.save_circuit_to_file(save_fname, &mut rng)?;
    Ok(())
}
