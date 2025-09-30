use crate::pauliproduct::{Operator, PauliProduct};
use crate::utils::Timer;

use plotters::prelude::*;
use std::{
    cell::RefCell,
    fs::File,
    io::{self, BufRead, BufReader, Write},
    path::Path,
};

pub struct Circuit {
    pub products: Vec<PauliProduct>,
    pub circuit_fname: String,
    pub num_qubits: usize,
    layers: RefCell<Option<Vec<Vec<usize>>>>,
}

impl Circuit {
    pub fn new(fname: &String) -> io::Result<Self> {
        let mut circuit = Circuit { products: Vec::new(),
                                    circuit_fname: fname.to_string(),
                                    num_qubits: 0,
                                    layers: RefCell::new(None) };
        circuit.load_circuit()?;
        Ok(circuit)
    }

    fn load_circuit(&mut self) -> io::Result<()> {
        let _timer = Timer::new("load_circuit");

        let file = File::open(&self.circuit_fname)?;
        let reader = BufReader::new(file);

        // Read and parse products
        for (i, line) in reader.lines().enumerate() {
            let product_string = line?.trim().to_string();
            let mut product = PauliProduct::new();
            product.set_from_str(i as i32, &product_string)
                   .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            self.products.push(product);
        }

        // Find maximum qubit
        self.num_qubits = self.products.iter().map(|pp| pp.max_qubit).max().unwrap_or(0) + 1;

        println!("Loaded circuit with {} products and {} qubits",
                 self.products.len(),
                 self.num_qubits);

        // Collect parent/child relationships
        let mut relationships = Vec::new();
        let mut current_pps = vec![-1; self.num_qubits];

        for pp in self.products.iter() {
            for op in &pp.operators {
                let current_id = current_pps[op.qubit];
                if current_id != -1 {
                    relationships.push((pp.id as i32, current_id));
                }
                current_pps[op.qubit] = pp.id as i32;
            }
        }
        // Apply relationships in batch
        for (child_id, parent_id) in relationships {
            self.products[child_id as usize].parents.push(parent_id);
            self.products[parent_id as usize].children.push(child_id);
        }

        Ok(())
    }

    pub fn split_ys(&mut self) {
        let mut modifications = Vec::new();
        let start_id = self.products.len() as i32;
        // split Ys and gather changes for original PPs in modifications vector
        for pp in self.products.iter() {
            if pp.num_ys > 0 {
                let mut new_pp = PauliProduct::new();
                new_pp.id = start_id + modifications.len() as i32;
                new_pp.is_clifford = pp.is_clifford;
                new_pp.num_ys = pp.num_ys;
                new_pp.need_ancilla = pp.num_ys % 2 == 1;
                new_pp.need_estabilizer = true;
                // Convert Y operators to X and Z parts
                let mut pp_operators_updated = Vec::new();
                for op in &pp.operators {
                    match op.basis {
                        'X' => pp_operators_updated.push(op.clone()),
                        'Y' => {
                            pp_operators_updated.push(Operator { qubit: op.qubit, basis: 'x' });
                            new_pp.operators.push(Operator { qubit: op.qubit, basis: 'z' });
                        }
                        'Z' => new_pp.operators.push(op.clone()),
                        _ => {}
                    }
                }
                new_pp.children = pp.children.clone();
                new_pp.parents.push(pp.id);
                modifications.push((pp.id, pp_operators_updated, new_pp));
            }
        }
        let modifications_len = modifications.len();
        // update original products from modification vector information, and add new products
        for (pp_id, pp_operators_updated, new_pp) in modifications {
            // Update original product
            let pp = &mut self.products[pp_id as usize];
            pp.operators = pp_operators_updated;
            pp.children = vec![new_pp.id];
            for child_id in new_pp.children.iter() {
                let pp_child = &mut self.products[*child_id as usize];
                if let Some(pos) = pp_child.parents.iter().position(|&x| x == pp_id) {
                    pp_child.parents[pos] = new_pp.id;
                }
            }
            // Add the new product
            self.products.push(new_pp);
        }

        println!("After splitting {} Y products there are {} products in the circuit",
                 modifications_len,
                 self.products.len());
    }

    pub fn plot(&self, show_product_ids: bool) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = Timer::new("plot");
        // Get circuit filename
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        // Create output files
        let layers = self.get_layers();
        let min_layer = 0;
        let max_layer = layers.len();
        // Split into chunks of 1000 layers
        const LAYERS_PER_FILE: usize = 1000;
        for chunk_start in (min_layer..max_layer).step_by(LAYERS_PER_FILE) {
            let chunk_end = (chunk_start + LAYERS_PER_FILE).min(max_layer);
            let chunk_layers = chunk_end - chunk_start;
            let png_fname = format!("{}.circuit-{}.png", circuit_stem, chunk_start);
            // Create drawing area
            let root = BitMapBackend::new(
                &png_fname,
                (
                    (chunk_layers as f32 * 0.17 * 100.0) as u32,
                    (self.num_qubits as f32 * 0.22 * 100.0) as u32,
                ),
            )
            .into_drawing_area();

            root.fill(&WHITE)?;

            let mut chart =
                ChartBuilder::on(&root).margin(50)
                                       .set_label_area_size(LabelAreaPosition::Left, 60)
                                       .set_label_area_size(LabelAreaPosition::Bottom, 40)
                                       .caption(format!("{} (Layers {}-{})",
                                                        circuit_stem,
                                                        chunk_start,
                                                        chunk_end - 1),
                                                ("sans-serif", 20))
                                       .build_cartesian_2d(chunk_start as f32..chunk_end as f32,
                                                           //-0.5f32..self.num_qubits as f32 + 0.5,
                                                           ((self.num_qubits - 1) as f32 + 0.5)
                                                           ..(-0.5f32))?;
            // Configure axes
            chart.configure_mesh()
                 .x_labels(chunk_layers / 5)
                 .x_label_formatter(&|x| format!("{}", x))
                 .y_labels((self.num_qubits + 1) as usize)
                 .y_label_formatter(&|y| format!("{}", y))
                 .x_desc("Time Steps")
                 .y_desc("Qubits")
                 .x_label_style(("sans-serif", 14))
                 .y_label_style(("sans-serif", 14))
                 .axis_desc_style(("sans-serif", 16))
                 .disable_mesh()
                 .draw()?;
            // Draw products
            for (col, layer) in layers[chunk_start..chunk_end].iter().enumerate() {
                for pp in layer {
                    let col = col + chunk_start;
                    // Draw background rectangle
                    let start_pos = pp.get_qubits()[0];
                    let end_pos = *pp.get_qubits().last().unwrap();
                    let rect_height = (end_pos - start_pos) as f32 + 0.8;

                    chart.draw_series(std::iter::once(Rectangle::new(
                        [
                            (col as f32 - 0.1, start_pos as f32 - 0.4),
                            (col as f32 + 0.7, start_pos as f32 + rect_height - 0.4),
                        ],
                        if pp.is_clifford {
                            RGBColor(0xCC, 0xCC, 0x22).filled()
                        } else {
                            RGBColor(0x22, 0xFF, 0x22).filled()
                        },
                    )))?;
                    if show_product_ids {
                        chart.draw_series(std::iter::once(Text::new(
                            pp.id.to_string(),
                            (col as f32, pp.get_qubits()[0] as f32 - 0.15),
                            ("monospace", 8).into_font().transform(FontTransform::Rotate90),
                        )))?;
                    } else {
                        for op in &pp.operators {
                            if op.basis != ' ' {
                                chart.draw_series(std::iter::once(Text::new(
                                    op.basis.to_string(),
                                    (col as f32 + 0.1, op.qubit as f32 - 0.4),
                                    ("monospace", 10).into_font(),
                                )))?;
                            }
                        }
                    }
                }
            }
            println!("Saved layers {}-{} to {}", chunk_start, chunk_end - 1, png_fname);
        }
        Ok(())
    }

    pub fn get_statistics(&self) -> usize {
        let layers = self.get_layers();
        let mut num_noncliffords = vec![0; layers.len()];
        let mut num_odd_ys = vec![0; layers.len()];
        let mut num_ys = vec![0; layers.len()];
        let mut num_nonclifford_layers = 0;

        for (i, layer) in layers.iter().enumerate() {
            let mut nonclifford_layer = false;
            for pp in layer {
                if !pp.is_clifford {
                    num_noncliffords[i] += 1;
                    nonclifford_layer = true;
                }
                if pp.num_ys > 0 {
                    num_ys[i] += 1;
                    if pp.num_ys % 2 == 1 {
                        num_odd_ys[i] += 1;
                    }
                }
            }
            if nonclifford_layer {
                num_nonclifford_layers += 1;
            }
        }

        println!("Circuit statistics:");
        println!("  Layers:                  {}", layers.len());
        println!("  Non-Clifford layers:     {}", num_nonclifford_layers);
        println!("  Non-Cliffords per layer:  {:.2} avg, {} max",
                 num_noncliffords.iter().sum::<i32>() as f64 / layers.len() as f64,
                 *num_noncliffords.iter().max().unwrap_or(&0));
        println!("  Odd Y products per layer: {:.2} avg, {} max",
                 num_odd_ys.iter().sum::<i32>() as f64 / layers.len() as f64,
                 *num_odd_ys.iter().max().unwrap_or(&0));
        println!("  Y products per layer:     {:.2} avg, {} max",
                 num_ys.iter().sum::<i32>() as f64 / layers.len() as f64,
                 *num_ys.iter().max().unwrap_or(&0));

        layers.len()
    }

    pub fn print(&self) -> io::Result<()> {
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        let output_fname = format!("{}.circuit.txt", circuit_stem);
        let mut file = File::create(&output_fname)?;

        let layers = self.get_layers();

        writeln!(file, "layer id product ancilla? ES? clifford? children parents")?;
        for (i, layer) in layers.iter().enumerate() {
            let mut sorted_layer = layer.clone();
            sorted_layer.sort_by_key(|pp| pp.id);
            for pp in sorted_layer {
                writeln!(file, "{}: {}", i, pp)?;
            }
        }
        println!("Wrote circuit to {}", output_fname);
        Ok(())
    }

    fn get_layers(&self) -> Vec<Vec<&PauliProduct>> {
        let _timer = Timer::new("get_layers");
        // Return cached layers if available
        if let Some(cached) = self.layers.borrow().as_ref() {
            return cached.iter()
                         .map(|layer| layer.iter().map(|&idx| &self.products[idx]).collect())
                         .collect();
        }
        // Pre-calculate in-degrees (number of unprocessed parents) for each product
        let mut in_degrees: Vec<usize> = self.products.iter().map(|pp| pp.parents.len()).collect();
        // Keep track of products ready to be processed (those with no remaining parents)
        let mut ready: Vec<usize> = in_degrees.iter()
                                              .enumerate()
                                              .filter(|&(_, &degree)| degree == 0)
                                              .map(|(idx, _)| idx)
                                              .collect();

        let mut index_layers = Vec::new();
        let mut processed = 0;
        // Process products level by level
        while !ready.is_empty() {
            // Current layer contains all ready products
            index_layers.push(ready.clone());
            processed += ready.len();
            // Find products that become ready after processing current layer
            let mut next_ready = Vec::new();
            for &current in &ready {
                // Decrease in-degree for all children
                for &child_id in &self.products[current].children {
                    let child_idx = child_id as usize;
                    in_degrees[child_idx] -= 1;
                    // If all parents processed, product becomes ready
                    if in_degrees[child_idx] == 0 {
                        next_ready.push(child_idx);
                    }
                }
            }
            ready = next_ready;
        }
        // Verify all products were processed
        debug_assert_eq!(processed,
                         self.products.len(),
                         "Circuit contains cycles or unreachable products");
        // Cache the computed layers
        *self.layers.borrow_mut() = Some(index_layers.clone());
        // Convert indices to references
        index_layers.iter()
                    .map(|layer| layer.iter().map(|&idx| &self.products[idx]).collect())
                    .collect()
    }
}
