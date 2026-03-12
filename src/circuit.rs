use crate::fn_timer;
use crate::pauliproduct::PauliProduct;
use plotters::coord::types::{RangedCoordf64, RangedCoordusize};
use plotters::prelude::*;
use std::fs::create_dir_all;
#[cfg(debug_assertions)]
use std::io::{BufWriter, Write};
use std::{
    cell::RefCell,
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};

/// Represents a quantum circuit as a DAG of Pauli products with dependency tracking.
/// Layers are lazily computed and cached for efficient iteration.
pub struct Circuit {
    pub(crate) products: Vec<PauliProduct>,
    layers: RefCell<Option<Vec<Vec<usize>>>>,
    pub circuit_fname: String,
    pub num_qubits: usize,
}

impl Circuit {
    /// Creates a new circuit from a filename (circuit is not loaded until `load_circuit()` is called).
    pub fn new(fname: &String) -> Self {
        let circuit = Circuit { products: Vec::new(),
                                circuit_fname: fname.to_string(),
                                num_qubits: 0,
                                layers: RefCell::new(None) };
        circuit
    }

    /// Loads Pauli products from file, skipping X and Z gates.
    /// Populates `num_qubits` and establishes parent-child dependencies.
    pub fn load_circuit(&mut self) -> io::Result<()> {
        let _timer = fn_timer!();

        let file = File::open(&self.circuit_fname)?;
        let reader = BufReader::new(file);
        let mut product_id: i32 = 0;
        for line in reader.lines() {
            let product_string = line?.trim().to_string();
            let mut product = PauliProduct::new();
            product.set_from_str(product_id, &product_string)
                   .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            if product.gate_type.is_x() || product.gate_type.is_z() {
                continue;
            }
            self.products.push(product);
            product_id += 1;
        }
        self.num_qubits =
            self.products.iter().map(|pp| pp.max_qubit as usize).max().unwrap_or(0) + 1;

        println!("Loaded circuit with {} products and {} qubits",
                 self.products.len(),
                 self.num_qubits);

        self.generate_dependencies();
        Ok(())
    }

    /// Establishes parent-child dependencies between products based on qubit operations.
    /// A product becomes a child of the last product to operate on each of its qubits.
    pub(crate) fn generate_dependencies(&mut self) {
        let mut relationships = Vec::new();
        let mut current_pps = vec![-1; self.num_qubits];

        for pp in self.products.iter() {
            for op in &pp.operators {
                let current_id = current_pps[op.qubit as usize];
                if current_id != -1 {
                    relationships.push((pp.id as i32, current_id));
                }
                current_pps[op.qubit as usize] = pp.id as i32;
            }
        }
        for (child_id, parent_id) in relationships {
            self.products[child_id as usize].parents.push(parent_id);
            self.products[parent_id as usize].children.push(child_id);
        }
    }

    /// Returns an iterator over products with no dependencies (ready to schedule).
    pub fn initial_products(&self) -> impl Iterator<Item = &PauliProduct> {
        self.products.iter().filter(|pp| pp.parents.is_empty())
    }

    /// Retrieves a product by its ID.
    pub fn get_product(&self, id: i32) -> &PauliProduct {
        &self.products[id as usize]
    }

    /// Returns the total number of products in the circuit.
    pub fn num_products(&self) -> usize {
        self.products.len()
    }

    /// Plots the circuit structure as PNG files (split into 1000-layer chunks).
    /// Each product is colored and labeled with Pauli operators.
    pub fn plot(&self, show_product_ids: bool) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        let plot_dir = format!("{}.circuit", circuit_stem);
        create_dir_all(&plot_dir)?;

        let layers = self.get_layers();
        let min_layer = 0;
        let max_layer = layers.len();
        const LAYERS_PER_FILE: usize = 1000;
        for chunk_start in (min_layer..max_layer).step_by(LAYERS_PER_FILE) {
            let chunk_end = (chunk_start + LAYERS_PER_FILE).min(max_layer);
            let chunk_layers = chunk_end - chunk_start;
            let plot_fname = format!("{}/{}-{}.png", plot_dir, circuit_stem, chunk_start);
            let root = BitMapBackend::new(
                &plot_fname,
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
                                       .build_cartesian_2d(chunk_start as f32..chunk_end as f32,
                                                           ((self.num_qubits - 1) as f32 + 0.5)
                                                           ..(-0.5f32))?;
            chart.configure_mesh()
                 .x_labels(chunk_layers / 5)
                 .x_label_formatter(&|x| format!("{}", x))
                 .y_labels((self.num_qubits + 1) as usize)
                 .y_label_formatter(&|y| format!("{}", y))
                 .x_desc("Layers")
                 .y_desc("Qubit Number")
                 .x_label_style(("sans-serif", 14))
                 .y_label_style(("sans-serif", 14))
                 .axis_desc_style(("sans-serif", 16))
                 .disable_mesh()
                 .draw()?;
            for (col, layer) in layers[chunk_start..chunk_end].iter().enumerate() {
                let mut sorted_layer: Vec<&PauliProduct> = layer.clone();
                sorted_layer.sort_by_key(|pp| pp.get_qubits()[0]);
                for (i, pp) in sorted_layer.iter().enumerate() {
                    let col = col + chunk_start;
                    let start_pos = pp.get_qubits()[0];
                    let end_pos = *pp.get_qubits().last().unwrap();
                    let rect_height = (end_pos - start_pos) as f32 + 0.8;
                    let product_color = self.get_layer_product_color(i);

                    chart.draw_series(std::iter::once(Rectangle::new(
                        [
                            (col as f32 - 0.1, start_pos as f32 - 0.5),
                            (col as f32 + 0.7, start_pos as f32 + rect_height - 0.4),
                        ],
                        if pp.gate_type.is_t() {
                            product_color.mix(0.2).filled()
                        } else {
                            RGBColor(0xCC, 0xCC, 0x22).mix(0.2).filled()
                        },
                    )))?;
                    chart.draw_series(std::iter::once(Rectangle::new(
                        [
                            (col as f32 - 0.1, start_pos as f32 - 0.5),
                            (col as f32 + 0.7, start_pos as f32 + rect_height - 0.4),
                        ],
                        product_color.stroke_width(1),
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
            println!("Plotted circuit layers {}-{} to {}", chunk_start, chunk_end - 1, plot_fname);
        }
        Ok(())
    }

    /// Returns a color for a product based on its index within a layer.
    fn get_layer_product_color(&self, product_index: usize) -> RGBColor {
        let colors = [
            RGBColor(255, 100, 100), // Light red
            RGBColor(100, 255, 100), // Light green
            RGBColor(100, 100, 255), // Light blue
            RGBColor(255, 255, 100), // Light yellow
            RGBColor(255, 100, 255), // Light magenta
            RGBColor(100, 255, 255), // Light cyan
            RGBColor(255, 150, 100), // Light orange
            RGBColor(150, 100, 255), // Light purple
            RGBColor(100, 255, 150), // Light mint
            RGBColor(255, 100, 150), // Light pink
            RGBColor(150, 255, 100), // Light lime
            RGBColor(100, 150, 255), // Light sky blue
            RGBColor(200, 200, 100), // Light olive
            RGBColor(200, 100, 200), // Light violet
            RGBColor(100, 200, 200), // Light teal
            RGBColor(255, 200, 100), // Light peach
            RGBColor(200, 255, 100), // Light chartreuse
            RGBColor(100, 200, 255), // Light cornflower
            RGBColor(255, 150, 150), // Light coral
            RGBColor(150, 255, 150), // Light seafoam
        ];
        colors[product_index % colors.len()]
    }

    /// Plots moving average statistics of layer properties (products per layer, product size, etc.).
    /// Generates an SVG file with configurable window size based on circuit size.
    pub fn plot_layer_stats(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        let plot_dir = format!("{}.circuit", circuit_stem);
        create_dir_all(&plot_dir)?;
        let layers = self.get_layers();
        let plot_fname = format!("{}.layer_stats.svg", circuit_stem);
        let root = SVGBackend::new(&plot_fname, (1800, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root).margin(60)
                                               .set_label_area_size(LabelAreaPosition::Left, 100)
                                               .set_label_area_size(LabelAreaPosition::Bottom, 100)
                                               .build_cartesian_2d(0..layers.len(),
                                                                   0.0f64..self.num_qubits as f64)?;
        chart.configure_mesh()
             .x_labels(20)
             .x_label_formatter(&|x| format!("{}", x))
             .y_labels(10)
             .y_label_formatter(&|y| format!("{}", y))
             .x_desc("Layer")
             .y_desc("Statistic")
             .x_label_style(("sans-serif", 40))
             .y_label_style(("sans-serif", 40))
             .axis_desc_style(("sans-serif", 48))
             .light_line_style(&TRANSPARENT)
             .draw()?;

        let mut window_size = 200;
        if layers.len() < 2000 {
            window_size = 10;
        } else if layers.len() < 5000 {
            window_size = 20;
        } else if layers.len() < 50000 {
            window_size = 100;
        } else if layers.len() < 20000 {
            window_size = 150;
        }

        self.plot_moving_average(&mut chart,
                                 &layers,
                                 window_size,
                                 |window| {
                                     let sum: usize = window.iter().map(|layer| layer.len()).sum();
                                     sum as f64 / window.len() as f64
                                 },
                                 RGBColor(0, 0, 255), // Blue
                                 "avg products/layer")?;

        self.plot_moving_average(&mut chart,
                                 &layers,
                                 window_size,
                                 |window| {
                                     window.iter().map(|layer| layer.len()).max().unwrap_or(0)
                                     as f64
                                 },
                                 RGBColor(255, 165, 0),
                                 "max products/layer")?;

        self.plot_moving_average(&mut chart,
                                 &layers,
                                 window_size,
                                 |window| {
                                     let (total_ops, total_products): (usize, usize) =
                                         window.iter()
                                               .map(|layer| {
                                                   let ops: usize =
                                                       layer.iter()
                                                            .map(|pp| pp.operators.len())
                                                            .sum();
                                                   (ops, layer.len())
                                               })
                                               .fold((0, 0),
                                                     |(acc_ops, acc_prods), (ops, prods)| {
                                                         (acc_ops + ops, acc_prods + prods)
                                                     });

                                     if total_products > 0 {
                                         total_ops as f64 / total_products as f64
                                     } else {
                                         0.0
                                     }
                                 },
                                 RGBColor(255, 0, 0), // Red
                                 "avg product size")?;

        self.plot_moving_average(&mut chart,
                                 &layers,
                                 window_size,
                                 |window| {
                                     window.iter()
                                           .map(|layer| {
                                               layer.iter()
                                                    .map(|pp| pp.operators.len())
                                                    .max()
                                                    .unwrap_or(0)
                                           })
                                           .max()
                                           .unwrap_or(0) as f64
                                 },
                                 RGBColor(0, 200, 200),
                                 "max product size")?;
        chart.configure_series_labels()
             .margin(20)
             .background_style(&WHITE)
             .border_style(&TRANSPARENT)
             .position(SeriesLabelPosition::UpperRight)
             .label_font(("sans-serif", 40))
             .draw()?;

        println!("Plotted layer statistics to {}", plot_fname);
        Ok(())
    }

    /// Plots a metric computed over a moving window of layers.
    fn plot_moving_average<F>(&self,
                              //chart: &mut ChartContext<BitMapBackend,
                              chart: &mut ChartContext<SVGBackend,
                                                Cartesian2d<RangedCoordusize,
                                                            RangedCoordf64>>,
                              layers: &[Vec<&PauliProduct>],
                              window_size: usize,
                              value_fn: F,
                              color: RGBColor,
                              label: &str)
                              -> Result<(), Box<dyn std::error::Error>>
        where F: Fn(&[Vec<&PauliProduct>]) -> f64
    {
        let data = self.compute_moving_average(layers, window_size, value_fn);
        // Assert that no y value exceeds num_qubits
        for (i, &y_value) in data.iter().enumerate() {
            assert!(y_value <= self.num_qubits as f64,
                    "Y value {} at index {} exceeds num_qubits {} for metric '{}'",
                    y_value,
                    i,
                    self.num_qubits,
                    label);
        }
        chart.draw_series(LineSeries::new(data.iter().enumerate().map(|(x, &y)| (x, y)),
                                          color.mix(0.8).stroke_width(2)))?
             .label(label)
             .legend(move |(x, y)| {
                 PathElement::new(vec![(x, y), (x + 20, y)], color.mix(0.8).stroke_width(2))
             });

        Ok(())
    }

    /// Computes moving average of a metric across layers using a sliding window.
    fn compute_moving_average<F>(&self, layers: &[Vec<&PauliProduct>], window_size: usize,
                                 value_fn: F)
                                 -> Vec<f64>
        where F: Fn(&[Vec<&PauliProduct>]) -> f64
    {
        layers.iter()
              .enumerate()
              .map(|(i, _)| {
                  let window_start = if i >= window_size { i - window_size } else { 0 };
                  let window_end = i + 1;
                  let window = &layers[window_start..window_end];
                  value_fn(window)
              })
              .collect()
    }

    /// Prints circuit statistics (products, Cliffords, layers, avg/max products per layer).
    /// Returns the total number of layers.
    pub fn print_statistics(&self) -> usize {
        let (num_layers, num_cliffords, avg_products, max_products) = self.get_statistics();
        println!("Circuit statistics:");
        println!("  Number of products:               {}", self.products.len());
        println!("  Number of Cliffords:              {}", num_cliffords);
        println!("  Layers:                           {}", num_layers);
        println!("  Products per layer:               {:.2} avg, {} max",
                 avg_products, max_products);
        num_layers
    }

    /// Computes circuit statistics: layer count, Clifford count, avg/max products per layer.
    fn get_statistics(&self) -> (usize, i32, f64, i32) {
        let layers = self.get_layers();
        let mut num_cliffords = 0;
        let mut num_products = vec![0; layers.len()];

        for (i, layer) in layers.iter().enumerate() {
            num_products[i] += layer.len();
            for pp in layer {
                if pp.gate_type.is_clifford() {
                    num_cliffords += 1;
                }
            }
        }
        let num_layers = layers.len();
        let avg_products = self.products.len() as f64 / num_layers as f64;
        let max_products = *num_products.iter().max().unwrap_or(&0);
        (layers.len(), num_cliffords, avg_products, max_products as i32)
    }

    /// Writes circuit layers to a text file (debug builds only).
    #[cfg(debug_assertions)]
    pub fn print(&self) -> io::Result<()> {
        let _timer = fn_timer!();
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        let output_fname = format!("{}.circuit.txt", circuit_stem);
        let file = File::create(&output_fname)?;
        let mut buf_file = BufWriter::new(file);

        let layers = self.get_layers();

        writeln!(buf_file, "layer id product ancilla? ES? clifford? children parents")?;
        for (i, layer) in layers.iter().enumerate() {
            let mut sorted_layer = layer.clone();
            sorted_layer.sort_by_key(|pp| pp.id);
            for pp in sorted_layer {
                writeln!(buf_file, "{}: {}", i, pp)?;
            }
        }
        println!("Wrote circuit to {}", output_fname);
        Ok(())
    }

    /// Computes circuit layers using topological sort (cached after first call).
    /// Returns products grouped by their topological level (layer).
    fn get_layers(&self) -> Vec<Vec<&PauliProduct>> {
        if let Some(cached) = self.layers.borrow().as_ref() {
            return cached.iter()
                         .map(|layer| layer.iter().map(|&idx| &self.products[idx]).collect())
                         .collect();
        }
        let mut in_degrees: Vec<usize> = self.products.iter().map(|pp| pp.parents.len()).collect();
        let mut ready: Vec<usize> = in_degrees.iter()
                                              .enumerate()
                                              .filter(|&(_, &degree)| degree == 0)
                                              .map(|(idx, _)| idx)
                                              .collect();

        let mut index_layers = Vec::new();
        let mut processed = 0;
        while !ready.is_empty() {
            index_layers.push(ready.clone());
            processed += ready.len();
            let mut next_ready = Vec::new();
            for &current in &ready {
                for &child_id in &self.products[current].children {
                    let child_idx = child_id as usize;
                    in_degrees[child_idx] -= 1;
                    if in_degrees[child_idx] == 0 {
                        next_ready.push(child_idx);
                    }
                }
            }
            ready = next_ready;
        }
        assert_eq!(processed,
                   self.products.len(),
                   "Circuit contains cycles or unreachable products");
        *self.layers.borrow_mut() = Some(index_layers.clone());
        index_layers.iter()
                    .map(|layer| layer.iter().map(|&idx| &self.products[idx]).collect())
                    .collect()
    }

    /// Plots a heatmap of qubit coupling frequency (which pairs of qubits interact).
    /// Uses log-scale intensity to highlight frequently coupled pairs.
    pub fn plot_qubit_coupling(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        let coupling_matrix = self.build_coupling_matrix();
        let dim = coupling_matrix.len();
        let plot_fname = format!("{}.qubit_coupling.svg", circuit_stem);
        let root = SVGBackend::new(&plot_fname, (1200, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let max_count = coupling_matrix.iter().flat_map(|row| row.iter()).max().unwrap_or(&0);
        let mut chart =
            ChartBuilder::on(&root).margin(60)
                                   .set_label_area_size(LabelAreaPosition::Left, 60)
                                   .set_label_area_size(LabelAreaPosition::Bottom, 60)
                                   .set_label_area_size(LabelAreaPosition::Right, 150)
                                   .caption(format!("{} - Qubit Coupling Matrix", circuit_stem),
                                            ("sans-serif", 24))
                                   .build_cartesian_2d(0..dim, 0..dim)?;
        chart.configure_mesh()
             .x_labels(10)
             .y_labels(10)
             .x_desc("Qubit Index")
             .y_desc("Qubit Index")
             .x_label_style(("sans-serif", 16))
             .y_label_style(("sans-serif", 16))
             .axis_desc_style(("sans-serif", 18))
             .draw()?;
        for i in 0..dim {
            for j in 0..dim {
                let count = coupling_matrix[i][j];
                if count > 0 {
                    let intensity = if *max_count > 0 {
                        (count as f64).ln() / (*max_count as f64).ln()
                    } else {
                        0.0
                    };
                    let color = if intensity < 0.5 {
                        let blue_intensity = (255.0 * (1.0 - 2.0 * intensity)) as u8;
                        let green_intensity = (255.0 * 2.0 * intensity) as u8;
                        RGBColor(0, green_intensity, blue_intensity)
                    } else {
                        let red_intensity = (255.0 * (2.0 * intensity - 1.0)) as u8;
                        let green_intensity = (255.0 * (2.0 - 2.0 * intensity)) as u8;
                        RGBColor(red_intensity, green_intensity, 0)
                    };
                    chart.draw_series(std::iter::once(Rectangle::new([(i, j), (i + 1, j + 1)],
                                                                     color.filled())))?;
                }
            }
        }
        println!("Plotted qubit coupling matrix to {}", plot_fname);
        Ok(())
    }

    /// Builds a qubit coupling matrix: counts how many products couple each qubit pair.
    fn build_coupling_matrix(&self) -> Vec<Vec<usize>> {
        let mut matrix = vec![vec![0; self.num_qubits]; self.num_qubits];
        for product in &self.products {
            let qubits: Vec<u16> = product.get_qubits();
            for i in 0..qubits.len() {
                for j in 0..qubits.len() {
                    let qubit_i = qubits[i] / 2;
                    let qubit_j = qubits[j] / 2;
                    if qubit_i == qubit_j {
                        continue;
                    }
                    matrix[qubit_i as usize * 2][qubit_j as usize * 2] += 1;
                    matrix[qubit_j as usize * 2][qubit_i as usize * 2] += 1;
                }
            }
        }
        self.print_coupling_frequency(&matrix);
        matrix
    }

    /// Prints qubit coupling frequency statistics in descending order.
    fn print_coupling_frequency(&self, coupling_matrix: &Vec<Vec<usize>>) {
        let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
        for i in 0..(self.num_qubits - 1) {
            for j in (i + 1)..self.num_qubits {
                pairs.push((i, j, coupling_matrix[i][j]));
            }
        }
        pairs.sort_by(|p1, p2| p1.2.cmp(&p2.2).reverse());
        eprintln!("Pair frequencies:");
        for (q1, q2, n) in pairs {
            if n != 0 {
                eprintln!("  {} {} {}", q1, q2, n);
            }
        }
    }
}
