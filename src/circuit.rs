use crate::fn_timer;
use crate::pauliproduct::PauliProduct;
use plotters::coord::types::{RangedCoordf64, RangedCoordusize};
use plotters::prelude::*;
use std::fs::create_dir_all;
#[cfg(debug_assertions)]
use std::io::BufWriter;
use std::{
    cell::RefCell,
    fs::File,
    io::{self, BufRead, BufReader, Write},
    path::Path,
};

pub struct Circuit {
    products: Vec<PauliProduct>,
    layers: RefCell<Option<Vec<Vec<usize>>>>,
    pub circuit_fname: String,
    pub num_qubits: usize,
}

impl Circuit {
    pub fn new(fname: &String) -> Self {
        let circuit = Circuit { products: Vec::new(),
                                circuit_fname: fname.to_string(),
                                num_qubits: 0,
                                layers: RefCell::new(None) };
        circuit
    }

    pub fn load_circuit(&mut self) -> io::Result<()> {
        let _timer = fn_timer!();

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

        self.generate_dependencies();
        Ok(())
    }

    fn generate_dependencies(&mut self) {
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
    }

    pub fn generate_random(&mut self, num_products: usize, num_qubits: usize,
                           spread_probability: f64, decay_factor: f64) {
        for product_id in 0..num_products {
            let product = PauliProduct::generate_random(product_id as i32,
                                                        num_qubits,
                                                        spread_probability,
                                                        decay_factor);
            self.products.push(product);
        }
        // Find maximum qubit
        self.num_qubits = self.products.iter().map(|pp| pp.max_qubit).max().unwrap_or(0) + 1;

        println!("Generated random circuit with {} products and {} qubits",
                 self.products.len(),
                 self.num_qubits);
        self.generate_dependencies();
    }

    pub fn save_circuit_to_file(&self, circuit_fname: String) -> io::Result<()> {
        let _timer = fn_timer!();
        let mut file = File::create(&circuit_fname)?;

        for product in &self.products {
            let circuit_line = product.to_circuit_format(self.num_qubits);
            writeln!(file, "{}", circuit_line)?;
        }

        println!("Saved random circuit to {}", self.circuit_fname);
        Ok(())
    }

    pub fn initial_products(&self) -> impl Iterator<Item = &PauliProduct> {
        self.products.iter().filter(|pp| pp.parents.is_empty())
    }

    pub fn get_product(&self, id: i32) -> &PauliProduct {
        &self.products[id as usize]
    }

    pub fn num_products(&self) -> usize {
        self.products.len()
    }

    pub fn plot(&self, show_product_ids: bool) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        // Get circuit filename
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        let plot_dir = format!("{}.circuit", circuit_stem);
        create_dir_all(&plot_dir)?;

        // Create output files
        let layers = self.get_layers();
        let min_layer = 0;
        let max_layer = layers.len();
        // Split into chunks of 1000 layers
        const LAYERS_PER_FILE: usize = 1000;
        for chunk_start in (min_layer..max_layer).step_by(LAYERS_PER_FILE) {
            let chunk_end = (chunk_start + LAYERS_PER_FILE).min(max_layer);
            let chunk_layers = chunk_end - chunk_start;
            let plot_fname = format!("{}/{}-{}.png", plot_dir, circuit_stem, chunk_start);
            // Create drawing area
            let root = BitMapBackend::new(
                &plot_fname,
                (
                    (chunk_layers as f32 * 0.17 * 100.0) as u32,
                    (self.num_qubits as f32 * 0.22 * 100.0) as u32,
                ),
            )
            .into_drawing_area();
            /*
             let plot_fname = format!("{}/{}-{}.svg", plot_dir, circuit_stem, chunk_start);
             let root = SVGBackend::new(
                 &plot_fname,
                 (
                     (chunk_layers as f32 * 0.17 * 100.0) as u32,
                     (self.num_qubits as f32 * 0.22 * 100.0) as u32,
                 ),
             )
             .into_drawing_area();
            */
            root.fill(&WHITE)?;

            let mut chart =
                ChartBuilder::on(&root).margin(50)
                                       .set_label_area_size(LabelAreaPosition::Left, 60)
                                       .set_label_area_size(LabelAreaPosition::Bottom, 40)
                                       /*
                                       .caption(format!("{} (Layers {}-{})",
                                                        circuit_stem,
                                                        chunk_start,
                                                        chunk_end - 1),
                                                ("sans-serif", 20))*/
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
                 .x_desc("Layers")
                 .y_desc("Qubit Number")
                 .x_label_style(("sans-serif", 14))
                 .y_label_style(("sans-serif", 14))
                 .axis_desc_style(("sans-serif", 16))
                 .disable_mesh()
                 .draw()?;
            // Draw products
            for (col, layer) in layers[chunk_start..chunk_end].iter().enumerate() {
                // Sort products by their lowest qubit id
                let mut sorted_layer: Vec<&PauliProduct> = layer.clone();
                sorted_layer.sort_by_key(|pp| pp.get_qubits()[0]);
                for (i, pp) in sorted_layer.iter().enumerate() {
                    let col = col + chunk_start;
                    // Draw background rectangle
                    let start_pos = pp.get_qubits()[0];
                    let end_pos = *pp.get_qubits().last().unwrap();
                    let rect_height = (end_pos - start_pos) as f32 + 0.8;
                    let product_color = self.get_layer_product_color(i);

                    chart.draw_series(std::iter::once(Rectangle::new(
                        [
                            (col as f32 - 0.1, start_pos as f32 - 0.5),
                            (col as f32 + 0.7, start_pos as f32 + rect_height - 0.4),
                        ],
                        if pp.is_tgate {
                            //RGBColor(0x22, 0xFF, 0x22).mix(0.2).filled()
                            product_color.mix(0.2).filled()
                        } else {
                            RGBColor(0xCC, 0xCC, 0x22).mix(0.2).filled()
                        },
                    )))?;
                    // Add dark green outline
                    chart.draw_series(std::iter::once(Rectangle::new(
                        [
                            (col as f32 - 0.1, start_pos as f32 - 0.5),
                            (col as f32 + 0.7, start_pos as f32 + rect_height - 0.4),
                        ],
                        //RGBColor(0x00, 0x80, 0x00).stroke_width(1), // Dark green outline
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

    fn get_layer_product_color(&self, product_index: usize) -> RGBColor {
        // Predefined color palette for products within a layer
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

    pub fn plot_layer_stats(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        // Get circuit filename
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        let plot_dir = format!("{}.circuit", circuit_stem);
        create_dir_all(&plot_dir)?;
        // Get layer statistics
        //let layers = self.get_layers()[252000..255000].to_vec();
        let layers = self.get_layers();
        //let plot_fname = format!("{}.layer_stats.png", circuit_stem);
        //let root = BitMapBackend::new(&plot_fname, (1800, 1000)).into_drawing_area();
        let plot_fname = format!("{}.layer_stats.svg", circuit_stem);
        let root = SVGBackend::new(&plot_fname, (1800, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root).margin(60)
                                               .set_label_area_size(LabelAreaPosition::Left, 100)
                                               .set_label_area_size(LabelAreaPosition::Bottom, 100)
                                               //.caption(format!("{} Layer Statistics", circuit_stem),
                                               //        ("sans-serif", 36))
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

        //let mut window_size = 10;
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

        /*
        self.plot_moving_average(&mut chart,
                                 &layers,
                                 window_size,
                                 |window| {
                                     window.iter()
                                           .map(|layer| {
                                               layer.iter()
                                                    .map(|pp| pp.max_qubit + 1)
                                                    .max()
                                                    .unwrap_or(0)
                                           })
                                           .max()
                                           .unwrap_or(0) as f64
                                 },
                                 RGBColor(180, 0, 180),
                                 "max qubit")?;
         */
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

        /*
        self.plot_moving_average(&mut chart,
                                 &layers,
                                 window_size,
                                 |window| {
                                     window.iter().map(|layer| layer.len()).min().unwrap_or(0)
                                     as f64
                                 },
                                 RGBColor(115, 200, 0),
                                 "min products/layer")?;

                                 self.plot_moving_average(&mut chart,
                                 &layers,
                                 window_size,
                                 |window| {
                                     let sum: usize =
                                         window.iter()
                                               .map(|layer| {
                                                   layer.iter().filter(|pp| pp.need_ancilla).count()
                                               })
                                               .sum();
                                     sum as f64 / window.len() as f64
                                 },
                                 RGBColor(255, 165, 0),
                                 "avg ancilla reqd")?;

        self.plot_moving_average(&mut chart,
                                 &layers,
                                 window_size,
                                 |window| {
                                     let sum: usize =
                                         window.iter()
                                               .map(|layer| {
                                                   layer.iter()
                                                        .filter(|pp| pp.need_estabilizer)
                                                        .count()
                                               })
                                               .sum();
                                     sum as f64 / window.len() as f64
                                 },
                                 RGBColor(115, 200, 0),
                                 "avg e-stabilizers reqd")?;
        */
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
        /*
        self.plot_moving_average(&mut chart,
                                 &layers,
                                 window_size,
                                 |window| {
                                     window.iter()
                                           .map(|layer| {
                                               layer.iter()
                                                    .map(|pp| pp.operators.len())
                                                    .min()
                                                    .unwrap_or(0)
                                           })
                                           .min()
                                           .unwrap_or(0) as f64
                                 },
                                 RGBColor(0, 150, 0),
                                 "min product size")?;
         */
        chart.configure_series_labels()
             .margin(20)
             .background_style(&WHITE)
             .border_style(&TRANSPARENT)
             .position(SeriesLabelPosition::UpperRight)
             .label_font(("sans-serif", 40))
             .draw()?;

        /*
        let (num_layers,
             num_cliffords,
             avg_products,
             max_products,
             avg_ancillas,
             max_ancillas,
             avg_estabilizers,
             max_estabilizers) = self.get_statistics();

        let stats_text = format!("Circuit: {} products; {} Cliffords; {} layers; \
                                 products/layer {:.2} avg, {} max; \
                                 ancilla required/layer {:.2} avg, {} max; \
                                 e-stabilizers required/layer {:.2} avg, {} max \
                                 (window {})",
                                 self.products.len(),
                                 num_cliffords,
                                 num_layers,
                                 avg_products,
                                 max_products,
                                 avg_ancillas,
                                 max_ancillas,
                                 avg_estabilizers,
                                 max_estabilizers,
                                 window_size);

        // Draw statistics text below the plot
        root.draw(&Text::new(stats_text,
                             (10, 970), // Center horizontally, near bottom
                             ("sans-serif", 24).into_font()))?;
         */
        println!("Plotted layer statistics to {}", plot_fname);
        Ok(())
    }

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

    fn get_statistics(&self) -> (usize, i32, f64, i32) {
        let layers = self.get_layers();
        let mut num_cliffords = 0;
        let mut num_products = vec![0; layers.len()];

        for (i, layer) in layers.iter().enumerate() {
            num_products[i] += layer.len();
            for pp in layer {
                if !pp.is_tgate {
                    num_cliffords += 1;
                }
            }
        }
        let num_layers = layers.len();
        let avg_products = self.products.len() as f64 / num_layers as f64;
        let max_products = *num_products.iter().max().unwrap_or(&0);
        (layers.len(), num_cliffords, avg_products, max_products as i32)
    }

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

    fn get_layers(&self) -> Vec<Vec<&PauliProduct>> {
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

    pub fn plot_qubit_coupling(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        // Get circuit filename
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        // Build coupling matrix
        let coupling_matrix = self.build_coupling_matrix();
        //let coupling_matrix = self.build_pair_coupling_matrix();
        let dim = coupling_matrix.len();
        // Create surface plot
        let plot_fname = format!("{}.qubit_coupling.svg", circuit_stem);
        let root = SVGBackend::new(&plot_fname, (1200, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        // Find max value for color scaling
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
        // Draw heatmap squares
        for i in 0..dim {
            for j in 0..dim {
                let count = coupling_matrix[i][j];
                if count > 0 {
                    // Color intensity based on count (log scale for better visualization)
                    let intensity = if *max_count > 0 {
                        (count as f64).ln() / (*max_count as f64).ln()
                    } else {
                        0.0
                    };
                    // Use a color gradient from blue (low) to red (high)
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

    fn build_coupling_matrix(&self) -> Vec<Vec<usize>> {
        let mut matrix = vec![vec![0; self.num_qubits]; self.num_qubits];
        for product in &self.products {
            let qubits: Vec<usize> = product.get_qubits();
            // Count coupling for all pairs of qubits in this product
            for i in 0..qubits.len() {
                for j in 0..qubits.len() {
                    let qubit_i = qubits[i] / 2;
                    let qubit_j = qubits[j] / 2;
                    if qubit_i == qubit_j {
                        continue;
                    }
                    assert!(qubit_i < self.num_qubits && qubit_j < self.num_qubits);
                    matrix[qubit_i * 2][qubit_j * 2] += 1;
                    matrix[qubit_j * 2][qubit_i * 2] += 1; // Make matrix symmetric
                }
            }
        }
        self.print_coupling_frequency(&matrix);
        matrix
    }

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
