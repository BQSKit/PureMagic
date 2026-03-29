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

/// A quantum circuit as a DAG of Pauli products with dependency tracking.
///
/// Products are ordered by their circuit-file position; dependencies are derived
/// from qubit overlap (the last product to touch a qubit becomes the parent of
/// the next one on that qubit). Layers are lazily computed via topological sort
/// and cached after the first call to avoid repeated recomputation.
pub(crate) struct Circuit {
    pub(crate) products: Vec<PauliProduct>,
    layers: RefCell<Option<Vec<Vec<usize>>>>,
    pub(crate) circuit_fname: String,
    pub(crate) num_qubits: usize,
}

impl Circuit {
    pub(crate) fn new(fname: &String) -> Self {
        let circuit = Circuit {
            products: Vec::new(),
            circuit_fname: fname.to_string(),
            num_qubits: 0,
            layers: RefCell::new(None),
        };
        circuit
    }

    /// Loads Pauli products from file, skipping X and Z gates.
    /// X and Z are Pauli corrections that are tracked classically in the
    /// Pauli frame and do not require physical operations on the layout.
    pub(crate) fn load_circuit(&mut self) -> io::Result<()> {
        let _timer = fn_timer!();

        let file = File::open(&self.circuit_fname)?;
        let reader = BufReader::new(file);
        let mut product_id: i32 = 0;
        for line in reader.lines() {
            let product_string = line?.trim().to_string();
            let mut product = PauliProduct::new();
            product
                .set_from_str(product_id, &product_string)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            if product.gate_type.is_x() || product.gate_type.is_z() {
                continue;
            }
            self.products.push(product);
            product_id += 1;
        }
        self.num_qubits =
            self.products.iter().map(|pp| pp.max_qubit as usize).max().unwrap_or(0) + 1;

        println!(
            "Loaded circuit with {} products and {} qubits",
            self.products.len(),
            self.num_qubits
        );

        self.generate_dependencies();
        Ok(())
    }

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

    pub(crate) fn initial_products(&self) -> impl Iterator<Item = &PauliProduct> {
        self.products.iter().filter(|pp| pp.parents.is_empty())
    }

    pub(crate) fn get_product(&self, id: i32) -> &PauliProduct {
        &self.products[id as usize]
    }

    pub(crate) fn num_products(&self) -> usize {
        self.products.len()
    }

    pub(crate) fn plot(&self, show_product_ids: bool) -> Result<(), Box<dyn std::error::Error>> {
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

            let mut chart = ChartBuilder::on(&root)
                .margin(50)
                .set_label_area_size(LabelAreaPosition::Left, 60)
                .set_label_area_size(LabelAreaPosition::Bottom, 40)
                .build_cartesian_2d(
                    chunk_start as f32..chunk_end as f32,
                    ((self.num_qubits - 1) as f32 + 0.5)..(-0.5f32),
                )?;
            chart
                .configure_mesh()
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

    fn get_layer_product_color(&self, product_index: usize) -> RGBColor {
        let colors = [
            RGBColor(255, 100, 100),
            RGBColor(100, 255, 100),
            RGBColor(100, 100, 255),
            RGBColor(255, 255, 100),
            RGBColor(255, 100, 255),
            RGBColor(100, 255, 255),
            RGBColor(255, 150, 100),
            RGBColor(150, 100, 255),
            RGBColor(100, 255, 150),
            RGBColor(255, 100, 150),
            RGBColor(150, 255, 100),
            RGBColor(100, 150, 255),
            RGBColor(200, 200, 100),
            RGBColor(200, 100, 200),
            RGBColor(100, 200, 200),
            RGBColor(255, 200, 100),
            RGBColor(200, 255, 100),
            RGBColor(100, 200, 255),
            RGBColor(255, 150, 150),
            RGBColor(150, 255, 150),
        ];
        colors[product_index % colors.len()]
    }

    pub(crate) fn plot_layer_stats(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        let plot_dir = format!("{}.circuit", circuit_stem);
        create_dir_all(&plot_dir)?;
        let layers = self.get_layers();
        let plot_fname = format!("{}.layer_stats.svg", circuit_stem);
        let root = SVGBackend::new(&plot_fname, (1800, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .margin(60)
            .set_label_area_size(LabelAreaPosition::Left, 100)
            .set_label_area_size(LabelAreaPosition::Bottom, 100)
            .build_cartesian_2d(0..layers.len(), 0.0f64..self.num_qubits as f64)?;
        chart
            .configure_mesh()
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

        self.plot_moving_average(
            &mut chart,
            &layers,
            window_size,
            |window| {
                let sum: usize = window.iter().map(|layer| layer.len()).sum();
                sum as f64 / window.len() as f64
            },
            RGBColor(0, 0, 255),
            "avg products/layer",
        )?;

        self.plot_moving_average(
            &mut chart,
            &layers,
            window_size,
            |window| window.iter().map(|layer| layer.len()).max().unwrap_or(0) as f64,
            RGBColor(255, 165, 0),
            "max products/layer",
        )?;

        self.plot_moving_average(
            &mut chart,
            &layers,
            window_size,
            |window| {
                let (total_ops, total_products): (usize, usize) = window
                    .iter()
                    .map(|layer| {
                        let ops: usize = layer.iter().map(|pp| pp.operators.len()).sum();
                        (ops, layer.len())
                    })
                    .fold((0, 0), |(acc_ops, acc_prods), (ops, prods)| {
                        (acc_ops + ops, acc_prods + prods)
                    });

                if total_products > 0 { total_ops as f64 / total_products as f64 } else { 0.0 }
            },
            RGBColor(255, 0, 0),
            "avg product size",
        )?;

        self.plot_moving_average(
            &mut chart,
            &layers,
            window_size,
            |window| {
                window
                    .iter()
                    .map(|layer| layer.iter().map(|pp| pp.operators.len()).max().unwrap_or(0))
                    .max()
                    .unwrap_or(0) as f64
            },
            RGBColor(0, 200, 200),
            "max product size",
        )?;
        chart
            .configure_series_labels()
            .margin(20)
            .background_style(&WHITE)
            .border_style(&TRANSPARENT)
            .position(SeriesLabelPosition::UpperRight)
            .label_font(("sans-serif", 40))
            .draw()?;

        println!("Plotted layer statistics to {}", plot_fname);
        Ok(())
    }

    fn plot_moving_average<F>(
        &self, chart: &mut ChartContext<SVGBackend, Cartesian2d<RangedCoordusize, RangedCoordf64>>,
        layers: &[Vec<&PauliProduct>], window_size: usize, value_fn: F, color: RGBColor,
        label: &str,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: Fn(&[Vec<&PauliProduct>]) -> f64,
    {
        let data: Vec<f64> = layers
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let window_start = if i >= window_size { i - window_size } else { 0 };
                let window_end = i + 1;
                let window = &layers[window_start..window_end];
                value_fn(window)
            })
            .collect();

        for (i, &y_value) in data.iter().enumerate() {
            assert!(
                y_value <= self.num_qubits as f64,
                "Y value {} at index {} exceeds num_qubits {} for metric '{}'",
                y_value,
                i,
                self.num_qubits,
                label
            );
        }
        chart
            .draw_series(LineSeries::new(
                data.iter().enumerate().map(|(x, &y)| (x, y)),
                color.mix(0.8).stroke_width(2),
            ))?
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.mix(0.8).stroke_width(2))
            });

        Ok(())
    }

    pub(crate) fn count_t_stats(&self) -> (usize, usize) {
        let layers = self.get_layers();
        let num_t_gates = self.products.iter().filter(|pp| pp.gate_type.is_t()).count();
        let num_t_layers =
            layers.iter().filter(|layer| layer.iter().any(|pp| pp.gate_type.is_t())).count();
        (num_t_gates, num_t_layers)
    }

    pub(crate) fn print_statistics(&self) -> usize {
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
        println!("Circuit statistics:");
        println!("  Number of products:               {}", self.products.len());
        println!("  Number of Cliffords:              {}", num_cliffords);
        println!("  Layers:                           {}", num_layers);
        println!(
            "  Products per layer:               {:.2} avg, {} max",
            avg_products, max_products
        );
        num_layers
    }

    #[cfg(debug_assertions)]
    pub(crate) fn print(&self) -> io::Result<()> {
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

    /// Computes circuit layers using Kahn's topological sort (cached after first call).
    /// Each layer contains all products whose parents have all been placed in earlier layers.
    fn get_layers(&self) -> Vec<Vec<&PauliProduct>> {
        if let Some(cached) = self.layers.borrow().as_ref() {
            return cached
                .iter()
                .map(|layer| layer.iter().map(|&idx| &self.products[idx]).collect())
                .collect();
        }
        let mut in_degrees: Vec<usize> = self.products.iter().map(|pp| pp.parents.len()).collect();
        let mut ready: Vec<usize> = in_degrees
            .iter()
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
        assert_eq!(
            processed,
            self.products.len(),
            "Circuit contains cycles or unreachable products"
        );
        *self.layers.borrow_mut() = Some(index_layers.clone());
        index_layers
            .iter()
            .map(|layer| layer.iter().map(|&idx| &self.products[idx]).collect())
            .collect()
    }

    pub(crate) fn plot_qubit_coupling(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        let circuit_path = Path::new(&self.circuit_fname);
        let circuit_stem = circuit_path.file_stem().and_then(|s| s.to_str()).unwrap_or("circuit");
        let coupling_matrix = self.build_coupling_matrix();
        let dim = coupling_matrix.len();
        let plot_fname = format!("{}.qubit_coupling.svg", circuit_stem);
        let root = SVGBackend::new(&plot_fname, (1200, 1000)).into_drawing_area();
        root.fill(&WHITE)?;
        let max_count = coupling_matrix.iter().flat_map(|row| row.iter()).max().unwrap_or(&0);
        let mut chart = ChartBuilder::on(&root)
            .margin(60)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 60)
            .set_label_area_size(LabelAreaPosition::Right, 150)
            .caption(format!("{} - Qubit Coupling Matrix", circuit_stem), ("sans-serif", 24))
            .build_cartesian_2d(0..dim, 0..dim)?;
        chart
            .configure_mesh()
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
                    chart.draw_series(std::iter::once(Rectangle::new(
                        [(i, j), (i + 1, j + 1)],
                        color.filled(),
                    )))?;
                }
            }
        }
        println!("Plotted qubit coupling matrix to {}", plot_fname);
        Ok(())
    }

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
        let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
        for i in 0..(self.num_qubits - 1) {
            for j in (i + 1)..self.num_qubits {
                pairs.push((i, j, matrix[i][j]));
            }
        }
        pairs.sort_by(|p1, p2| p1.2.cmp(&p2.2).reverse());
        eprintln!("Pair frequencies:");
        for (q1, q2, n) in pairs {
            if n != 0 {
                eprintln!("  {} {} {}", q1, q2, n);
            }
        }
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_circuit_file(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f
    }

    #[test]
    fn new_creates_empty_circuit() {
        let c = Circuit::new(&"test.trans".to_string());
        assert_eq!(c.circuit_fname, "test.trans");
        assert_eq!(c.num_qubits, 0);
        assert_eq!(c.num_products(), 0);
    }

    #[test]
    fn load_circuit_single_t_gate() {
        let f = make_circuit_file(&["+_X______<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.num_products(), 1);
        assert_eq!(c.num_qubits, 2);
    }

    #[test]
    fn load_circuit_skips_x_and_z_gates() {
        let f = make_circuit_file(&["+X_<X>", "+_Z<Z>", "+XZ<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.num_products(), 1);
        assert!(c.get_product(0).gate_type.is_t());
    }

    #[test]
    fn load_circuit_multiple_products() {
        let f = make_circuit_file(&["+_X______<T>", "+___X____<T>", "+_____X__<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.num_products(), 3);
    }

    #[test]
    fn load_circuit_assigns_sequential_ids() {
        let f = make_circuit_file(&["+X_<T>", "+_X<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.get_product(0).id, 0);
        assert_eq!(c.get_product(1).id, 1);
    }

    #[test]
    fn load_circuit_nonexistent_file_returns_error() {
        let mut c = Circuit::new(&"/nonexistent/path/circuit.trans".to_string());
        assert!(c.load_circuit().is_err());
    }

    #[test]
    fn generate_dependencies_independent_products_have_no_parents() {
        let f = make_circuit_file(&["+X_<T>", "+_X<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert!(c.get_product(0).parents.is_empty());
        assert!(c.get_product(1).parents.is_empty());
    }

    #[test]
    fn generate_dependencies_sequential_same_qubit() {
        let f = make_circuit_file(&["+X_<T>", "+X_<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        let pp0 = c.get_product(0);
        let pp1 = c.get_product(1);
        assert!(pp0.parents.is_empty());
        assert_eq!(pp1.parents, vec![0]);
        assert_eq!(pp0.children, vec![1]);
        assert!(pp1.children.is_empty());
    }

    #[test]
    fn generate_dependencies_chain_of_three() {
        let f = make_circuit_file(&["+X__<T>", "+X__<T>", "+X__<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert!(c.get_product(0).parents.is_empty());
        assert_eq!(c.get_product(1).parents, vec![0]);
        assert_eq!(c.get_product(2).parents, vec![1]);
        assert_eq!(c.get_product(0).children, vec![1]);
        assert_eq!(c.get_product(1).children, vec![2]);
        assert!(c.get_product(2).children.is_empty());
    }

    #[test]
    fn initial_products_returns_products_with_no_parents() {
        let f = make_circuit_file(&["+X_<T>", "+_X<T>", "+X_<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        let initial_ids: Vec<i32> = c.initial_products().map(|pp| pp.id).collect();
        assert!(initial_ids.contains(&0));
        assert!(initial_ids.contains(&1));
        assert!(!initial_ids.contains(&2));
    }

    #[test]
    fn get_product_returns_correct_product() {
        let f = make_circuit_file(&["+X_<T>", "+_X<M>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert!(c.get_product(0).gate_type.is_t());
        assert!(c.get_product(1).gate_type.is_m());
    }

    #[test]
    fn num_products_matches_loaded_count() {
        let f = make_circuit_file(&["+X_<T>", "+_X<T>", "+XZ<CX>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.num_products(), 3);
    }

    #[test]
    fn num_qubits_is_max_qubit_plus_one() {
        let f = make_circuit_file(&["+___X<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.num_qubits, 4);
    }

    #[test]
    fn print_statistics_returns_correct_layer_count_linear_chain() {
        let f = make_circuit_file(&["+X__<T>", "+X__<T>", "+X__<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        let num_layers = c.print_statistics();
        assert_eq!(num_layers, 3);
    }

    #[test]
    fn print_statistics_parallel_products_in_one_layer() {
        let f = make_circuit_file(&["+X_<T>", "+_X<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        let num_layers = c.print_statistics();
        assert_eq!(num_layers, 1);
    }

    #[test]
    fn moving_average_does_not_panic_on_single_product() {
        let f = make_circuit_file(&["+X_<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        let _ = c.print_statistics();
    }

    #[test]
    fn count_t_stats_counts_t_gates_and_t_layers() {
        let f = make_circuit_file(&["+X_<T>", "+_X<T>", "+XZ<CX>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        let (num_t, num_t_layers) = c.count_t_stats();
        assert_eq!(num_t, 2);
        assert_eq!(num_t_layers, 1);
    }

    #[test]
    fn count_t_stats_no_t_gates() {
        let f = make_circuit_file(&["+XZ<CX>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        let (num_t, num_t_layers) = c.count_t_stats();
        assert_eq!(num_t, 0);
        assert_eq!(num_t_layers, 0);
    }

    #[test]
    fn count_t_stats_all_t_gates_in_separate_layers() {
        let f = make_circuit_file(&["+X_<T>", "+X_<T>", "+X_<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        let (num_t, num_t_layers) = c.count_t_stats();
        assert_eq!(num_t, 3);
        assert_eq!(num_t_layers, 3);
    }

    #[test]
    fn build_coupling_matrix_symmetric_for_cx_gate() {
        let f = make_circuit_file(&["+XZ<CX>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.num_qubits, 2);
        assert_eq!(c.num_products(), 1);
    }

    #[test]
    fn get_layers_diamond_dependency() {
        let f = make_circuit_file(&["+X___<T>", "+_X__<T>", "+X___<T>", "+_X__<T>", "+__X_<T>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        let num_layers = c.print_statistics();
        assert_eq!(num_layers, 2);
    }

    #[test]
    fn load_circuit_keeps_m_gate() {
        let f = make_circuit_file(&["+X_<M>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.num_products(), 1);
        assert!(c.get_product(0).gate_type.is_m());
    }

    #[test]
    fn load_circuit_keeps_s_gate() {
        let f = make_circuit_file(&["+X_<S>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.num_products(), 1);
        assert!(c.get_product(0).gate_type.is_s());
    }

    #[test]
    fn load_circuit_keeps_cx_gate() {
        let f = make_circuit_file(&["+XZ<CX>"]);
        let mut c = Circuit::new(&f.path().to_string_lossy().to_string());
        c.load_circuit().unwrap();
        assert_eq!(c.num_products(), 1);
        assert!(c.get_product(0).gate_type.is_cx());
    }
}
