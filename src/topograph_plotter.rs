use crate::fn_timer;
use crate::node::NodeType;
use crate::pauliproduct::PauliProduct;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use plotters::prelude::*;
use std::rc::Rc;

static STATUS_FONT_SIZE: u32 = 26;
static TLABEL_FONT_SIZE: u32 = 40;
static CULTIVATION_LABEL_FONT_SIZE: u32 = 30;
static DATA_QUBIT_LABEL_FONT_SIZE: u32 = 36;
static PRODUCT_LABEL_FONT_SIZE: u32 = 28;
static BOXED_TERM_LABEL_FONT_SIZE: u32 = 30;

/// Geometry for one double-data-qubit group (X row + Z row sharing a column).
pub(crate) struct DataGroup {
    pub col: f32,
    pub y_x: f32,
    pub y_z: f32,
    pub x_left_id: u16,
    pub x_right_id: u16,
    pub z_left_id: u16,
    pub z_right_id: u16,
}

/// Side of a data node a routing neighbor is on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataSide {
    Left,
    Right,
    Top,
    Bottom,
}

type PlotChart<'a> = plotters::prelude::ChartContext<
    'a,
    BitMapBackend<'a>,
    plotters::prelude::Cartesian2d<
        plotters::coord::types::RangedCoordf32,
        plotters::coord::types::RangedCoordf32,
    >,
>;

/// Handles all visualization/plotting for a [`TopoGraph`].
pub(crate) struct TopoGraphPlotter<'a> {
    topo: &'a TopoGraph,
}

impl<'a> TopoGraphPlotter<'a> {
    pub(crate) fn new(topo: &'a TopoGraph) -> Self {
        TopoGraphPlotter { topo }
    }

    pub(crate) fn plot(
        &self, fname_added: &str, pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>, u32)],
        title_str: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        let topo_stem = self.topo.circuit_stem();
        let plot_fname = format!("{}{}.png", topo_stem, fname_added);

        let root = BitMapBackend::new(
            &plot_fname,
            (self.topo.num_cols as u32 * 90, self.topo.num_rows as u32 * 90),
        )
        .into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .set_label_area_size(LabelAreaPosition::Bottom, 50)
            .build_cartesian_2d(
                -1f32..self.topo.num_cols as f32,
                -1f32..self.topo.num_rows as f32,
            )?;

        let product_label_positions = self.compute_product_label_positions(pauli_product_paths);
        let mut product_label_covered: std::collections::HashSet<(i32, i32)> =
            std::collections::HashSet::new();
        for opt in &product_label_positions {
            if let Some((cx, cy, tw, _)) = *opt {
                let x_min = cx - tw / 2.0 - 0.05;
                let x_max = cx + tw / 2.0 + 0.05;
                let row_key = (cy * 10.0).round() as i32;
                for node in self.topo.iter_nodes() {
                    if node.node_type == NodeType::Data {
                        continue;
                    }
                    let (nx, ny) = node.pos;
                    if (ny * 10.0).round() as i32 == row_key && nx >= x_min && nx <= x_max {
                        product_label_covered.insert(((nx * 10.0).round() as i32, row_key));
                    }
                }
            }
        }

        self.draw_all_nodes_plain(&mut chart, pauli_product_paths, &product_label_covered)?;
        self.draw_data_group_borders_plain(&mut chart, self.topo.data_groups())?;
        self.draw_path_overlays(&mut chart, pauli_product_paths, &product_label_covered)?;
        self.draw_data_group_labels(&mut chart, self.topo.data_groups())?;
        self.draw_product_labels(&mut chart, pauli_product_paths, &product_label_positions)?;

        if !title_str.is_empty() {
            for (i, line) in title_str.split('\n').enumerate() {
                draw_text(
                    &mut chart,
                    line,
                    -0.5,
                    -0.8 - (i as f32 * 0.33),
                    ("monotype", STATUS_FONT_SIZE),
                )?;
            }
        }
        root.present()?;
        println!("Plotted topology to {}", plot_fname);
        Ok(())
    }

    /// Returns a stable, visually distinct color for a product ID.
    /// Uses the golden-ratio hue sequence to maximise perceptual separation
    /// between consecutive product IDs.
    fn product_color(pp_id: i32) -> RGBAColor {
        const GOLDEN: f64 = 0.618_033_988_749_895;
        let hue = (pp_id as f64 * GOLDEN).fract().abs();
        let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
        RGBColor(r, g, b).to_rgba()
    }

    /// Draws all node fills and routing node borders with no path coloring.
    /// Magic nodes show: their label (no paths), nothing (in a path), remaining
    /// cultivation lcycles (cultivating), or "T" (ready).
    fn draw_all_nodes_plain(
        &self, chart: &mut PlotChart, pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>, u32)],
        product_label_covered: &std::collections::HashSet<(i32, i32)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_node_ids: std::collections::HashSet<u16> =
            pauli_product_paths.iter().flat_map(|(_, tree, _)| tree.iter_nodes()).collect();

        for node in self.topo.iter_nodes() {
            let (x, y) = node.pos;
            // Data nodes are narrower (hx=0.25) to show the X/Z pair side-by-side.
            let (hx, hy) =
                if node.node_type == NodeType::Data { (0.25f32, 0.5f32) } else { (0.5f32, 0.5f32) };

            let fill = match node.node_type {
                NodeType::Magic => RGBColor(0xFF, 0xDD, 0x44).mix(0.25).filled(),
                NodeType::Bus => RGBColor(0x88, 0x88, 0x88).mix(0.15).filled(),
                NodeType::Data => RGBColor(0x44, 0x88, 0xFF).mix(0.25).filled(),
            };
            draw_rect(chart, x, y, hx, hy, fill)?;

            if node.node_type != NodeType::Data {
                draw_rect(chart, x, y, hx, hy, BLACK.stroke_width(1))?;
                let pos_key = ((x * 10.0).round() as i32, (y * 10.0).round() as i32);
                if product_label_covered.contains(&pos_key) {
                    continue;
                }
                let label = &self.topo.labels[node.id as usize];
                let label_text = match node.node_type {
                    NodeType::Magic => {
                        if pauli_product_paths.is_empty() {
                            label.clone()
                        } else if path_node_ids.contains(&node.id) {
                            String::new() // Covered by path overlay.
                        } else if self.topo.is_cultivating(node.id) {
                            // Show remaining cultivation lcycles.
                            (self.topo.cultivation_times[node.id as usize]
                                - self.topo.busy_counts[node.id as usize])
                                .to_string()
                        } else {
                            "T".to_string() // Ready to use.
                        }
                    }
                    NodeType::Bus => {
                        if pauli_product_paths.is_empty() {
                            label.clone()
                        } else {
                            String::new()
                        }
                    }
                    NodeType::Data => String::new(),
                };
                if !label_text.is_empty() {
                    draw_text(
                        chart,
                        &label_text,
                        x - 0.17,
                        y + 0.09,
                        if label_text == "T" {
                            ("monotype", TLABEL_FONT_SIZE, FontStyle::Bold)
                        } else {
                            ("monotype", CULTIVATION_LABEL_FONT_SIZE, FontStyle::Normal)
                        },
                    )?;
                }
            }
        }
        Ok(())
    }

    fn draw_data_group_borders_plain(
        &self, chart: &mut PlotChart, data_groups: &[DataGroup],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for group in data_groups {
            let left = group.col - 0.5;
            let right = group.col + 0.5;
            let y_top = group.y_x.max(group.y_z);
            let y_bot = group.y_x.min(group.y_z);
            let top = y_top + 0.5;
            let bot = y_bot - 0.5;
            let x_is_top = group.y_x > group.y_z;

            if x_is_top {
                draw_dashed(chart, false, left, top, right - left, BLACK.to_rgba())?;
            } else {
                draw_line(chart, (left, top), (right, top), BLACK.stroke_width(2))?;
            }

            if x_is_top {
                draw_line(chart, (left, bot), (right, bot), BLACK.stroke_width(2))?;
            } else {
                draw_dashed(chart, false, left, bot, right - left, BLACK.to_rgba())?;
            }

            let (x_left_y, z_left_y) = (group.y_x, group.y_z);
            draw_dashed(chart, true, left, x_left_y - 0.5, 1.0, BLACK.to_rgba())?;
            draw_line(
                chart,
                (left, z_left_y - 0.5),
                (left, z_left_y + 0.5),
                BLACK.stroke_width(2),
            )?;

            draw_dashed(chart, true, right, x_left_y - 0.5, 1.0, BLACK.to_rgba())?;
            draw_line(
                chart,
                (right, z_left_y - 0.5),
                (right, z_left_y + 0.5),
                BLACK.stroke_width(2),
            )?;
        }
        Ok(())
    }

    fn data_side_of_neighbor(data_pos: (f32, f32), nbor_pos: (f32, f32)) -> Option<DataSide> {
        let dx = nbor_pos.0 - data_pos.0;
        let dy = nbor_pos.1 - data_pos.1;
        if dx.abs() > dy.abs() {
            if dx > 0.0 { Some(DataSide::Right) } else { Some(DataSide::Left) }
        } else if dy.abs() > 0.0 {
            if dy > 0.0 { Some(DataSide::Top) } else { Some(DataSide::Bottom) }
        } else {
            None
        }
    }

    fn draw_data_node_side_highlight(
        chart: &mut PlotChart, data_pos: (f32, f32), side: DataSide, color: RGBAColor,
        is_x_node: bool, group_outer_y: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (x, y) = data_pos;
        let (hx, hy) = (0.25f32, 0.5f32);
        let style = color.stroke_width(3);
        match side {
            DataSide::Left => {
                let p1 = (x - hx, y - hy);
                let p2 = (x - hx, y + hy);
                if is_x_node {
                    draw_dashed(chart, true, p1.0, p1.1, hy * 2.0, color)?;
                } else {
                    draw_line(chart, p1, p2, style)?;
                }
            }
            DataSide::Right => {
                let p1 = (x + hx, y - hy);
                let p2 = (x + hx, y + hy);
                if is_x_node {
                    draw_dashed(chart, true, p1.0, p1.1, hy * 2.0, color)?;
                } else {
                    draw_line(chart, p1, p2, style)?;
                }
            }
            DataSide::Top => {
                if is_x_node {
                    draw_dashed(chart, false, x - hx, group_outer_y, hx * 2.0, color)?;
                } else {
                    draw_line(chart, (x - hx, group_outer_y), (x + hx, group_outer_y), style)?;
                }
            }
            DataSide::Bottom => {
                if is_x_node {
                    draw_dashed(chart, false, x - hx, group_outer_y, hx * 2.0, color)?;
                } else {
                    draw_line(chart, (x - hx, group_outer_y), (x + hx, group_outer_y), style)?;
                }
            }
        }
        Ok(())
    }

    fn draw_path_overlays(
        &self, chart: &mut PlotChart, pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>, u32)],
        product_label_covered: &std::collections::HashSet<(i32, i32)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (pp, path_graph, _cycle) in pauli_product_paths.iter() {
            let color = Self::product_color(pp.id);
            let is_t = pp.gate_type.is_t();

            let routing_ids: Vec<u16> = path_graph
                .iter_nodes()
                .filter(|&id| self.topo.get_node(id).node_type != NodeType::Data)
                .collect();

            let routing_pos_set: std::collections::HashSet<(i32, i32)> = routing_ids
                .iter()
                .map(|&id| {
                    let (px, py) = self.topo.get_node(id).pos;
                    ((px * 10.0).round() as i32, (py * 10.0).round() as i32)
                })
                .collect();

            for &id in &routing_ids {
                let node = self.topo.get_node(id);
                let is_root = path_graph.root_node_id == Some(id);
                self.draw_routing_node_overlay(
                    chart,
                    id,
                    node.pos,
                    color,
                    is_t,
                    is_root,
                    &routing_pos_set,
                    product_label_covered,
                )?;
                self.draw_data_node_connections(chart, id, path_graph, color)?;
            }
        }
        Ok(())
    }

    fn draw_routing_node_overlay(
        &self, chart: &mut PlotChart, id: u16, pos: (f32, f32), color: RGBAColor, is_t: bool,
        is_root: bool, routing_pos_set: &std::collections::HashSet<(i32, i32)>,
        product_label_covered: &std::collections::HashSet<(i32, i32)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (x, y) = pos;
        let (hx, hy) = (0.5f32, 0.5f32);

        draw_rect(chart, x, y, hx, hy, color.mix(0.35).filled())?;

        let pos_key = ((x * 10.0).round() as i32, (y * 10.0).round() as i32);
        if is_root && is_t && !product_label_covered.contains(&pos_key) {
            draw_text(
                chart,
                "T",
                x - 0.17,
                y + 0.09,
                ("monotype", TLABEL_FONT_SIZE, FontStyle::Bold),
            )?;
        }

        let xi = (x * 10.0).round() as i32;
        let yi = (y * 10.0).round() as i32;
        let step_x = (hx * 2.0 * 10.0).round() as i32;
        let step_y = (hy * 2.0 * 10.0).round() as i32;
        for &(p1, p2, dx, dy) in &[
            ((x - hx, y - hy), (x + hx, y - hy), 0i32, -step_y),
            ((x - hx, y + hy), (x + hx, y + hy), 0i32, step_y),
            ((x - hx, y - hy), (x - hx, y + hy), -step_x, 0i32),
            ((x + hx, y - hy), (x + hx, y + hy), step_x, 0i32),
        ] {
            if !routing_pos_set.contains(&(xi + dx, yi + dy)) {
                draw_line(chart, p1, p2, BLACK.stroke_width(2))?;
            }
        }
        let _ = id;
        Ok(())
    }

    fn draw_data_node_connections(
        &self, chart: &mut PlotChart, routing_id: u16, path_graph: &TreeGraph, color: RGBAColor,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let routing_node = self.topo.get_node(routing_id);
        for &nbor_id in path_graph.neighbors(routing_id) {
            let nbor = self.topo.get_node(nbor_id);
            if nbor.node_type != NodeType::Data {
                continue;
            }
            let nbor_label = &self.topo.labels[nbor_id as usize];
            let is_x_node = nbor_label.ends_with('X');
            if let Some(side) = Self::data_side_of_neighbor(nbor.pos, routing_node.pos) {
                let group_outer_y = if side == DataSide::Top || side == DataSide::Bottom {
                    if let Some(&gi) = self.topo.node_to_group().get(&nbor_id) {
                        let groups = self.topo.data_groups();
                        let group = &groups[gi];
                        let y_top = group.y_x.max(group.y_z) + 0.5;
                        let y_bot = group.y_x.min(group.y_z) - 0.5;
                        if side == DataSide::Top { y_top } else { y_bot }
                    } else {
                        if side == DataSide::Top { nbor.pos.1 + 0.5 } else { nbor.pos.1 - 0.5 }
                    }
                } else {
                    if side == DataSide::Top { nbor.pos.1 + 0.5 } else { nbor.pos.1 - 0.5 }
                };
                Self::draw_data_node_side_highlight(
                    chart,
                    nbor.pos,
                    side,
                    color,
                    is_x_node,
                    group_outer_y,
                )?;
                let letter = if is_x_node { "X" } else { "Z" };
                let (lx, ly) = match side {
                    DataSide::Left => (nbor.pos.0 - 0.25, nbor.pos.1),
                    DataSide::Right => (nbor.pos.0 + 0.25, nbor.pos.1),
                    DataSide::Top | DataSide::Bottom => (nbor.pos.0, group_outer_y),
                };
                draw_boxed_label(chart, letter, lx, ly, color)?;
            }
        }
        Ok(())
    }

    fn draw_data_group_labels(
        &self, chart: &mut PlotChart, data_groups: &[DataGroup],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for group in data_groups {
            let y_top = group.y_x.max(group.y_z);
            let y_bot = group.y_x.min(group.y_z);
            let x_is_top = group.y_x > group.y_z;
            let (x_row_y, z_row_y) = if x_is_top { (y_top, y_bot) } else { (y_bot, y_top) };
            let x_label = self.topo.labels[group.x_left_id as usize]
                .trim_end_matches('X')
                .trim_start_matches('d')
                .to_string();
            let z_label = self.topo.labels[group.z_right_id as usize]
                .trim_end_matches('Z')
                .trim_start_matches('d')
                .to_string();
            let label_x = group.col - 0.17;
            draw_text(
                chart,
                &x_label,
                label_x,
                x_row_y + 0.09,
                ("monotype", DATA_QUBIT_LABEL_FONT_SIZE),
            )?;
            draw_text(
                chart,
                &z_label,
                label_x,
                z_row_y + 0.09,
                ("monotype", DATA_QUBIT_LABEL_FONT_SIZE),
            )?;
        }
        Ok(())
    }

    fn compute_product_label_positions(
        &self, pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>, u32)],
    ) -> Vec<Option<(f32, f32, f32, usize)>> {
        let mut labeled_positions: std::collections::HashSet<(i32, i32)> =
            std::collections::HashSet::new();
        for (pp, path_graph, _cycle) in pauli_product_paths.iter() {
            if let Some(root_id) = path_graph.root_node_id {
                if pp.gate_type.is_t() {
                    let (px, py) = self.topo.get_node(root_id).pos;
                    labeled_positions
                        .insert(((px * 10.0).round() as i32, (py * 10.0).round() as i32));
                }
            }
        }
        for node in self.topo.iter_nodes() {
            if self.topo.is_cultivating(node.id) {
                let (px, py) = node.pos;
                labeled_positions.insert(((px * 10.0).round() as i32, (py * 10.0).round() as i32));
            }
        }

        pauli_product_paths
            .iter()
            .enumerate()
            .map(|(i, (pp, path_graph, _cycle))| {
                let mut row_map: std::collections::HashMap<i32, Vec<f32>> =
                    std::collections::HashMap::new();
                for id in path_graph.iter_nodes() {
                    let node = self.topo.get_node(id);
                    if node.node_type != NodeType::Data {
                        let (px, py) = node.pos;
                        row_map.entry((py * 10.0).round() as i32).or_default().push(px);
                    }
                }
                let eligible: Vec<(&i32, &Vec<f32>)> = row_map
                    .iter()
                    .filter(|(row_key, xs)| {
                        let ry = **row_key;
                        !xs.iter().any(|&px| {
                            labeled_positions.contains(&((px * 10.0).round() as i32, ry))
                        })
                    })
                    .collect();
                let best = if !eligible.is_empty() {
                    eligible.iter().max_by_key(|(_, xs)| xs.len()).copied()
                } else {
                    row_map.iter().max_by_key(|(_, xs)| xs.len()).map(|(k, v)| (k, v))
                };
                best.map(|(&row_key, xs)| {
                    let row_y = row_key as f32 / 10.0;
                    let center_x = (xs.iter().cloned().fold(f32::INFINITY, f32::min)
                        + xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
                        / 2.0;
                    // Width accounts for optional "/n" cycle suffix (3 chars × 0.125 each)
                    let op_str = pp.to_operator_str();
                    let tw = op_str.len() as f32 * 0.125;
                    (center_x, row_y, tw, i)
                })
            })
            .collect()
    }

    fn draw_product_labels(
        &self, chart: &mut PlotChart, pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>, u32)],
        positions: &[Option<(f32, f32, f32, usize)>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (pos_opt, (pp, _path_graph, cycle)) in positions.iter().zip(pauli_product_paths.iter())
        {
            if let Some(&(center_x, row_y, tw, _i)) = pos_opt.as_ref() {
                draw_text(
                    chart,
                    &pp.to_operator_str(),
                    center_x - tw / 2.0,
                    row_y + 0.22,
                    ("monotype", PRODUCT_LABEL_FONT_SIZE),
                )?;
                let id_str =
                    if *cycle > 1 { format!("{} ({})", pp.id, cycle) } else { pp.id.to_string() };
                let id_w = id_str.len() as f32 * 0.10;
                draw_text(
                    chart,
                    &id_str,
                    center_x - id_w / 2.0,
                    row_y - 0.05,
                    ("monotype", PRODUCT_LABEL_FONT_SIZE),
                )?;
            }
        }
        Ok(())
    }
}

fn draw_rect(
    chart: &mut PlotChart, cx: f32, cy: f32, hx: f32, hy: f32, style: ShapeStyle,
) -> Result<(), Box<dyn std::error::Error>> {
    chart.draw_series(std::iter::once(Rectangle::new(
        [(cx - hx, cy - hy), (cx + hx, cy + hy)],
        style,
    )))?;
    Ok(())
}

fn draw_rect_coords(
    chart: &mut PlotChart, x0: f32, y0: f32, x1: f32, y1: f32, style: ShapeStyle,
) -> Result<(), Box<dyn std::error::Error>> {
    chart.draw_series(std::iter::once(Rectangle::new([(x0, y0), (x1, y1)], style)))?;
    Ok(())
}

fn draw_line(
    chart: &mut PlotChart, p1: (f32, f32), p2: (f32, f32), style: ShapeStyle,
) -> Result<(), Box<dyn std::error::Error>> {
    chart.draw_series(LineSeries::new(vec![p1, p2], style))?;
    Ok(())
}

fn draw_text<'a, F: plotters::style::IntoFont<'a>>(
    chart: &mut PlotChart, text: &str, x: f32, y: f32, font: F,
) -> Result<(), Box<dyn std::error::Error>> {
    chart.draw_series(std::iter::once(Text::new(text.to_string(), (x, y), font.into_font())))?;
    Ok(())
}

fn draw_dashed(
    chart: &mut PlotChart, vertical: bool, x0: f32, y0: f32, len: f32, c: RGBAColor,
) -> Result<(), Box<dyn std::error::Error>> {
    let end = if vertical { (x0, y0 + len) } else { (x0 + len, y0) };
    draw_line(chart, (x0, y0), end, WHITE.stroke_width(3))?;
    let (dash, gap) = (0.08f32, 0.06f32);
    let mut pos = 0.0f32;
    let mut on = true;
    while pos < len {
        let seg = if on { dash } else { gap }.min(len - pos);
        if on {
            let (p1, p2) = if vertical {
                ((x0, y0 + pos), (x0, y0 + pos + seg))
            } else {
                ((x0 + pos, y0), (x0 + pos + seg, y0))
            };
            draw_line(chart, p1, p2, c.stroke_width(2))?;
        }
        pos += seg;
        on = !on;
    }
    Ok(())
}

fn draw_boxed_label(
    chart: &mut PlotChart, letter: &str, cx: f32, cy: f32, c: RGBAColor,
) -> Result<(), Box<dyn std::error::Error>> {
    let (bh, bw) = (0.18f32, 0.13f32);
    draw_rect_coords(chart, cx - bw, cy - bh, cx + bw, cy + bh, WHITE.filled())?;
    draw_rect_coords(
        chart,
        cx - bw,
        cy - bh,
        cx + bw,
        cy + bh,
        Into::<ShapeStyle>::into(c).stroke_width(2),
    )?;
    chart.draw_series(std::iter::once(Text::new(
        letter.to_string(),
        (cx - bw * 0.45, cy + bh * 0.35 + 0.05),
        ("monotype", BOXED_TERM_LABEL_FONT_SIZE).into_font().color(&c),
    )))?;
    Ok(())
}

/// Converts HSV (hue ∈ [0,1), saturation, value) to RGB bytes.
pub(crate) fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0).floor() as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8)
}
