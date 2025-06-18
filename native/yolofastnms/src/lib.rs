use rustler::{Binary, Env, Term, NifResult, Encoder};

use std::collections::HashSet;
use std::convert::TryInto;

#[derive(Debug, Clone)]
struct BBox {
    prob: f32,
    class: u16,
    cx: i32,
    cy: i32,
    w: i32,
    h: i32,
}

#[rustler::nif]
fn run_with_binary<'a>(
    env: Env<'a>, 
    binary: Binary, 
    prob_threshold: f32,
    iou_threshold: f32,
    rows: usize,
    columns: usize,
    transpose: bool
) -> NifResult<Term<'a>> {
    
    // load the matrix `Vec<Vec<f32>>` from binary.
    let matrix = binary_to_matrix(&binary, rows, columns);

    // transpose {rows, columns} to {columns, rows} if needed
    let matrix = if transpose {
        transpose_matrix(&matrix)
    } else {
        matrix
    };
    
    let bboxes = matrix_to_bboxes(&matrix);
    
    //keep only the bboxes with prob > prob_threshold
    let filtered_bboxes = bboxes.into_iter().filter(|b| b.prob >= prob_threshold).collect();

    //run NMS
    let final_bboxes = nms(&filtered_bboxes, iou_threshold);

    //convert BBox to [cx, cy, w, h, prob, class_idx]
    let result: Vec<Vec<f32>> = final_bboxes.into_iter().map(|bbox| {
        vec![
                bbox.cx as f32,
                bbox.cy as f32,
                bbox.w as f32,
                bbox.h as f32,
                bbox.prob,
                bbox.class as f32,
            ]
    }).collect();

    Ok(result.encode(env))
}

// tensor binary to a vector of vectors, each row of `row_size` bytes.
fn binary_to_matrix(binary: &Binary, rows: usize, columns: usize) -> Vec<Vec<f32>> {
    // Ensure the binary length is divisible by the row size
    let f32size = std::mem::size_of::<f32>();
    let row_size = (columns * f32size).try_into().unwrap();
    let total_size = rows * row_size;
    assert!(binary.len() == total_size, "Binary size ({}) is less than the total size ({})", binary.len(), total_size);
    assert!(binary.len() >= row_size, "Binary size ({}) is less than the row size ({})", binary.len(), row_size);
    assert!(binary.len() % row_size == 0, "Binary size ({}) is not a multiple of the row size ({})", binary.len(), row_size);
    
    binary
        .as_slice()
        .chunks(row_size)
        .map(|chunk| {
            chunk
                .chunks(f32size)
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn matrix_to_bboxes(matrix: &Vec<Vec<f32>>) -> Vec<BBox> {
    matrix
        .iter()
        .map(|row| bbox_from_row(&row))
        .collect()
}

fn bbox_from_row(row: &Vec<f32>) -> BBox {    
    let cx = row[0].round() as i32;
    let cy = row[1].round() as i32;
    let w = row[2].round() as i32;
    let h = row[3].round() as i32;


    //find the class with the highest probability
    let (max_prob, class) = row[4..].iter().enumerate()
        .fold((f32::MIN, 0), |(max_prob, max_class), (i, &prob)| {
            if prob > max_prob {
                (prob, i as u16)                
            } else {
                (max_prob, max_class)
            }
        });

    BBox {
        prob: max_prob,
        class,
        cx,
        cy,
        w,
        h
    }
}

fn nms(bboxes: &Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
    let mut final_boxes: Vec<BBox> = Vec::new();
    let mut class_kept_boxes: Vec<BBox> = Vec::new();

    for class in get_classes(&bboxes) {
        let class_boxes = sorted_boxes_filtered_by_class(&bboxes, class);
        class_kept_boxes.clear();

        for bbox in &class_boxes {
            let mut max_iou: f32 = 0.0;
            for kb in &class_kept_boxes {
                max_iou = calc_iou(&bbox, kb).max(max_iou);
            }
            if max_iou <= iou_threshold {
                class_kept_boxes.push(bbox.clone());
            }
        }
        final_boxes.extend(class_kept_boxes.drain(..));
    }

    final_boxes
}

fn calc_iou(a: &BBox, b: &BBox) -> f32 {
    // Calculate the coordinates of the intersection rectangle
    let x1 = (a.cx - a.w / 2).max(b.cx - b.w / 2);
    let y1 = (a.cy - a.h / 2).max(b.cy - b.h / 2);
    let x2 = (a.cx + a.w / 2).min(b.cx + b.w / 2);
    let y2 = (a.cy + a.h / 2).min(b.cy + b.h / 2);

    // Calculate the area of intersection
    let intersection_area = (x2 - x1).max(0) * (y2 - y1).max(0);

    // Calculate the area of both bounding boxes
    let a_area = a.w * a.h;
    let b_area = b.w * b.h;

    // Calculate the area of union
    let union_area = a_area + b_area - intersection_area;

    // Calculate and return the IoU
    if union_area == 0 {
        0.0
    } else {
        intersection_area as f32 / union_area as f32
    }
}

fn sorted_boxes_filtered_by_class(bboxes: &Vec<BBox>, class: u16) -> Vec<BBox> {
    let class_bboxes: Vec<BBox> = bboxes.iter().filter(|b| b.class == class).cloned().collect();
    let mut sorted_bboxes = class_bboxes.clone();
    sorted_bboxes.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());
    sorted_bboxes
}

fn get_classes(boxes: &Vec<BBox>) -> HashSet<u16> {
    let mut classes = HashSet::new();
    for b in boxes {
        classes.insert(b.class);
    }
    classes
}


fn transpose_matrix(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols = if rows > 0 { matrix[0].len() } else { 0 };

    let mut transposed = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }

    transposed
}

rustler::init!("Elixir.YoloFastNMS");