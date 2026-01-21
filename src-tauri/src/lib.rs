// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::sync::Mutex;
use tauri::State;
use serde::Serialize;

struct AppState {
    preview_context: Mutex<Option<PreviewContext>>,
}

struct PreviewContext {
    width: u32,
    height: u32,
    data: Vec<f32>, // RGB interleaved
}

#[derive(serde::Deserialize)]
struct ImageParams {
    exposure: f32,
    contrast: f32,
    temperature: f32,
    tint: f32,
}

#[derive(Serialize)]
struct ImageResult {
    width: u32,
    height: u32,
    data: Vec<f32>, // Linear RGB Float data
}

// 3x3 Matrix Multiplication
fn mult_matrix(m: &[[f32;3];3], v: &[f32;3]) -> [f32;3] {
    [
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
        m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
    ]
}

// Simple Superpixel / Bilinear Demosaic + Color Matrix
fn generate_preview_buffer(raw: &rawloader::RawImage) -> Result<PreviewContext, String> {
    let raw_data = match &raw.data {
        rawloader::RawImageData::Integer(v) => v,
        _ => return Err("Float raw data not supported".into()),
    };

    let full_w = raw.width;
    let full_h = raw.height;
    
    // Target size ~2000px wide
    let mut step = (full_w as f32 / 2000.0).ceil() as usize;
    if step < 2 { step = 2; }
    if step % 2 != 0 { step += 1; } 
    
    let w = full_w / step;
    let h = full_h / step;
    
    let mut data: Vec<f32> = Vec::with_capacity(w * h * 4); // RGBA
    
    // 1. Prepare Matrices
    // Camera -> XYZ (D65 usually)
    let cam_to_xyz_4x3 = raw.cam_to_xyz_normalized();
    // Convert 4x3 to 3x3 (ignoring 4th row if present or handling conversion)
    // rawloader returns [[f32;4];3] -> Wait, definition says: [[f32;4];3] which is 3 rows of 4 columns?
    // Let's check `rawloader` definition... usually it's RGB -> XYZ so 3x3.
    // The previous view_file showed: `pub fn cam_to_xyz_normalized(&self) -> [[f32;4];3]`
    // The struct `xyz_to_cam` is `[[f32;3];4]` (4 rows (RGBE), 3 cols (XYZ)).
    // Inversion `cam_to_xyz` returns 3 rows (XYZ), 4 columns (RGBE contribution).
    // Standard sensors are RGB, so 4th input might be zero or E=G2.
    // We will drop the 4th column for standard RGB processing if it's negligible.
    
    let mut cam_to_xyz = [[0.0;3];3];
    for i in 0..3 {
        for j in 0..3 {
            cam_to_xyz[i][j] = cam_to_xyz_4x3[i][j];
        }
    }

    // XYZ (D65) -> sRGB Matrix
    let xyz_to_srgb = [
         [ 3.2404542, -1.5371385, -0.4985314],
         [-0.9692660,  1.8760108,  0.0415560],
         [ 0.0556434, -0.2040259,  1.0572252],
    ];
    
    // Combine: cam_to_srgb = xyz_to_srgb * cam_to_xyz
    let mut cam_to_srgb = [[0.0;3];3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                cam_to_srgb[i][j] += xyz_to_srgb[i][k] * cam_to_xyz[k][j];
            }
        }
    }
    
    // Check if matrix is all zeros (identity fallback)
    let sum: f32 = cam_to_srgb.iter().flatten().sum();
    if sum == 0.0 {
        cam_to_srgb = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        println!("DEBUG: Zero Matrix detected, using Identity");
    }
    
    // Debug Info
    println!("DEBUG: Make={}, Model={}", raw.clean_make, raw.clean_model);
    println!("DEBUG: Cam -> sRGB Matrix={:?}", cam_to_srgb);
    
    let mut data: Vec<f32> = Vec::with_capacity(w * h * 4); // RGBA
    
    // Normalize WB relative to unit Green
    // wb_coeffs are RGBE. Usually Green is at 1 (and 3).
    let wb = raw.wb_coeffs;
    let g_val = if wb[1] > 0.0 { wb[1] } else { 1.0 }; // Assume Index 1 is Green
    
    // Safety check if they are integer scaled (e.g. > 100) or float (1.0-4.0)
    // If we normalize by Green, it handles both cases auto-magically.
    let mut wb_norm = [1.0; 4];
    for i in 0..4 {
        if g_val > 0.0001 {
            wb_norm[i] = wb[i] / g_val;
        }
    }
    
    println!("DEBUG: Raw WB={:?}, Norm WB={:?}", wb, wb_norm);

    // Safety: Ensure white_level / black_level is sane.
    let base_white = raw.whitelevels[1] as f32;
    // Use Green channel black level as baseline or average
    let black_level = raw.blacklevels[1] as f32; 
    let white_range = base_white - black_level;
    
    println!("DEBUG: BlackLevel={}, WhiteLevel={}, Range={}", black_level, base_white, white_range);

    for y in 0..h {
        for x in 0..w {
            let src_x = x * step;
            let src_y = y * step;
            
            let mut r_sum = 0.0;
            let mut g_sum = 0.0;
            let mut b_sum = 0.0;
            let mut r_cnt = 0.0;
            let mut g_cnt = 0.0;
            let mut b_cnt = 0.0;
            
            for by in 0..step {
                if src_y + by >= full_h { continue; }
                for bx in 0..step {
                    if src_x + bx >= full_w { continue; }
                    
                    let idx = (src_y + by) * full_w + (src_x + bx);
                    let color = raw.cfa.color_at(src_x + bx, src_y + by);
                    
                    // 1. Black Subtraction & Normalization
                    let raw_val = raw_data[idx] as f32;
                    // Use per-channel black level if possible, else use green
                    let bl = if color < 4 { raw.blacklevels[color] as f32 } else { black_level };
                    let val = ((raw_val - bl) / white_range).max(0.0);

                    // 2. White Balance (Using Normalized Coeffs)
                    let wb_gain = if color < 4 { wb_norm[color] } else { 1.0 };
                    let wb_val = val * wb_gain;
                    
                    match color {
                        0 => { r_sum += wb_val; r_cnt += 1.0; }, // R
                        1 => { g_sum += wb_val; g_cnt += 1.0; }, // G
                        2 => { b_sum += wb_val; b_cnt += 1.0; }, // B
                        _ => { g_sum += wb_val; g_cnt += 1.0; }, // G neighbors
                    }
                }
            }
            
            let r_avg = if r_cnt > 0.0 { r_sum / r_cnt } else { 0.0 };
            let g_avg = if g_cnt > 0.0 { g_sum / g_cnt } else { 0.0 };
            let b_avg = if b_cnt > 0.0 { b_sum / b_cnt } else { 0.0 };
            
            // 3. Apply Camera -> sRGB Matrix
            let srgb = mult_matrix(&cam_to_srgb, &[r_avg, g_avg, b_avg]);
            
            // 4. Clamp to avoid whiteout artifacts
            data.push(srgb[0].clamp(0.0, 1.0));
            data.push(srgb[1].clamp(0.0, 1.0));
            data.push(srgb[2].clamp(0.0, 1.0));
            data.push(1.0); // Alpha
        }
    }
    
    Ok(PreviewContext { width: w as u32, height: h as u32, data })
}

#[tauri::command]
fn load_raw(state: State<AppState>, path: &str) -> Result<ImageResult, String> {
    let raw = rawloader::decode_file(path).map_err(|e| e.to_string())?;
    
    let preview = generate_preview_buffer(&raw)?;
    
    let result = ImageResult {
        width: preview.width,
        height: preview.height,
        data: preview.data.clone(),
    };
    
    *state.preview_context.lock().unwrap() = Some(preview);
    
    Ok(result)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState { preview_context: Mutex::new(None) })
        .invoke_handler(tauri::generate_handler![load_raw])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
