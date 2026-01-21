// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::sync::Mutex;
use tauri::State;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use image::{ImageBuffer, Rgb};

struct AppState {
    preview_context: Mutex<Option<PreviewContext>>,
}

struct PreviewContext {
    width: u32,
    height: u32,
    data: Vec<f32>, // RGB interleaved
}

#[derive(serde::Deserialize, serde::Serialize)]
struct ImageParams {
    exposure: f32,
    contrast: f32,
    temperature: f32,
    tint: f32,
    highlights: f32,
    shadows: f32,
    whites: f32,
    blacks: f32,
    saturation: f32,
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

fn calculate_cam_to_srgb(raw: &rawloader::RawImage) -> [[f32;3];3] {
    let cam_to_xyz_4x3 = raw.cam_to_xyz_normalized();
    let mut cam_to_xyz = [[0.0;3];3];
    for i in 0..3 {
        for j in 0..3 {
            cam_to_xyz[i][j] = cam_to_xyz_4x3[i][j];
        }
    }

    let xyz_to_srgb = [
         [ 3.2404542, -1.5371385, -0.4985314],
         [-0.9692660,  1.8760108,  0.0415560],
         [ 0.0556434, -0.2040259,  1.0572252],
    ];
    
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
        return [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
    }
    
    cam_to_srgb
}

fn calculate_wb_norm(raw: &rawloader::RawImage) -> [f32; 4] {
    let wb = raw.wb_coeffs;
    let g_val = if wb[1] > 0.0 { wb[1] } else { 1.0 }; 
    
    let mut wb_norm = [1.0; 4];
    for i in 0..4 {
        if g_val > 0.0001 {
            wb_norm[i] = wb[i] / g_val;
        }
    }
    wb_norm
}

fn process_bayer(raw: &rawloader::RawImage, full_quality: bool) -> Result<PreviewContext, String> {
      let raw_data = match &raw.data {
        rawloader::RawImageData::Integer(v) => v,
        _ => return Err("Float raw data not supported".into()),
    };

    let full_w = raw.width;
    let full_h = raw.height;
    
    // Calculate step
    let mut step = if full_quality {
        // Full resolution (or half res superpixel).
        // For MVP stability, let's use step=2 for high quality block averaging.
        2 
    } else {
        // Preview: Target ~1024px wide for performance (Tauri IPC JSON limit)
        let s = (full_w as f32 / 1024.0).ceil() as usize;
        if s < 2 { 2 } else { s }
    };
    
    // Ensure even step for Superpixel logic (2x2 blocks)
    if step % 2 != 0 { step += 1; }

    let w = full_w / step;
    let h = full_h / step;
    
    let mut data: Vec<f32> = Vec::with_capacity(w * h * 4); // RGBA
    
    let cam_to_srgb = calculate_cam_to_srgb(raw);
    let wb_norm = calculate_wb_norm(raw);
    
    let base_white = raw.whitelevels[1] as f32;
    let black_level = raw.blacklevels[1] as f32; 
    let white_range = base_white - black_level;

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
                    
                    let raw_val = raw_data[idx] as f32;
                    let bl = if color < 4 { raw.blacklevels[color] as f32 } else { black_level };
                    let val = ((raw_val - bl) / white_range).max(0.0);

                    let wb_gain = if color < 4 { wb_norm[color] } else { 1.0 };
                    let wb_val = val * wb_gain;
                    
                    match color {
                        0 => { r_sum += wb_val; r_cnt += 1.0; }, 
                        1 => { g_sum += wb_val; g_cnt += 1.0; }, 
                        2 => { b_sum += wb_val; b_cnt += 1.0; }, 
                        _ => { g_sum += wb_val; g_cnt += 1.0; }, 
                    }
                }
            }
            
            let r_avg = if r_cnt > 0.0 { r_sum / r_cnt } else { 0.0 };
            let g_avg = if g_cnt > 0.0 { g_sum / g_cnt } else { 0.0 };
            let b_avg = if b_cnt > 0.0 { b_sum / b_cnt } else { 0.0 };
            
            let srgb = mult_matrix(&cam_to_srgb, &[r_avg, g_avg, b_avg]);
            
            data.push(srgb[0].clamp(0.0, 1.0));
            data.push(srgb[1].clamp(0.0, 1.0));
            data.push(srgb[2].clamp(0.0, 1.0));
            data.push(1.0); // Alpha
        }
    }
    
    Ok(PreviewContext { width: w as u32, height: h as u32, data })
}

// Logic mirroring Fragment Shader
fn apply_processing(r: f32, g: f32, b: f32, params: &ImageParams) -> (f32, f32, f32) {
    let mut rgb = [r, g, b];
    
    // 1. White Balance (Temp/Tint)
    // Matches JS logic: 5500K base.
    let ratio = (params.temperature - 5500.0) / 5500.0;
    let wb_r = 1.0 + ratio.max(0.0);
    let wb_b = 1.0 - ratio.min(0.0);
    let wb_g = 1.0 + params.tint / 100.0;
    
    rgb[0] *= wb_r;
    rgb[1] *= wb_g;
    rgb[2] *= wb_b;
    
    // 2. Exposure
    if params.exposure != 0.0 {
        let mag = 2.0_f32.powf(params.exposure);
        rgb[0] *= mag; rgb[1] *= mag; rgb[2] *= mag;
    }
    
    // 3. Contrast
    if params.contrast != 0.0 {
        let c = 1.0 + params.contrast;
        rgb[0] = (rgb[0] - 0.5) * c + 0.5;
        rgb[1] = (rgb[1] - 0.5) * c + 0.5;
        rgb[2] = (rgb[2] - 0.5) * c + 0.5;
    }
    
    // 4. Luma for Tone Mapping
    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    
    // 5. Highlights / Shadows
    // Simple mask approx logic from Shader
    // Note: Shader uses smoothstep(0.0, 0.6) for shadows and (0.4, 1.0) for highlights
    // Rust clamp based approx:
    let shadow_mask = 1.0 - (luma / 0.6).clamp(0.0, 1.0); 
    let high_mask = ((luma - 0.4) / 0.6).clamp(0.0, 1.0);
    
    if params.shadows != 0.0 {
        let lift = 2.0_f32.powf(params.shadows) - 1.0;
        let fact = lift * shadow_mask * 0.5;
        rgb[0] += rgb[0] * fact;
        rgb[1] += rgb[1] * fact;
        rgb[2] += rgb[2] * fact;
    }
    
    if params.highlights != 0.0 {
        let gain = 2.0_f32.powf(params.highlights) - 1.0;
        let fact = gain * high_mask * 0.5;
        rgb[0] += rgb[0] * fact;
        rgb[1] += rgb[1] * fact;
        rgb[2] += rgb[2] * fact;
    }
    
    // 6. Levels (Whites / Blacks)
    let black_point = params.blacks * 0.2;
    let mut white_point = 1.0 + params.whites * 0.2;
    if white_point - black_point < 0.001 { white_point = black_point + 0.001; }
    let range = white_point - black_point;
    
    rgb[0] = (rgb[0] - black_point) / range;
    rgb[1] = (rgb[1] - black_point) / range;
    rgb[2] = (rgb[2] - black_point) / range;

    // 7. Saturation
    if params.saturation != 0.0 {
        let l = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        let sat_mult = 1.0 + params.saturation;
        rgb[0] = l + (rgb[0] - l) * sat_mult;
        rgb[1] = l + (rgb[1] - l) * sat_mult;
        rgb[2] = l + (rgb[2] - l) * sat_mult;
    }
    
    // 7. Gamma Correction ( Linear -> sRGB )
    // Standard approx gamma 2.2
    let gamma = 1.0 / 2.2;
    rgb[0] = rgb[0].max(0.0).powf(gamma);
    rgb[1] = rgb[1].max(0.0).powf(gamma);
    rgb[2] = rgb[2].max(0.0).powf(gamma);

    (rgb[0], rgb[1], rgb[2])
}

#[tauri::command]
fn load_raw(state: State<AppState>, path: &str) -> Result<ImageResult, String> {
    let raw = rawloader::decode_file(path).map_err(|e| e.to_string())?;
    // Default Preview Scale (Small)
    let preview = process_bayer(&raw, false)?;
    
    let result = ImageResult {
        width: preview.width,
        height: preview.height,
        data: preview.data.clone(),
    };
    *state.preview_context.lock().unwrap() = Some(preview);
    Ok(result)
}

#[tauri::command]
fn export_image(path: &str, params: ImageParams, save_path: &str) -> Result<(), String> {
    let raw = rawloader::decode_file(path).map_err(|e| e.to_string())?;
    
    // Full Res (High Quality)
    let processed = process_bayer(&raw, true)?;
    
    let w = processed.width;
    let h = processed.height;
    
    // Create Image Buffer
    let mut imgbuf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w, h);
    
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let idx = (y * w + x) as usize * 4;
        let r_lin = processed.data[idx];
        let g_lin = processed.data[idx+1];
        let b_lin = processed.data[idx+2];
        
        let (r_out, g_out, b_out) = apply_processing(r_lin, g_lin, b_lin, &params);
        
        let r8 = (r_out.clamp(0.0, 1.0) * 255.0) as u8;
        let g8 = (g_out.clamp(0.0, 1.0) * 255.0) as u8;
        let b8 = (b_out.clamp(0.0, 1.0) * 255.0) as u8;
        
        *pixel = Rgb([r8, g8, b8]);
    }
    
    imgbuf.save(save_path).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn save_params(path: &str, params: ImageParams) -> Result<(), String> {
    let json_val = serde_json::to_string_pretty(&params).map_err(|e| e.to_string())?;
    let mut file = File::create(path).map_err(|e| e.to_string())?;
    file.write_all(json_val.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn load_params(path: &str) -> Result<ImageParams, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let params: ImageParams = serde_json::from_reader(file).map_err(|e| e.to_string())?;
    Ok(params)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState { preview_context: Mutex::new(None) })
        .invoke_handler(tauri::generate_handler![load_raw, export_image, save_params, load_params])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
