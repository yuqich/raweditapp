// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::sync::Mutex;
use tauri::State;
use serde::Serialize;

struct AppState {
    // We store a downsampled float buffer (RGB) for fast processing
    // Range 0.0 - 1.0 (normalized)
    preview_context: Mutex<Option<PreviewContext>>,
}

struct PreviewContext {
    width: u32,
    height: u32,
    data: Vec<f32>, // RGB interleaved
}

#[derive(serde::Deserialize)]
struct ImageParams {
    exposure: f32, // EV
    contrast: f32, // 0.0 = neutral.
    temperature: f32, // Kelvin. 5500 neutral
    tint: f32, // -100 to 100
}

#[derive(Serialize)]
struct ImageResult {
    width: u32,
    height: u32,
    data: Vec<f32>, // Linear RGB Float data
}

// Simple Superpixel Demosaic (2x2 -> 1 pixel)
fn generate_preview_buffer(raw: &rawloader::RawImage) -> Result<PreviewContext, String> {
    let raw_data = match &raw.data {
        rawloader::RawImageData::Integer(v) => v,
        _ => return Err("Float raw data not supported".into()),
    };

    let full_w = raw.width;
    let full_h = raw.height;
    
    // Target size: approx 2000px wide (max) - Browsers generally handle 2k-4k textures fine
    // We can go higher quality now since we only do it once.
    let step = ((full_w as f32 / 2000.0).ceil() as usize / 2 * 2).max(2);
    
    let w = full_w / step;
    let h = full_h / step;
    
    let mut data = Vec::with_capacity(w * h * 3);
    
    let white = 4095.0; // Todo: get real whitelevel

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
            
            // Average over the step block
            for by in 0..step {
                if src_y + by >= full_h { continue; }
                for bx in 0..step {
                    if src_x + bx >= full_w { continue; }
                    
                    let idx = (src_y + by) * full_w + (src_x + bx);
                    let val = raw_data[idx] as f32 / white;
                    let color = raw.cfa.color_at(src_x + bx, src_y + by);
                    
                    match color {
                        0 => { r_sum += val; r_cnt += 1.0; }, // R
                        1 => { g_sum += val; g_cnt += 1.0; }, // G
                        2 => { b_sum += val; b_cnt += 1.0; }, // B
                        _ => { g_sum += val; g_cnt += 1.0; }, // Fallback
                    }
                }
            }
            
            let r = if r_cnt > 0.0 { r_sum / r_cnt } else { 0.0 };
            let g = if g_cnt > 0.0 { g_sum / g_cnt } else { 0.0 };
            let b = if b_cnt > 0.0 { b_sum / b_cnt } else { 0.0 };
            
            data.push(r);
            data.push(g);
            data.push(b);
            data.push(1.0); // Alpha (Padding for WebGL alignment/support)
        }
    }
    
    Ok(PreviewContext { width: w as u32, height: h as u32, data })
}

#[tauri::command]
fn load_raw(state: State<AppState>, path: &str) -> Result<ImageResult, String> {
    let raw = rawloader::decode_file(path).map_err(|e| e.to_string())?;
    
    let preview = generate_preview_buffer(&raw)?;
    
    // With WebGL, we just return the RAW linear float buffer.
    // The frontend will handle all color/exposure processing.
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
