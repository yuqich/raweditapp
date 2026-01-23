// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use image::{ImageBuffer, Rgb};
use serde::Serialize;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::ptr;
use std::sync::Mutex;
use tauri::State;

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

fn apply_processing(r: f32, g: f32, b: f32, params: &ImageParams) -> (f32, f32, f32) {
    let mut rgb = [r, g, b];

    // 1. White Balance (Temp/Tint)
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
        rgb[0] *= mag;
        rgb[1] *= mag;
        rgb[2] *= mag;
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
    if white_point - black_point < 0.001 {
        white_point = black_point + 0.001;
    }
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

    // 8. Gamma (to sRGB)
    let gamma = 1.0 / 2.2;
    rgb[0] = rgb[0].max(0.0).powf(gamma);
    rgb[1] = rgb[1].max(0.0).powf(gamma);
    rgb[2] = rgb[2].max(0.0).powf(gamma);

    (rgb[0], rgb[1], rgb[2])
}

fn process_libraw(path: &str, target_width: Option<usize>) -> Result<PreviewContext, String> {
    unsafe {
        let raw_data = libraw_sys::libraw_init(0);
        if raw_data.is_null() {
            return Err("Failed to init libraw".into());
        }

        let c_path = CString::new(path).map_err(|_| "Invalid path")?;
        if libraw_sys::libraw_open_file(raw_data, c_path.as_ptr()) != 0 {
            libraw_sys::libraw_close(raw_data);
            return Err("Failed to open file".into());
        }

        if libraw_sys::libraw_unpack(raw_data) != 0 {
            libraw_sys::libraw_close(raw_data);
            return Err("Failed to unpack".into());
        }

        // Configure Params (accessing raw_data->params)
        // Note: libraw_sys usage might require dereferencing raw pointers carefully
        // output_bps = 16
        // output_color = 1 (sRGB)
        // no_auto_bright = 1
        // use_camera_wb = 1
        // gamm = [1.0, 1.0]

        (*raw_data).params.output_bps = 16;
        (*raw_data).params.user_qual = 3; // AHD interpolation
        (*raw_data).params.output_color = 1; // sRGB
        (*raw_data).params.no_auto_bright = 1;
        (*raw_data).params.use_camera_wb = 1;
        (*raw_data).params.gamm[0] = 1.0;
        (*raw_data).params.gamm[1] = 1.0;

        if libraw_sys::libraw_dcraw_process(raw_data) != 0 {
            libraw_sys::libraw_close(raw_data);
            return Err("Failed to process".into());
        }

        let processed = libraw_sys::libraw_dcraw_make_mem_image(raw_data, ptr::null_mut());
        if processed.is_null() {
            libraw_sys::libraw_close(raw_data);
            return Err("Failed to make mem image".into());
        }

        let w = (*processed).width as usize;
        let h = (*processed).height as usize;
        let channels = (*processed).colors as usize; // usually 3
        let bits = (*processed).bits as usize; // should be 16

        println!(
            "LibRaw Render: {}x{} channels={} bits={} data_size={}",
            w,
            h,
            channels,
            bits,
            (*processed).data_size
        );

        // Data is in (*processed).data which is slice of bytes
        let data_size = (*processed).data_size as usize;
        let raw_bytes = std::slice::from_raw_parts((*processed).data.as_ptr(), data_size);

        // Determine Step
        let step = if let Some(target) = target_width {
            let s = (w as f32 / target as f32).ceil() as usize;
            if s < 1 {
                1
            } else {
                s
            }
        } else {
            1
        };

        let out_w = w / step;
        let out_h = h / step;
        println!("Output: {}x{} (step={})", out_w, out_h, step);

        let mut out_data = Vec::with_capacity(out_w * out_h * 4);

        let read_val = |x: usize, y: usize, c: usize| -> f32 {
            let pixel_idx = y * w + x;

            if bits == 16 {
                let byte_idx = pixel_idx * channels * 2 + c * 2;
                if byte_idx + 1 >= raw_bytes.len() {
                    return 0.0;
                }
                let val = u16::from_le_bytes([raw_bytes[byte_idx], raw_bytes[byte_idx + 1]]);
                val as f32 / 65535.0
            } else {
                // 8-bit case
                let byte_idx = pixel_idx * channels + c;
                if byte_idx >= raw_bytes.len() {
                    return 0.0;
                }
                let val = raw_bytes[byte_idx];
                val as f32 / 255.0
            }
        };

        for y in 0..out_h {
            let src_y = y * step;
            for x in 0..out_w {
                let src_x = x * step;

                let r = read_val(src_x, src_y, 0);
                let g = read_val(src_x, src_y, 1);
                let b = read_val(src_x, src_y, 2);

                out_data.push(r);
                out_data.push(g);
                out_data.push(b);
                out_data.push(1.0);
            }
        }

        libraw_sys::libraw_dcraw_clear_mem(processed);
        libraw_sys::libraw_close(raw_data);

        Ok(PreviewContext {
            width: out_w as u32,
            height: out_h as u32,
            data: out_data,
        })
    }
}

#[tauri::command]
fn load_raw(state: State<AppState>, path: &str) -> Result<ImageResult, String> {
    // Preview Target: 1024px
    let preview = process_libraw(path, Some(1024))?;

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
    // Full Export: No target width (Full Res)
    let processed = process_libraw(path, None)?;

    let w = processed.width;
    let h = processed.height;

    let mut imgbuf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w, h);

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let idx = (y * w + x) as usize * 4;
        let r_lin = processed.data[idx];
        let g_lin = processed.data[idx + 1];
        let b_lin = processed.data[idx + 2];

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
    file.write_all(json_val.as_bytes())
        .map_err(|e| e.to_string())?;
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
        .manage(AppState {
            preview_context: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            load_raw,
            export_image,
            save_params,
            load_params
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
