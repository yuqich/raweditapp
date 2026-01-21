import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import "./App.css";

interface ImageResult {
  width: number;
  height: number;
  data: number[]; // Linear RGB Float array (flat)
}

interface WebGLParams {
  exposure: number;
  contrast: number;
  temperature: number;
  tint: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
  saturation: number;
}

interface HistogramData {
  r: number[];
  g: number[];
  b: number[];
  l: number[];
}

// Shader Sources
const VS_SOURCE = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;
  void main() {
    gl_Position = vec4(a_position, 0, 1);
    v_texCoord = a_texCoord;
  }
`;

const FS_SOURCE = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_image;
  
  uniform float u_exposure;
  uniform float u_contrast;
  uniform vec3 u_whiteBalance;
  
  // Advanced
  uniform float u_saturation;
  uniform float u_highlights;
  uniform float u_shadows;
  uniform float u_whites;
  uniform float u_blacks;
  
  float getLuminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
  }
  
  void main() {
    vec4 color = texture2D(u_image, v_texCoord);
    vec3 rgb = color.rgb;
    
    // 1. White Balance
    rgb = rgb * u_whiteBalance;
    
    // 2. Exposure
    float exposureMult = pow(2.0, u_exposure);
    rgb = rgb * exposureMult;
    
    // 3. Contrast
    float contrastFactor = (1.0 + u_contrast) * (1.0 + u_contrast);
    rgb = (rgb - 0.5) * contrastFactor + 0.5;
    
    // --- Advanced Tone Mapping ---
    float luma = getLuminance(rgb);
    
    // Shadows/Highlights
    float shadowMask = 1.0 - smoothstep(0.0, 0.6, luma);
    float highlightMask = smoothstep(0.4, 1.0, luma);
    
    if (u_shadows != 0.0) {
        float shadowLift = pow(2.0, u_shadows) - 1.0;
        rgb += rgb * shadowLift * shadowMask * 0.5;
    }
    
    if (u_highlights != 0.0) {
        float highlightGain = pow(2.0, u_highlights) - 1.0;
        rgb += rgb * highlightGain * highlightMask * 0.5;
    }
    
    // Levels
    float blackPoint = u_blacks * 0.2;
    float whitePoint = 1.0 + u_whites * 0.2;
    if (whitePoint - blackPoint < 0.001) whitePoint = blackPoint + 0.001;
    rgb = (rgb - blackPoint) / (whitePoint - blackPoint);

    // Saturation
    luma = getLuminance(rgb);
    vec3 grey = vec3(luma);
    float satMult = 1.0 + u_saturation;
    rgb = mix(grey, rgb, satMult);
    
    // Gamma
    float gamma = 1.0 / 2.2;
    rgb = pow(max(rgb, 0.0), vec3(gamma));
    
    gl_FragColor = vec4(rgb, 1.0);
  }
`;

// --- Histogram Calculation (CPU JS) ---
function calculateHistogram(image: ImageResult, params: WebGLParams): HistogramData {
  const buckets = 256;
  const hist = {
    r: new Array(buckets).fill(0),
    g: new Array(buckets).fill(0),
    b: new Array(buckets).fill(0),
    l: new Array(buckets).fill(0),
  };

  const data = image.data;
  const step = 20; // 5% sampling

  const ratio = (params.temperature - 5500.0) / 5500.0;
  const wb_r = 1.0 + Math.max(0, ratio);
  const wb_b = 1.0 - Math.min(0, ratio);
  const wb_g = 1.0 + params.tint / 100.0;

  const exposureMult = Math.pow(2.0, params.exposure);
  const contrastFactor = (1.0 + params.contrast) * (1.0 + params.contrast);
  const shadowLift = params.shadows !== 0 ? Math.pow(2.0, params.shadows) - 1.0 : 0;
  const highlightGain = params.highlights !== 0 ? Math.pow(2.0, params.highlights) - 1.0 : 0;

  const blackPoint = params.blacks * 0.2;
  const whitePoint = 1.0 + params.whites * 0.2;
  const range = (whitePoint - blackPoint) < 0.001 ? 0.001 : (whitePoint - blackPoint);
  const satMult = 1.0 + params.saturation;
  const gammaInv = 1.0 / 2.2;

  // Data stride is 4 because backend sends RGBA
  for (let i = 0; i < data.length; i += 4 * step) {
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];

    // WB
    r *= wb_r; g *= wb_g; b *= wb_b;

    // Exposure
    r *= exposureMult; g *= exposureMult; b *= exposureMult;

    // Contrast
    r = (r - 0.5) * contrastFactor + 0.5;
    g = (g - 0.5) * contrastFactor + 0.5;
    b = (b - 0.5) * contrastFactor + 0.5;

    // Advanced
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    if (shadowLift !== 0 || highlightGain !== 0) {
      let t_s = luma / 0.6;
      t_s = t_s < 0 ? 0 : (t_s > 1 ? 1 : t_s * t_s * (3 - 2 * t_s));
      let shadowMask = 1.0 - t_s;

      let t_h = (luma - 0.4) / 0.6;
      let highlightMask = t_h < 0 ? 0 : (t_h > 1 ? 1 : t_h * t_h * (3 - 2 * t_h));

      if (shadowLift !== 0) {
        let lift = shadowLift * shadowMask * 0.5;
        r += r * lift; g += g * lift; b += b * lift;
      }
      if (highlightGain !== 0) {
        let gain = highlightGain * highlightMask * 0.5;
        r += r * gain; g += g * gain; b += b * gain;
      }
    }

    // Levels
    r = (r - blackPoint) / range;
    g = (g - blackPoint) / range;
    b = (b - blackPoint) / range;

    // Saturation
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    r = luma + (r - luma) * satMult;
    g = luma + (g - luma) * satMult;
    b = luma + (b - luma) * satMult;

    // Gamma
    r = r > 0 ? Math.pow(r, gammaInv) : 0;
    g = g > 0 ? Math.pow(g, gammaInv) : 0;
    b = b > 0 ? Math.pow(b, gammaInv) : 0;

    const ir = Math.min(255, Math.max(0, Math.floor(r * 255)));
    const ig = Math.min(255, Math.max(0, Math.floor(g * 255)));
    const ib = Math.min(255, Math.max(0, Math.floor(b * 255)));

    hist.r[ir]++;
    hist.g[ig]++;
    hist.b[ib]++;
    hist.l[Math.floor((ir + ig + ib) / 3)]++;
  }
  return hist;
}

function Histogram({ data }: { data: HistogramData | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    const w = canvasRef.current.width;
    const h = canvasRef.current.height;
    ctx.clearRect(0, 0, w, h);

    let max = 0;
    for (let i = 0; i < 256; i++) {
      max = Math.max(max, data.r[i], data.g[i], data.b[i]);
    }
    if (max === 0) return;

    const drawChannel = (arr: number[], color: string, composite: GlobalCompositeOperation = 'screen') => {
      ctx.globalCompositeOperation = composite;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let i = 0; i < 256; i++) {
        const val = arr[i] / max;
        const x = (i / 255) * w;
        const y = h - (val * h);
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.fill();
    };

    drawChannel(data.r, 'rgba(255,0,0,0.5)');
    drawChannel(data.g, 'rgba(0,255,0,0.5)');
    drawChannel(data.b, 'rgba(0,0,255,0.5)');
    drawChannel(data.l, 'rgba(200,200,200,0.3)', 'source-over');

    ctx.globalCompositeOperation = 'source-over';
  }, [data]);

  return <canvas ref={canvasRef} width={256} height={100} style={{ width: '100%', height: '100px', background: '#222', borderRadius: '4px', marginBottom: '10px' }} />;
}

// GL Helpers
function createShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext, vs: string, fs: string) {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vs);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fs);
  if (!vertexShader || !fragmentShader) return null;

  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

function WebGLViewer({ image, params }: { image: ImageResult | null, params: WebGLParams }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const frameIdRef = useRef<number>(0);

  // Initialize GL
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) {
      console.error("WebGL not supported");
      return;
    }
    glRef.current = gl as WebGLRenderingContext;

    if (!(gl instanceof WebGL2RenderingContext)) {
      gl.getExtension("OES_texture_float");
    }

    const program = createProgram(gl as WebGLRenderingContext, VS_SOURCE, FS_SOURCE);
    if (!program) {
      console.error("Shader compile failed");
      return;
    }
    programRef.current = program;

    try {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      textureRef.current = texture;
    } catch (e: any) {
      console.error("Setup Error:", e);
    }
  }, []);

  // Upload Texture
  useEffect(() => {
    const gl = glRef.current;
    if (!gl || !image || !textureRef.current) return;

    try {
      gl.bindTexture(gl.TEXTURE_2D, textureRef.current);
      const data = new Float32Array(image.data);

      let internalFormat;
      if (gl instanceof WebGL2RenderingContext) {
        internalFormat = gl.RGBA32F;
      } else {
        internalFormat = gl.RGBA;
      }

      gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, image.width, image.height, 0, gl.RGBA, gl.FLOAT, data);

      if (canvasRef.current) {
        canvasRef.current.width = image.width;
        canvasRef.current.height = image.height;
        gl.viewport(0, 0, image.width, image.height);
      }
    } catch (e: any) {
      console.error("Upload Exception:", e);
    }
  }, [image]);

  // Render Loop
  useEffect(() => {
    const gl = glRef.current;
    const program = programRef.current;

    if (!gl || !program) return;

    const render = () => {
      if (!image || !textureRef.current) {
        gl.clearColor(0.0, 0.0, 0.0, 0.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        frameIdRef.current = requestAnimationFrame(render);
        return;
      }

      gl.useProgram(program);

      const positionLoc = gl.getAttribLocation(program, "a_position");
      const texCoordLoc = gl.getAttribLocation(program, "a_texCoord");

      const pBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, pBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(positionLoc);
      gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

      const tBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, tBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(texCoordLoc);
      gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);

      const ratio = (params.temperature - 5500.0) / 5500.0;
      const wb_r = 1.0 + Math.max(0, ratio);
      const wb_b = 1.0 - Math.min(0, ratio);
      const wb_g = 1.0 + params.tint / 100.0;

      gl.uniform1f(gl.getUniformLocation(program, "u_exposure"), params.exposure);
      gl.uniform1f(gl.getUniformLocation(program, "u_contrast"), params.contrast);
      gl.uniform3f(gl.getUniformLocation(program, "u_whiteBalance"), wb_r, wb_g, wb_b);

      gl.uniform1f(gl.getUniformLocation(program, "u_highlights"), params.highlights);
      gl.uniform1f(gl.getUniformLocation(program, "u_shadows"), params.shadows);
      gl.uniform1f(gl.getUniformLocation(program, "u_whites"), params.whites);
      gl.uniform1f(gl.getUniformLocation(program, "u_blacks"), params.blacks);
      gl.uniform1f(gl.getUniformLocation(program, "u_saturation"), params.saturation);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, textureRef.current);
      gl.uniform1i(gl.getUniformLocation(program, "u_image"), 0);

      gl.drawArrays(gl.TRIANGLES, 0, 6);

      gl.deleteBuffer(pBuffer);
      gl.deleteBuffer(tBuffer);

      frameIdRef.current = requestAnimationFrame(render);
    };

    frameIdRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(frameIdRef.current);

  }, [image, params]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#1a1a1a' }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", objectFit: "contain", display: 'block' }} />
      {!image && (
        <div style={{
          position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          color: '#666', fontSize: '1.2rem', pointerEvents: 'none'
        }}>
          No Image Loaded
        </div>
      )}
    </div>
  );
}

function App() {
  const [imageResult, setImageResult] = useState<ImageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [histData, setHistData] = useState<HistogramData | null>(null);

  const [params, setParams] = useState<WebGLParams>({
    exposure: 0.0,
    contrast: 0.0,
    temperature: 5500.0,
    tint: 0.0,
    highlights: 0.0,
    shadows: 0.0,
    whites: 0.0,
    blacks: 0.0,
    saturation: 0.0,
  });

  const handleOpenFile = async () => {
    try {
      const file = await open({
        multiple: false,
        directory: false,
        filters: [{
          name: 'RAW Images',
          extensions: ['arw', 'cr2', 'nef', 'dng', 'raf', 'orf']
        }]
      });

      if (file) {
        setLoading(true);
        setError(null);
        try {
          const data = await invoke<ImageResult>("load_raw", { path: file as string });
          setImageResult(data);

          setParams({
            exposure: 0.0,
            contrast: 0.0,
            temperature: 5500.0,
            tint: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            saturation: 0.0,
          });

        } catch (e: any) {
          console.error(e);
          setError("Failed to load image: " + e);
        } finally {
          setLoading(false);
        }
      }
    } catch (err: any) {
      setError(err.toString());
    }
  };

  const handleParamChange = (key: string, value: number) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  // Histogram Calc
  useEffect(() => {
    if (!imageResult) {
      setHistData(null);
      return;
    }
    const timer = setTimeout(() => {
      const h = calculateHistogram(imageResult, params);
      setHistData(h);
    }, 50);
    return () => clearTimeout(timer);
  }, [imageResult, params]);

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-title">RAW Editor (WebGL - GPU)</div>
        <button onClick={handleOpenFile} disabled={loading} className="primary">
          {loading ? "Loading..." : "Open File"}
        </button>
      </header>

      <div className="main-content">
        <div className="image-area">
          {error && <div style={{ color: 'red', position: 'absolute', top: 20 }}>{error}</div>}
          <WebGLViewer image={imageResult} params={params} />
        </div>

        <aside className="sidebar" style={{ overflowY: 'auto' }}>
          <h3>Histogram</h3>
          <Histogram data={histData} />

          <h3>Basic</h3>

          <div className="control-group">
            <label className="control-label">Temperature ({params.temperature}K)</label>
            <input
              type="range" min="2000" max="10000" step="100"
              value={params.temperature}
              onChange={(e) => handleParamChange('temperature', parseFloat(e.target.value))}
              disabled={!imageResult}
            />
          </div>

          <div className="control-group">
            <label className="control-label">Tint ({params.tint})</label>
            <input
              type="range" min="-50" max="50" step="1"
              value={params.tint}
              onChange={(e) => handleParamChange('tint', parseFloat(e.target.value))}
              disabled={!imageResult}
            />
          </div>

          <div className="control-group">
            <label className="control-label">Exposure ({params.exposure.toFixed(1)})</label>
            <input
              type="range" min="-3" max="3" step="0.1"
              value={params.exposure}
              onChange={(e) => handleParamChange('exposure', parseFloat(e.target.value))}
              disabled={!imageResult}
            />
          </div>

          <div className="control-group">
            <label className="control-label">Contrast ({params.contrast.toFixed(1)})</label>
            <input
              type="range" min="-0.5" max="0.5" step="0.05"
              value={params.contrast}
              onChange={(e) => handleParamChange('contrast', parseFloat(e.target.value))}
              disabled={!imageResult}
            />
          </div>

          <h3 style={{ marginTop: '20px' }}>Light</h3>

          <div className="control-group">
            <label className="control-label">Highlights ({params.highlights.toFixed(2)})</label>
            <input
              type="range" min="-1" max="1" step="0.05"
              value={params.highlights}
              onChange={(e) => handleParamChange('highlights', parseFloat(e.target.value))}
              disabled={!imageResult}
            />
          </div>

          <div className="control-group">
            <label className="control-label">Shadows ({params.shadows.toFixed(2)})</label>
            <input
              type="range" min="-1" max="1" step="0.05"
              value={params.shadows}
              onChange={(e) => handleParamChange('shadows', parseFloat(e.target.value))}
              disabled={!imageResult}
            />
          </div>

          <div className="control-group">
            <label className="control-label">Whites ({params.whites.toFixed(2)})</label>
            <input
              type="range" min="-1" max="1" step="0.05"
              value={params.whites}
              onChange={(e) => handleParamChange('whites', parseFloat(e.target.value))}
              disabled={!imageResult}
            />
          </div>

          <div className="control-group">
            <label className="control-label">Blacks ({params.blacks.toFixed(2)})</label>
            <input
              type="range" min="-1" max="1" step="0.05"
              value={params.blacks}
              onChange={(e) => handleParamChange('blacks', parseFloat(e.target.value))}
              disabled={!imageResult}
            />
          </div>

          <h3 style={{ marginTop: '20px' }}>Color</h3>

          <div className="control-group">
            <label className="control-label">Saturation ({params.saturation.toFixed(2)})</label>
            <input
              type="range" min="-1" max="1" step="0.05"
              value={params.saturation}
              onChange={(e) => handleParamChange('saturation', parseFloat(e.target.value))}
              disabled={!imageResult}
            />
          </div>


          <div style={{ marginTop: 'auto', fontSize: '0.8rem', color: '#666', paddingTop: '20px' }}>
            Rust + React + Tauri + WebGL
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
