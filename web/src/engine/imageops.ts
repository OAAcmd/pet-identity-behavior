// Image operations mirroring the OpenCV calls of the Python reference.
// Differences that remain (documented in docs/ON_DEVICE.md and measured by the
// engine parity test): OpenCV uses fixed-point arithmetic for 8-bit bilinear
// and warpAffine, so individual pixels may differ by 1; frame decoding and
// down-scaling are done by the browser instead of FFmpeg.

export interface RGBImage {
  width: number;
  height: number;
  data: Uint8Array; // packed RGB, row-major
}

export function fromRGBA(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): RGBImage {
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < width * height; i++, j += 4) {
    out[i * 3] = rgba[j];
    out[i * 3 + 1] = rgba[j + 1];
    out[i * 3 + 2] = rgba[j + 2];
  }
  return { width, height, data: out };
}

/** cv2.resize(..., INTER_LINEAR) semantics (half-pixel centres, edge clamp), rounded to uint8. */
export function resizeBilinear(img: RGBImage, w: number, h: number): RGBImage {
  const out = new Uint8Array(w * h * 3);
  const sx = img.width / w;
  const sy = img.height / h;
  const x0s = new Int32Array(w), x1s = new Int32Array(w), fxs = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let fx = (x + 0.5) * sx - 0.5;
    let x0 = Math.floor(fx);
    fx -= x0;
    if (x0 < 0) { x0 = 0; fx = 0; }
    if (x0 >= img.width - 1) { x0 = img.width - 1; fx = 0; }
    x0s[x] = x0; x1s[x] = Math.min(x0 + 1, img.width - 1); fxs[x] = fx;
  }
  for (let y = 0; y < h; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    let y0 = Math.floor(fy);
    fy -= y0;
    if (y0 < 0) { y0 = 0; fy = 0; }
    if (y0 >= img.height - 1) { y0 = img.height - 1; fy = 0; }
    const y1 = Math.min(y0 + 1, img.height - 1);
    const r0 = y0 * img.width * 3, r1 = y1 * img.width * 3;
    for (let x = 0; x < w; x++) {
      const a = x0s[x] * 3, b = x1s[x] * 3, fx = fxs[x];
      for (let c = 0; c < 3; c++) {
        const top = img.data[r0 + a + c] * (1 - fx) + img.data[r0 + b + c] * fx;
        const bot = img.data[r1 + a + c] * (1 - fx) + img.data[r1 + b + c] * fx;
        out[(y * w + x) * 3 + c] = Math.round(top * (1 - fy) + bot * fy);
      }
    }
  }
  return { width: w, height: h, data: out };
}

/** cv2 INTER_AREA for down-scaling (fractional-coverage box filter, separable). */
export function resizeArea(img: RGBImage, w: number, h: number): RGBImage {
  if (w > img.width || h > img.height) return resizeBilinear(img, w, h);
  const weights = (n: number, m: number) => {
    const scale = n / m;
    const taps: { i: number; w: number }[][] = [];
    for (let d = 0; d < m; d++) {
      const s0 = d * scale, s1 = (d + 1) * scale;
      const list: { i: number; w: number }[] = [];
      for (let i = Math.floor(s0); i < Math.min(n, Math.ceil(s1)); i++) {
        const cov = Math.min(s1, i + 1) - Math.max(s0, i);
        if (cov > 1e-9) list.push({ i, w: cov / scale });
      }
      taps.push(list);
    }
    return taps;
  };
  const tx = weights(img.width, w), ty = weights(img.height, h);
  const tmp = new Float32Array(img.height * w * 3);
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (const t of tx[x]) {
        const k = (y * img.width + t.i) * 3;
        r += img.data[k] * t.w; g += img.data[k + 1] * t.w; b += img.data[k + 2] * t.w;
      }
      const o = (y * w + x) * 3;
      tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
    }
  }
  const out = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (const t of ty[y]) {
        const k = (t.i * w + x) * 3;
        r += tmp[k] * t.w; g += tmp[k + 1] * t.w; b += tmp[k + 2] * t.w;
      }
      const o = (y * w + x) * 3;
      out[o] = Math.min(255, Math.round(r)); out[o + 1] = Math.min(255, Math.round(g)); out[o + 2] = Math.min(255, Math.round(b));
    }
  }
  return { width: w, height: h, data: out };
}

/** numpy slicing rgb[int(y1*h):max(int(y2*h), int(y1*h)+1), int(x1*w):...] */
export function cropBox(img: RGBImage, box: ArrayLike<number>): RGBImage {
  const x1 = Math.trunc(box[0] * img.width), y1 = Math.trunc(box[1] * img.height);
  const x2 = Math.min(img.width, Math.max(Math.trunc(box[2] * img.width), x1 + 1));
  const y2 = Math.min(img.height, Math.max(Math.trunc(box[3] * img.height), y1 + 1));
  const w = Math.max(1, x2 - x1), h = Math.max(1, y2 - y1);
  const out = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const src = ((y1 + y) * img.width + x1) * 3;
    out.set(img.data.subarray(src, src + w * 3), y * w * 3);
  }
  return { width: w, height: h, data: out };
}

/** cv2.warpAffine square crop (scale/translation only), bilinear, constant border 127. */
export function squareCrop(img: RGBImage, box: ArrayLike<number>, scale = 1.15, size = 224): { crop: RGBImage; x0: number; y0: number; side: number } {
  const x1 = box[0] * img.width, y1 = box[1] * img.height, x2 = box[2] * img.width, y2 = box[3] * img.height;
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  const side = Math.max(x2 - x1, y2 - y1, 8) * scale;
  const x0 = cx - side / 2, y0 = cy - side / 2;
  const k = side / size;
  const out = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++) {
    const sy = y0 + y * k;
    const yi = Math.floor(sy), fy = sy - yi;
    for (let x = 0; x < size; x++) {
      const sx = x0 + x * k;
      const xi = Math.floor(sx), fx = sx - xi;
      for (let c = 0; c < 3; c++) {
        const px = (xx: number, yy: number) => (xx < 0 || yy < 0 || xx >= img.width || yy >= img.height ? 127 : img.data[(yy * img.width + xx) * 3 + c]);
        const v = (px(xi, yi) * (1 - fx) + px(xi + 1, yi) * fx) * (1 - fy) + (px(xi, yi + 1) * (1 - fx) + px(xi + 1, yi + 1) * fx) * fy;
        out[(y * size + x) * 3 + c] = Math.round(v);
      }
    }
  }
  return { crop: { width: size, height: size, data: out }, x0, y0, side };
}

export function toFloat(img: RGBImage, mean: number, std: number): Float32Array {
  const out = new Float32Array(img.data.length);
  for (let i = 0; i < out.length; i++) out[i] = (img.data[i] - mean) / std;
  return out;
}

/** cv2.cvtColor(RGB2GRAY) 8-bit fixed point. */
export function grayscale(img: RGBImage): Uint8Array {
  const n = img.width * img.height;
  const g = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const r = img.data[i * 3], gg = img.data[i * 3 + 1], b = img.data[i * 3 + 2];
    g[i] = (r * 4899 + gg * 9617 + b * 1868 + 8192) >> 14;
  }
  return g;
}

/** Mean luma and variance of cv2.Laplacian(ksize=1, BORDER_REFLECT_101). */
export function lumaAndLaplacianVar(gray: Uint8Array, w: number, h: number): { luma: number; lapVar: number } {
  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const ref = (v: number, n: number) => (v < 0 ? -v : v >= n ? 2 * n - v - 2 : v);
  let s1 = 0, s2 = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = gray[y * w + x];
      const l = gray[y * w + ref(x - 1, w)], r = gray[y * w + ref(x + 1, w)];
      const u = gray[ref(y - 1, h) * w + x], d = gray[ref(y + 1, h) * w + x];
      const v = l + r + u + d - 4 * c;
      s1 += v; s2 += v * v;
    }
  }
  const n = w * h;
  const mean = s1 / n;
  return { luma: sum / n, lapVar: s2 / n - mean * mean };
}
