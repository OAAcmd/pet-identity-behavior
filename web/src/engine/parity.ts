// Deterministic inputs identical to src/petperception/ondevice/parity.py
// (integer formulas + xorshift32), used for model-level numerical parity.

export function xorshift32(seed: number, n: number): Uint32Array {
  const out = new Uint32Array(n);
  let x = seed >>> 0;
  for (let i = 0; i < n; i++) {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    out[i] = x;
  }
  return out;
}

export function syntheticAudio(n = 15600, seed = 12345): Float32Array {
  const r = xorshift32(seed, n);
  const out = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const tri = (((k * 7) % 200) - 100) / 400;
    const sq = Math.floor(k / 40) % 2 === 0 ? 0.1 : -0.1;
    const noise = ((r[k] >>> 16) / 65536 - 0.5) * 0.1;
    out[k] = tri + sq + noise;
  }
  return out;
}

export function syntheticImage(size: number, seed = 777): Uint8Array {
  const r = xorshift32(seed, size * size * 3);
  const out = new Uint8Array(size * size * 3);
  let i = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      for (let c = 0; c < 3; c++, i++) {
        const v = x * 3 + y * 5 + c * 70 + ((Math.floor(x / 16) + Math.floor(y / 16)) % 2) * 60 + ((r[i] >>> 24) % 32);
        out[i] = v % 256;
      }
    }
  }
  return out;
}

export function normalize(u8: Uint8Array, mean: number, std: number): Float32Array {
  const out = new Float32Array(u8.length);
  for (let i = 0; i < u8.length; i++) out[i] = (u8[i] - mean) / std;
  return out;
}
