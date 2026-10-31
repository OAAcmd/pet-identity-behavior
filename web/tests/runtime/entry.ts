// Browser entry for the runtime parity test (bundled by esbuild in the spec).
import { normalize, syntheticAudio, syntheticImage } from "../../src/engine/parity";
import { compileModel, initRuntime, runModel, type Accel, type Typed } from "../../src/engine/runtime";

interface CaseSpec { model_file: string; inputs: { file: string; dtype: string; shape: number[] }[]; outputs: { file: string; shape: number[] }[] }

function jsInput(name: string): Typed[] {
  if (name === "yamnet") return [syntheticAudio()];
  if (name === "detector_float") return [normalize(syntheticImage(320), 127.5, 127.5)];
  if (name === "detector_uint8") return [syntheticImage(320)];
  if (name === "backbone") return [normalize(syntheticImage(224), 127, 128)];
  if (name === "detector_f16") return [normalize(syntheticImage(320), 127.5, 127.5)];
  if (name === "detector_int8") return [syntheticImage(320)];
  if (name === "backbone_int8") return [syntheticImage(224)];
  if (name === "detector_tfhub_raw") return [syntheticImage(320)];
  if (name.startsWith("student_")) return [normalize(syntheticImage(224), 0, 1)]; // E33 student: float 0-255
  throw new Error(name);
}

async function bin(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.arrayBuffer();
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

(window as unknown as { runParity: unknown }).runParity = async (opts: { accel: Accel; threads: boolean; repeats: number; only?: string[] }) => {
  const t0 = performance.now();
  await initRuntime("/wasm/", { threads: opts.threads });
  const initMs = performance.now() - t0;
  const manifest = JSON.parse(new TextDecoder().decode(await bin("/parity/manifest.json")));
  const report: Record<string, unknown> = { initMs, accel: opts.accel, threads: opts.threads, crossOriginIsolated: self.crossOriginIsolated,
    hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, cases: {} };
  for (const [name, spec] of Object.entries(manifest.cases as Record<string, CaseSpec>)) {
    if (opts.only && !opts.only.includes(name)) continue;
    const entry: Record<string, unknown> = {};
    try {
      const bytes = new Uint8Array(await bin(`/models/${spec.model_file}`));
      const tc = performance.now();
      const m = await compileModel(name, bytes, opts.accel);
      entry.compileMs = performance.now() - tc;
      entry.inputs = m.inputs;
      entry.outputs = m.outputs;
      const inputs = jsInput(name);
      // The JS generator must reproduce the Python input bit-exactly.
      const refIn = await bin(`/parity/${spec.inputs[0].file}`);
      const a = new Uint8Array(refIn);
      const b = new Uint8Array(inputs[0].buffer, inputs[0].byteOffset, inputs[0].byteLength);
      entry.inputBitExact = a.length === b.length && a.every((v, i) => v === b[i]);
      const times: number[] = [];
      let outs: Typed[] = [];
      for (let k = 0; k < opts.repeats + 1; k++) {
        const tr = performance.now();
        outs = await runModel(m, inputs);
        if (k > 0) times.push(performance.now() - tr);
      }
      entry.firstRunIncluded = false;
      entry.medianMs = median(times);
      entry.minMs = Math.min(...times);
      const cmp = [];
      for (let i = 0; i < spec.outputs.length; i++) {
        const ref = new Float32Array(await bin(`/parity/${spec.outputs[i].file}`));
        // Match by element count (output order can differ between runtimes).
        const j = outs.findIndex((o, jj) => o.length === ref.length && !cmp.some((c) => c.matchedOutput === jj));
        if (j < 0) { cmp.push({ ref: i, matchedOutput: -1 }); continue; }
        const got = outs[j];
        let maxAbs = 0, sumAbs = 0, maxRef = 0;
        for (let e = 0; e < ref.length; e++) {
          const d = Math.abs(Number(got[e]) - ref[e]);
          maxAbs = Math.max(maxAbs, d); sumAbs += d; maxRef = Math.max(maxRef, Math.abs(ref[e]));
        }
        let argRef = 0, argGot = 0;
        for (let e = 1; e < ref.length; e++) { if (ref[e] > ref[argRef]) argRef = e; if (Number(got[e]) > Number(got[argGot])) argGot = e; }
        cmp.push({ ref: i, matchedOutput: j, elements: ref.length, maxAbsDiff: maxAbs, meanAbsDiff: sumAbs / ref.length, maxAbsRef: maxRef, argmaxEqual: argRef === argGot });
      }
      entry.comparisons = cmp;
      m.model.delete();
    } catch (e) {
      entry.error = String(e instanceof Error ? e.stack ?? e.message : e);
    }
    (report.cases as Record<string, unknown>)[name] = entry;
  }
  return report;
};
