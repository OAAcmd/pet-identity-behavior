// LiteRT.js wrapper: one runtime per page, models compiled for WASM (XNNPACK)
// or WebGPU. Same .tflite files as the Python reference (ai_edge_litert).
import { loadAndCompile, loadLiteRt, Tensor, type CompiledModel, type TypedArray } from "@litertjs/core";

export type Accel = "wasm" | "webgpu";

let ready: Promise<unknown> | null = null;

export function initRuntime(wasmBase: string, opts: { threads?: boolean } = {}): Promise<unknown> {
  if (!ready) ready = loadLiteRt(wasmBase, { threads: opts.threads ?? false });
  return ready;
}

export interface LoadedModel {
  name: string;
  model: CompiledModel;
  accel: Accel;
  inputs: { name: string; shape: number[]; dtype: string }[];
  outputs: { name: string; shape: number[]; dtype: string }[];
}

export async function compileModel(name: string, bytes: Uint8Array, accel: Accel = "wasm", numThreads?: number): Promise<LoadedModel> {
  const model = await loadAndCompile(bytes, { accelerator: accel, ...(numThreads ? { cpuOptions: { numThreads } } : {}) });
  const info = (d: readonly { name: string; shape: Int32Array; dtype: unknown }[]) =>
    d.map((x) => ({ name: x.name, shape: Array.from(x.shape), dtype: String(x.dtype) }));
  return { name, model, accel, inputs: info(model.getInputDetails()), outputs: info(model.getOutputDetails()) };
}

export type Typed = Float32Array | Uint8Array | Int32Array;

/** Run with positional inputs; returns outputs copied to plain typed arrays. */
export async function runModel(m: LoadedModel, inputs: Typed[]): Promise<Typed[]> {
  const ts = inputs.map((x, i) => new Tensor(x as unknown as TypedArray, m.inputs[i].shape));
  try {
    const outs = await m.model.run(ts);
    const res: Typed[] = [];
    for (const o of outs) {
      res.push(((await o.data()) as Typed).slice());
      o.delete();
    }
    return res;
  } finally {
    ts.forEach((t) => t.delete());
  }
}
