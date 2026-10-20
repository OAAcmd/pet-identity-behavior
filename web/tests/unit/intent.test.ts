import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { f32FromB64 } from "../../src/engine/assets";
import { IntentModels, pool, type IntentFrame } from "../../src/engine/intent";

// Python reference: python -m petperception.intent.parity (src/petperception/intent/parity.py).
const fx = JSON.parse(readFileSync(resolve(process.cwd(), "tests/unit/fixtures/intent_parity.json"), "utf-8"));

describe("intent inference matches the Python reference", () => {
  const models = new IntentModels(fx.assets);
  for (const c of fx.cases) {
    it(c.name, () => {
      const frames: IntentFrame[] | null = c.frames
        ? c.frames.map((f: any) => ({ t: f.t, whole: f32FromB64(f.whole_b64), objects: f32FromB64(f.objects_b64), pet: f.pet, box: f.box,
          crop: f.crop_b64 ? f32FromB64(f.crop_b64) : null, person: f.person, motion: f.motion }))
        : null;
      const audio = c.audio ? { scores: c.audio.scores_b64.map((s: string) => f32FromB64(s)), rmsDb: c.audio.rmsDb } : null;
      const { branches, cues } = pool(frames, audio, c.videoState, c.audioState);
      for (const [b, ref] of Object.entries(c.expected.branches)) {
        const got = (branches as any)[b];
        if (ref === null) { expect(got).toBeNull(); continue; }
        const r = f32FromB64(ref as string);
        expect(got.length).toBe(r.length);
        let maxErr = 0;
        for (let i = 0; i < r.length; i++) maxErr = Math.max(maxErr, Math.abs(got[i] - r[i]) / Math.max(1, Math.abs(r[i])));
        expect(maxErr).toBeLessThan(1e-5);
      }
      const res = models.infer(branches, cues, fx.yamnet_labels);
      const e = c.expected;
      expect(res.species).toBe(e.species);
      expect(res.status).toBe(e.status);
      expect(res.missing_inputs).toEqual(e.missing_inputs);
      expect(res.conflicts).toEqual(e.conflicts);
      expect(res.qualifiers).toEqual(e.qualifiers);
      expect(res.situation === null).toBe(e.situation === null);
      if (e.situation) {
        expect(res.situation!.label).toBe(e.situation.label);
        expect(res.situation!.status).toBe(e.situation.status);
        expect(res.situation!.score).toBeCloseTo(e.situation.score, 3);
      }
      expect(res.readings.map((r) => [r.channel, r.top, r.confident])).toEqual(e.readings.map((r: any) => [r.channel, r.top, r.confident]));
      res.readings.forEach((r, i) => {
        for (const [k, v] of Object.entries(e.readings[i].scores)) expect(r.scores[k]).toBeCloseTo(v as number, 3);
      });
      expect(res.hypotheses.map((h) => [h.name, h.status])).toEqual(e.hypotheses.map((h: any) => [h.name, h.status]));
    });
  }
});
