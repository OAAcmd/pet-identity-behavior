import { describe, expect, it } from "vitest";
import { vocalConflictTags, vocalTags } from "../../src/engine/audio";

// Same cases as tests/test_e21_vocal_types.py and tests/test_subtype_gate.py (Python reference).
const all = new Set(["hiss", "growl", "purr", "bark", "howl"]);
const tags = (label: string, sub: Record<string, number>, shown: Set<string> = all) => [...vocalTags(label, sub, shown)].sort();

describe("vocal tags", () => {
  it("follows the shared rule when every subtype is validated", () => {
    expect(tags("cat_vocalization", {})).toEqual(["vocalization"]);
    expect(tags("cat_vocalization", { meow: 0.9 })).toEqual(["vocalization"]);
    expect(tags("cat_vocalization", { hiss: 0.5 })).toEqual(["hiss"]);
    expect(tags("cat_vocalization", { hiss: 0.29 })).toEqual(["vocalization"]);
    expect(tags("dog_vocalization", { bark: 0.4, growl: 0.35 })).toEqual(["bark", "growl", "vocalization"]);
    expect(tags("growl", {})).toEqual(["growl"]);
  });
  it("drops unvalidated subtypes but keeps them from making an event request-like", () => {
    const shown = new Set(["meow", "bark"]);
    expect(tags("cat_vocalization", { purr: 0.8 }, shown)).toEqual([]);
    expect(tags("cat_vocalization", { hiss: 0.6, meow: 0.5 }, shown)).toEqual(["vocalization"]);
    expect(tags("dog_vocalization", { bark: 0.5, growl: 0.4 }, shown)).toEqual(["bark", "vocalization"]);
    expect(tags("growl", { growl: 0.7 }, shown)).toEqual([]);
  });
});

describe("conflict-only tags", () => {
  it("are the confident unvalidated subtypes, and an unvalidated growl event", () => {
    const shown = new Set(["meow", "bark"]);
    expect([...vocalConflictTags("cat_vocalization", { purr: 0.8, meow: 0.9, hiss: 0.2 }, shown)].sort()).toEqual(["purr"]);
    expect([...vocalConflictTags("growl", { growl: 0.7 }, shown)]).toEqual(["growl"]);
    expect([...vocalConflictTags("dog_vocalization", { bark: 0.9 }, shown)]).toEqual([]);
  });
});
