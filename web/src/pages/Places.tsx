import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { ConfirmDialog, Empty, ErrorBanner, Field, InfoBanner } from "../components/ui";
import { api } from "../lib/api";
import type { Place, SemanticTag, Zone, ZoneConfig } from "../lib/types";
import { useApp } from "../state";

const TAGS: [SemanticTag, string][] = [
  ["food_area", "Food bowl"], ["water_area", "Water bowl"], ["exit_door", "Door to outside"], ["rest_area", "Bed / resting spot"],
  ["play_area", "Play area"], ["litter_area", "Litter box / toilet area"], ["window", "Window"], ["other", "Other"],
];

interface Rect { tag: SemanticTag; label: string; x1: number; y1: number; x2: number; y2: number }

export function rectsToZones(rects: Rect[]): ZoneConfig {
  const clamp = (v: number) => Math.max(0, Math.min(1, Number(v.toFixed(4))));
  return {
    schema: "petperception.zones/1", coordinate_space: "normalized_image", anchor: "bottom_center",
    zones: rects.map((r, i) => {
      const x1 = clamp(Math.min(r.x1, r.x2)), x2 = clamp(Math.max(r.x1, r.x2)), y1 = clamp(Math.min(r.y1, r.y2)), y2 = clamp(Math.max(r.y1, r.y2));
      return { zone_id: `${r.tag}-${i + 1}`, label: r.label || r.tag, semantic_tag: r.tag, polygon: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] } as Zone;
    }),
  };
}

function zonesToRects(z: ZoneConfig): Rect[] {
  return z.zones.map((zone) => {
    const xs = zone.polygon.map((p) => p[0]), ys = zone.polygon.map((p) => p[1]);
    return { tag: zone.semantic_tag, label: zone.label, x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) };
  });
}

function ZoneEditor({ image, rects, setRects }: { image: string | null; rects: Rect[]; setRects: (r: Rect[]) => void }) {
  const box = useRef<HTMLDivElement>(null);
  const [tag, setTag] = useState<SemanticTag>("food_area");
  const [drag, setDrag] = useState<Rect | null>(null);
  const pos = (e: PointerEvent) => {
    const r = box.current!.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
  };
  return (
    <div>
      <Field label="Area type to draw" id="zone-tag">
        <select id="zone-tag" value={tag} onChange={(e) => setTag((e.target as HTMLSelectElement).value as SemanticTag)}>
          {TAGS.map(([t, l]) => <option key={t} value={t}>{l}</option>)}
        </select>
      </Field>
      <p class="hint">Drag on the picture to mark the area. The picture should be from the same camera position you will record from.</p>
      <div ref={box} class="zone-canvas" role="application" aria-label="Zone drawing area"
        onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); const p = pos(e); setDrag({ tag, label: TAGS.find((x) => x[0] === tag)![1], x1: p.x, y1: p.y, x2: p.x, y2: p.y }); }}
        onPointerMove={(e) => { if (drag) { const p = pos(e); setDrag({ ...drag, x2: p.x, y2: p.y }); } }}
        onPointerUp={() => { if (drag && Math.abs(drag.x2 - drag.x1) > 0.02 && Math.abs(drag.y2 - drag.y1) > 0.02) setRects([...rects, drag]); setDrag(null); }}>
        {image ? <img src={image} alt="Reference view" draggable={false} /> : <div class="zone-placeholder">No picture - draw on the blank frame</div>}
        {[...rects, ...(drag ? [drag] : [])].map((r, i) => (
          <div key={i} class={`zone-rect zone-${r.tag}`} style={{ left: `${100 * Math.min(r.x1, r.x2)}%`, top: `${100 * Math.min(r.y1, r.y2)}%`,
            width: `${100 * Math.abs(r.x2 - r.x1)}%`, height: `${100 * Math.abs(r.y2 - r.y1)}%` }}>
            <span>{r.label}</span>
          </div>
        ))}
      </div>
      <ul class="list">
        {rects.map((r, i) => (
          <li key={i} class="list-item">
            <span>{r.label}</span>
            <button type="button" class="btn btn-link" onClick={() => setRects(rects.filter((_, j) => j !== i))}>Remove</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function PlacesPage() {
  const { places, refreshPlaces, toast } = useApp();
  const [editing, setEditing] = useState<{ id?: string; name: string; rects: Rect[] } | null>(null);
  const [image, setImage] = useState<Blob | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [toDelete, setToDelete] = useState<Place | null>(null);
  const imageUrl = useMemo(() => (image ? URL.createObjectURL(image) : null), [image]);
  useEffect(() => () => { if (imageUrl) URL.revokeObjectURL(imageUrl); }, [imageUrl]);

  const save = async (e: Event) => {
    e.preventDefault();
    if (!editing) return;
    try {
      const body = { name: editing.name, zones: rectsToZones(editing.rects) };
      if (editing.id) await api.patch(`/places/${editing.id}`, body);
      else await api.post("/places", body);
      await refreshPlaces();
      toast("Place saved.");
      setEditing(null);
      setImage(null);
    } catch (err) {
      setError(err);
    }
  };

  return (
    <section>
      <h1 tabIndex={-1}>Places</h1>
      <InfoBanner>
        A place records where things are in one camera view (bowl, door, bed). Being near the bowl is never treated as hunger on its own;
        it only adds context to sounds or behavior seen at the same time.
      </InfoBanner>
      {places.length === 0 && !editing && <Empty title="No places yet" />}
      <ul class="list">
        {places.map((p) => (
          <li key={p.id} class="list-item">
            <div><strong>{p.name}</strong> <span class="muted">{p.zones.zones.map((z) => z.label).join(", ") || "no areas"}</span></div>
            <div class="row">
              <button type="button" class="btn btn-secondary" onClick={() => setEditing({ id: p.id, name: p.name, rects: zonesToRects(p.zones) })}>Edit</button>
              <button type="button" class="btn btn-link" onClick={() => setToDelete(p)}>Delete</button>
            </div>
          </li>
        ))}
      </ul>
      {!editing && <button type="button" class="btn btn-primary" onClick={() => setEditing({ name: "", rects: [] })}>Add a place</button>}
      {editing && (
        <form class="card" onSubmit={save}>
          <h2>{editing.id ? "Edit place" : "New place"}</h2>
          <Field label="Name" id="place-name">
            <input id="place-name" required maxLength={60} value={editing.name} placeholder="e.g. Kitchen camera"
              onInput={(e) => setEditing({ ...editing, name: (e.target as HTMLInputElement).value })} />
          </Field>
          <Field label="Reference picture (stays on this device)" id="place-img" hint="Take or choose a photo from the camera position.">
            <input id="place-img" type="file" accept="image/*" capture="environment" onChange={(e) => setImage((e.target as HTMLInputElement).files?.[0] ?? null)} />
          </Field>
          <ZoneEditor image={imageUrl} rects={editing.rects} setRects={(r) => setEditing({ ...editing, rects: r })} />
          <ErrorBanner error={error} />
          <div class="row">
            <button type="submit" class="btn btn-primary">Save place</button>
            <button type="button" class="btn btn-secondary" onClick={() => { setEditing(null); setImage(null); }}>Cancel</button>
          </div>
        </form>
      )}
      <ConfirmDialog open={toDelete !== null} title={`Delete ${toDelete?.name ?? ""}?`} danger confirmLabel="Delete" body={<p>Past analyses keep the zones they used.</p>}
        onCancel={() => setToDelete(null)}
        onConfirm={async () => {
          const p = toDelete!;
          setToDelete(null);
          try {
            await api.del(`/places/${p.id}`);
            await refreshPlaces();
          } catch (e) {
            setError(e);
          }
        }} />
    </section>
  );
}
