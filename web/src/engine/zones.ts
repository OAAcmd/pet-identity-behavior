// Owner-drawn zones: membership, dwell intervals, distance (mirrors context/zones.py).

export interface Zone { zone_id: string; semantic_tag: string; label: string; polygon: [number, number][] }
export interface ZoneConfig { schema?: string; anchor: "center" | "bottom_center"; zones: Zone[] }

export function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-12) + xi) inside = !inside;
  }
  return inside;
}

export function anchorPoint(b: ArrayLike<number>, anchor: string): [number, number] {
  return [(b[0] + b[2]) / 2, anchor === "bottom_center" ? b[3] : (b[1] + b[3]) / 2];
}

export interface Occupancy { zone_id: string; semantic_tag: string; label: string; intervals_s: [number, number][]; dwell_s: number; fraction_of_track_detections: number; anchor: string }

export function occupancy(times: number[], boxes: ArrayLike<number>[], cfg: ZoneConfig, maxGap: number): Occupancy[] {
  const out: Occupancy[] = [];
  for (const z of cfg.zones) {
    const inside = boxes.map((b) => { const [x, y] = anchorPoint(b, cfg.anchor); return pointInPolygon(x, y, z.polygon); });
    const iv: [number, number][] = [];
    times.forEach((t, i) => {
      if (!inside[i]) return;
      if (iv.length && t - iv[iv.length - 1][1] <= maxGap) iv[iv.length - 1][1] = t;
      else iv.push([t, t]);
    });
    const n = inside.filter(Boolean).length;
    if (n) out.push({ zone_id: z.zone_id, semantic_tag: z.semantic_tag, label: z.label, intervals_s: iv, dwell_s: iv.reduce((s, [a, b]) => s + b - a, 0),
      fraction_of_track_detections: n / boxes.length, anchor: cfg.anchor });
  }
  return out;
}

/** Distance (scaled units) from a point to a polygon; 0 inside or on the edge (cv2.pointPolygonTest semantics). */
export function zoneDistance(pt: [number, number], poly: [number, number][], sx: number, sy: number): number {
  const P = poly.map(([x, y]) => [x * sx, y * sy] as [number, number]);
  const px = pt[0] * sx, py = pt[1] * sy;
  let best = Infinity;
  for (let i = 0, j = P.length - 1; i < P.length; j = i++) {
    const [ax, ay] = P[j], [bx, by] = P[i];
    const dx = bx - ax, dy = by - ay;
    const L = dx * dx + dy * dy;
    const t = L > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L)) : 0;
    best = Math.min(best, Math.hypot(px - (ax + t * dx), py - (ay + t * dy)));
  }
  if (best < 1e-12) return 0;
  return pointInPolygon(px, py, P) ? 0 : best;
}
