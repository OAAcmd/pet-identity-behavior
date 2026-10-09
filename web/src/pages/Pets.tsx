import { useState } from "preact/hooks";
import { ConfirmDialog, Empty, ErrorBanner, Field } from "../components/ui";
import { api } from "../lib/api";
import type { Pet } from "../lib/types";
import { useApp } from "../state";

interface Draft {
  id?: string;
  name: string;
  species: "cat" | "dog";
  birth_year: string;
  notes: string;
}

const EMPTY: Draft = { name: "", species: "cat", birth_year: "", notes: "" };

export function PetsPage() {
  const { pets, refreshPets, toast } = useApp();
  const [draft, setDraft] = useState<Draft | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [toDelete, setToDelete] = useState<Pet | null>(null);

  const save = async (e: Event) => {
    e.preventDefault();
    if (!draft) return;
    setError(null);
    const body = { name: draft.name.trim(), species: draft.species, birth_year: draft.birth_year ? Number(draft.birth_year) : null, notes: draft.notes };
    try {
      if (draft.id) await api.patch(`/pets/${draft.id}`, body);
      else await api.post("/pets", body);
      await refreshPets();
      toast(draft.id ? "Pet updated." : "Pet added.");
      setDraft(null);
    } catch (err) {
      setError(err);
    }
  };

  return (
    <section>
      <h1 tabIndex={-1}>Pets</h1>
      <p class="hint">
        Profiles let you label which animal a recording shows. The app does not recognize individual animals; tracks in results are
        per-recording labels only.
      </p>
      {pets.length === 0 && !draft && <Empty title="No pets yet" />}
      <ul class="list">
        {pets.map((p) => (
          <li key={p.id} class="list-item">
            <div>
              <strong>{p.name}</strong> <span class="muted">{p.species}{p.birth_year ? `, born ${p.birth_year}` : ""}</span>
              {p.notes && <div class="muted">{p.notes}</div>}
              <a href={`#/history`} class="muted" onClick={() => sessionStorage.setItem("historyPet", p.id)}>Recordings</a>
            </div>
            <div class="row">
              <button type="button" class="btn btn-secondary" onClick={() => setDraft({ id: p.id, name: p.name, species: p.species, birth_year: p.birth_year ? String(p.birth_year) : "", notes: p.notes })}>Edit</button>
              <button type="button" class="btn btn-link" onClick={() => setToDelete(p)}>Delete</button>
            </div>
          </li>
        ))}
      </ul>
      {!draft && <button type="button" class="btn btn-primary" onClick={() => setDraft({ ...EMPTY })}>Add a pet</button>}
      {draft && (
        <form class="card" onSubmit={save}>
          <h2>{draft.id ? "Edit pet" : "New pet"}</h2>
          <Field label="Name" id="pet-name">
            <input id="pet-name" required maxLength={60} value={draft.name} onInput={(e) => setDraft({ ...draft, name: (e.target as HTMLInputElement).value })} />
          </Field>
          <fieldset class="field">
            <legend>Species</legend>
            <label class="radio"><input type="radio" name="species" checked={draft.species === "cat"} onChange={() => setDraft({ ...draft, species: "cat" })} /> Cat</label>
            <label class="radio"><input type="radio" name="species" checked={draft.species === "dog"} onChange={() => setDraft({ ...draft, species: "dog" })} /> Dog</label>
          </fieldset>
          <Field label="Birth year (optional)" id="pet-year">
            <input id="pet-year" inputMode="numeric" pattern="[0-9]{4}" maxLength={4} value={draft.birth_year} onInput={(e) => setDraft({ ...draft, birth_year: (e.target as HTMLInputElement).value })} />
          </Field>
          <Field label="Notes (optional)" id="pet-notes">
            <textarea id="pet-notes" maxLength={1000} rows={2} value={draft.notes} onInput={(e) => setDraft({ ...draft, notes: (e.target as HTMLTextAreaElement).value })} />
          </Field>
          <ErrorBanner error={error} />
          <div class="row">
            <button type="submit" class="btn btn-primary">Save</button>
            <button type="button" class="btn btn-secondary" onClick={() => setDraft(null)}>Cancel</button>
          </div>
        </form>
      )}
      <ConfirmDialog open={toDelete !== null} title={`Delete ${toDelete?.name ?? ""}?`} danger confirmLabel="Delete"
        body={<p>The profile is removed. Analyses stay but are no longer linked to this pet.</p>}
        onCancel={() => setToDelete(null)}
        onConfirm={async () => {
          const p = toDelete!;
          setToDelete(null);
          try {
            await api.del(`/pets/${p.id}`);
            await refreshPets();
            toast("Pet deleted.");
          } catch (e) {
            setError(e);
          }
        }} />
    </section>
  );
}
