# User flow: ask about a behaviour (web and Android)

## Responsibilities

| Part | Does | Code |
| --- | --- | --- |
| Client (web app; Android shell loads the same app) | capture or import, the owner's species and scene, light checks (type, length, size), progress, answers, corrections, history, drafts and unfinished uploads kept on the device; optional on-device observation engine | `web/src/pages/Ask.tsx`, `components/BehaviorAnswerView.tsx`, `android-app/` |
| Application backend | uploads (resumable), analysis jobs, result and answer storage, model invocation (local pipeline; cloud provider when enabled), consented research samples and their annotation layers, deletion and export | `src/petperception/app/`, `understand/run.py`, `understand/cloud.py` |
| Cloud model provider (optional, off by default) | multimodal understanding and grounded explanation from the backend's evidence packet | `understand/cloud.py` (`AnthropicProvider`; one class per provider behind `name`, `available()`, `complete()`) |

The working path today is client -> backend -> local pipeline (no provider key is configured here); the answer says
when the extended AI service was not available.

Primary journey (one obvious action per screen; species and the owner's description are used as given):

```
Home ──[Ask about my pet]──> 1 Pet & concern ──[Next]──> 2 Recording ──[Send]──> 3 Progress ──> 4 Answer
  │                              │ species (required)        │ guidance + Record / Choose          │ upload %, then        │ summary, what we saw (tap a time to jump),
  │                              │ what you noticed (req.)   │ preview, length and size check      │ "looking at the       │ what it could mean, not sure about,
  │                              │ where / when (optional)   │ options: research sample (opt-in)   │ recording", "answering│ watch for next time, see a vet if
  └─[History]──> list ──> 4      │ draft kept on the device  │                                     │ your question"        │ [Correct this] [Add another recording]
```

| Screen | Layout (top to bottom) | Required | States designed in |
| --- | --- | --- | --- |
| Home | "Ask about my pet's behaviour" (primary button), unfinished work card (resume / discard), recent answers | - | offline banner; resumable uploads |
| 1 Pet & concern | pet chips (or Cat / Dog), "What did you notice?", "Where and when?" with quick chips | species, concern (3+ characters) | draft restored after the app was closed |
| 2 Recording | short guidance (10-60 s, whole animal in view, steady, keep the sound), [Record video] [Choose video], preview with length, options | one clip | camera permission denied (use Choose video / system camera); recording interrupted (keep what was recorded or record again); too short, too long, too large, unreadable file |
| 3 Progress | step list: Uploading (%), Looking at the recording, Answering your question; plain-time estimate | - | upload paused (offline) with automatic resume; server busy (retry later); analysis failed (retry) |
| 4 Answer | one-paragraph answer; "What we saw" with times; "What it could mean" grouped in plain words; "Not sure about"; "Watch for next time"; "See a vet if"; sources (folded); recording check notes (dark, blurry, pet not found) | - | question could not be answered (the observations stay; retry); nothing recognized; poor media advice |
| Correct | per observation: Right / Not right; "What did your pet actually do?" (list + own words); "What happened before or after?" | none | saved; offline (try again) |

Kept out of the journey: model names, scores, thresholds, experiment ids, prompt text. They stay in the stored
result JSON and the developer documents.

Platform patterns: on Android the Record button opens the system camera (it handles permissions, rotation and
interruptions and returns a file), Choose opens the system picker; the app is a thin native shell around the same
web app (`android/`), so the logic is shared. On desktop browsers Record falls back to the file picker or the
in-page recorder of the Record page.

Data the owner creates: corrections and notes are stored as owner reports; on a donated recording they are added to
the research sample as `owner_reported`, never as reviewed labels ([RESEARCH_SAMPLES.md](RESEARCH_SAMPLES.md)).
Screenshots of each screen: `docs/ux/` (made by `web/tests/e2e/journey.spec.ts` against a running server).

Screenshots show a frame of "Tail Chase" by Jason Brackins (Internet Archive, CC BY-NC-SA 2.0), used for this research documentation under that licence.
