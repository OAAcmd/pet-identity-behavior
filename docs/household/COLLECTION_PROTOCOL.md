# Collection protocol

## Households

| Group | Minimum | Notes |
| --- | --- | --- |
| Cat households | 30 | at least 8 with two or more cats |
| Dog households | 30 | at least 8 with two or more dogs |
| Cat and dog together | 5 | counted in both groups |
| Phones | both platforms in each group | at least 10 iPhones and 10 Android phones overall; record model, OS and browser |

Recruit for morphology spread and record it as the owner reports it (never
inferred): breed or "mixed", coat length, approximate weight, age, and for
dogs ear carriage (pricked, folded, drop) and tail type (long, curled,
docked). Flat-faced breeds, long-haired cats, black animals and toy and
giant dogs are each needed in at least 3 households, because detection
varies by breed (E16) and dark coats and occlusion are known failure modes.

Each household is assigned to **development** or **test** at enrolment
(50/50, stratified by species and multi-pet status) and stays there.

## Situations and counts per household

At least 20 recordings, 10-60 seconds each, spread over at least 3 days:

| Situation (`owner_context.situation`) | Per household | How |
| --- | --- | --- |
| `before_feeding` | 3 | start recording when you begin preparing food, before serving |
| `waiting_at_door` | 2 | when the pet waits at a door it uses |
| `play` | 2 | ordinary play with a toy or another pet |
| `greeting` | 2 | when you come home or wake up |
| `visitor_or_doorbell` | 1 | only when it happens naturally |
| `rest` | 3 | the pet resting or sleeping |
| `grooming_or_handling` | 1 | brushing or picking up, as you normally do |
| `alone_at_home` | 1 | phone propped up, owner out of the room (optional) |
| `ordinary` | 3 | any moment with no particular situation |
| `pet_free_control` | 2 | the same rooms with no pet present (false-alarm control) |
| `conflict_or_tension` | only if it occurs | never provoke it |

Include at least 2 recordings in low light (evening lamp light), and in
multi-pet homes at least 4 recordings with two pets in view.

## How to record

* Use the app's Record tab (it records video with sound) or the phone's
  camera app; keep the original file. Do not edit, trim or add filters.
* Hold the phone at your normal height or prop it up; avoid zooming and
  quick pans. Keep the pet in view where possible, but do not follow it
  closely; ordinary footage is what the app sees.
* Leave the TV or radio as it normally is, and note it.
* Before you look at any analysis result, fill in the owner context: the
  situation, what happened just before, what the pet did next (for example
  "was fed", "went out", "kept playing"), and anything else.

## Safety and welfare rules

* Never provoke fear, conflict, pain or frustration to get a recording.
* Stop recording if the pet shows distress, and attend to the pet.
* No recordings of veterinary procedures or of medication by force.
* People who do not want to be filmed leave the frame or are blurred
  before annotation (see the consent document); children appear only with
  a guardian's consent.

## Intake checks (project side)

For every received file: duration 5-120 s; a video track; an audio track
(or a note why not); container and codec recorded (iPhone HEVC `.mov`,
Android H.264/HEVC `.mp4`); SHA-256 recorded; owner context present and
entered before results were seen (the app records the time); consent flags
complete. The app's own quality flags (low light, blur, camera motion) are
stored with the recording but do not exclude it: difficult footage is part
of what is measured.
