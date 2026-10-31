# Research samples and owner feedback (consent version `rs-1`)

What the owner agrees to when ticking "Keep this recording as a research sample" (asked per recording, separate from
server analysis, cloud understanding and the older feedback consent):

* the recording, the species, scene and question are copied to the server's research store
  (`storage.research_root`, default `var/app/research_samples/`, mode 0700; never in Git);
* they are used only to label, train and evaluate behaviour models of this project;
* the owner can delete a sample at any time (`DELETE /api/v1/research-samples/{id}`), and deleting the account
  deletes every sample with its files;
* answers produced by a model never count as labels.

## Data layers (kept apart in `sample_annotations`)

| Source | Status | Who writes it | Use |
| --- | --- | --- | --- |
| `model` | `provisional` | the worker, from the answer (cloud model or local recognizer) | candidate for review; never training or test labels by itself |
| `owner` | `owner_reported` | the owner: observed behaviour, context, corrections, what happened afterwards (`POST .../feedback`) | context and weak labels; reported as owner reports |
| `reviewer` | `human_confirmed` | an administrator after reviewing the clip (`POST .../review`: behaviours with times, certainty) | training and evaluation labels |

Independent evaluation data must come from samples whose labels are `human_confirmed` and whose owner (household)
is not in the training split; splits group all samples of one account together. The database enforces the
source/status pairs (migration `0004_questions_research_samples.sql`).

## API

`GET /research-samples`, `GET /research-samples/{id}`, `DELETE /research-samples/{id}`,
`POST /research-samples/{id}/feedback` (owner), `POST /research-samples/{id}/review` (administrator). Samples are
included in the account export (`GET /export`); the recordings themselves are not.
