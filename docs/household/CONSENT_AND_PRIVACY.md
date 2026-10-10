# Consent and privacy

Household recordings show the inside of people's homes and often people and
their voices. This document lists what consent must cover and the decisions
the data controller (the organisation running the study) must take before
the first recording. It is a specification for the consent text, not the
text itself; the final wording must match the controller's jurisdiction.

## Decisions before collection

| Decision | Options / what to settle |
| --- | --- |
| Data controller | the legal entity responsible; its contact for participants |
| Legal basis | explicit consent (for example GDPR Art. 6(1)(a) for participants in the EU/EEA), with research purposes stated |
| Storage location | where media are stored (region), encryption at rest, who holds keys |
| Access | named roles: study staff, annotators (employed or contracted; contractors need a data processing agreement) |
| Retention | how long media and annotations are kept (for example deleted 24 months after the study ends); aggregate results kept |
| Cross-border transfer | whether media leave the participants' country, and the safeguard used |
| Minors | whether children may appear; guardian consent wording |
| Publication | aggregate metrics only; no media or frames published without separate, specific consent |

## What the participant consents to (each a separate tick box)

1. Recording video with sound of their pets in their home for this study.
2. Transfer of the recordings to the study's storage.
3. Viewing and labelling by named study roles (staff and annotators).
4. Use of the labelled recordings to evaluate (and, if ticked separately,
   to train) the app's models.
5. Publication of aggregate results.

## People and voices in recordings

* Other adults in the home consent in writing or stay out of the frame;
  otherwise their faces are blurred **before** annotation.
* Speech: recordings with identifiable conversation of non-consenting
  people have those segments muted before annotation (muting affects the
  sound evaluation; such recordings are flagged).
* Children appear only with a guardian's consent.

## Participant rights

Withdrawal at any time without giving a reason; on withdrawal all their
media and annotations are deleted within 30 days and excluded from analyses
not yet published. Participants can request a copy of their recordings and
labels. The app itself stays usable without taking part: study
participation is a separate, opt-in research upload; the app's default
on-device analysis uploads nothing.

## What the app already does

On-device analysis keeps media on the phone; the optional result backup
sends only result JSON and feedback; feedback carries an explicit research
consent flag (`consent_research_use`). A research upload flow that sends
media to the study storage only with the consents above is not built yet.
