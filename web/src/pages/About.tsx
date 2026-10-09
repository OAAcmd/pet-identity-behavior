export function AboutPage() {
  return (
    <section class="prose">
      <h1 tabIndex={-1}>How it works - and what it cannot do</h1>
      <h2>What it does</h2>
      <ul>
        <li>Finds cats and dogs in video frames and follows them through the clip (per-recording tracks, not identities).</li>
        <li>Detects cat and dog sounds (meows, barks, growl-like sounds) and some household sounds.</li>
        <li>Measures apparent movement and time spent in places you marked (bowl, door, bed).</li>
        <li>Estimates coarse posture (lying, sitting, standing) frame by frame; experimental, often "uncertain".</li>
        <li>Reports everything with timestamps and quality limits such as low light or camera movement.</li>
      </ul>
      <h2>Where the analysis runs</h2>
      <p>
        After the one-time model download (Settings), recordings are analysed <strong>on this device</strong>, also without a network
        connection. Media is uploaded to the server only if you turn on server analysis in Settings and choose "Upload and analyze".
      </p>
      <h2>Interpretations are experimental</h2>
      <p>
        When several independent observations line up (for example a meow while the cat stands at the marked food bowl), the app may show an
        experimental interpretation such as "possibly food-related". These rules have <strong>not</strong> been validated on real household
        recordings; their scores are not probabilities and the app often abstains. Your feedback is what allows them to be evaluated.
      </p>
      <h2>What it will not do</h2>
      <ul>
        <li>Translate what your pet "says" or tell you what it feels.</li>
        <li>Diagnose illness or pain. If you are worried about your pet's health or behavior, contact a veterinarian.</li>
        <li>Recognize which of your pets is in a recording - you tell it.</li>
        <li>Read ears, tail or facial expressions: no model with suitable data terms is installed (no cat data exists for this at all).</li>
      </ul>
      <h2>Privacy</h2>
      <p>
        On-device results are stored only in this browser until you delete them. If you use server analysis, recordings are deleted from
        the server right after analysis unless you choose to keep them; server results and feedback stay until you delete them or your
        account. Reference pictures for places stay on your device.
      </p>
      <h2>Measured performance (public datasets, not your home)</h2>
      <ul>
        <li>Sound tagging, cat/dog vs other sounds (ESC-50 clips): macro-F1 about 0.90.</li>
        <li>Cat/dog detection on Open Images photos: average precision about 0.92 (cats) and 0.84 (dogs); other animals such as foxes are sometimes mistaken for dogs or cats.</li>
        <li>A second check rejects many false detections: on photos verified to contain no dog, a dog was still reported in about 13% of them (44% without the check); for cats 30% (38% without).</li>
        <li>Posture agrees with labels derived from photo captions about 65% of the time; when it is confident, about 85%.</li>
      </ul>
      <p class="muted">See the project documentation for the full evaluation and its limits.</p>
    </section>
  );
}
