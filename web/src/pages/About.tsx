export function AboutPage() {
  return (
    <section class="prose">
      <h1 tabIndex={-1}>How it works - and what it cannot do</h1>
      <h2>What it does</h2>
      <ul>
        <li>Finds cats and dogs in video frames and follows them through the clip (per-recording tracks, not identities).</li>
        <li>Detects cat and dog sounds (meows, barks, growl-like sounds) and some household sounds.</li>
        <li>Measures apparent movement and time spent in places you marked (bowl, door, bed).</li>
        <li>Reports everything with timestamps and quality limits such as low light or camera movement.</li>
      </ul>
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
        <li>Read ears, tail or posture details: no validated model for cat/dog body keypoints is installed.</li>
      </ul>
      <h2>Privacy</h2>
      <p>
        Recordings are deleted from the server right after analysis unless you choose to keep them. Results and feedback stay until you delete
        them or your account. Reference pictures for places stay on your device.
      </p>
      <h2>Measured performance (public datasets, not your home)</h2>
      <ul>
        <li>Sound tagging, cat/dog vs other sounds (ESC-50 clips): macro-F1 about 0.90.</li>
        <li>Cat/dog detection on Open Images photos: average precision about 0.92 (cats) and 0.84 (dogs); other animals such as foxes are sometimes mistaken for dogs or cats.</li>
      </ul>
      <p class="muted">See the project documentation for the full evaluation and its limits.</p>
    </section>
  );
}
