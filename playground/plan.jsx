import { useState, useRef, useEffect, useCallback } from "react";

// ─── colour palette ───────────────────────────────────────────────────────────
const C = {
  bg:      "#07090f",
  surface: "#0e1118",
  surf2:   "#141720",
  border:  "#1c2030",
  cyan:    "#00e5ff",
  purple:  "#9b5de5",
  green:   "#10b981",
  amber:   "#f59e0b",
  red:     "#ef4444",
  dim:     "#64748b",
  muted:   "#94a3b8",
  text:    "#e2e8f0",
};

// ─── helper: draw a spectrogram-like canvas ────────────────────────────────
function useCanvas(drawFn, deps) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    drawFn(ctx, c.width, c.height);
  }, deps);
  return ref;
}

// ─── noise helper ────────────────────────────────────────────────────────────
function seededRand(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
}

// ════════════════════════════════════════════════════════════════════════════
// DIAGRAM COMPONENTS
// ════════════════════════════════════════════════════════════════════════════

// P01–P06 → Mel Spectrogram comparison
function MelSpectrogramDiagram() {
  function draw(ctx, W, H) {
    const half = W / 2 - 4;
    const rng  = seededRand(42);

    // helper: draw one spectrogram panel
    function drawPanel(ox, label, isReal) {
      const cols = 80, rows = 40;
      const cw = half / cols, ch = H / (rows + 16);
      // title
      ctx.fillStyle = isReal ? C.green : C.red;
      ctx.font = "bold 11px monospace";
      ctx.fillText(label, ox + 6, 14);
      // grid
      for (let t = 0; t < cols; t++) {
        for (let f = 0; f < rows; f++) {
          // real: natural broadband energy + formant bands
          // fake: smooth, over-regularised
          let energy;
          if (isReal) {
            // formant peaks at ~5, 12, 20 + natural noise
            const harmonic = Math.exp(-((f - 5) ** 2) / 8) * 0.9
              + Math.exp(-((f - 12) ** 2) / 10) * 0.7
              + Math.exp(-((f - 22) ** 2) / 14) * 0.5;
            const temporal = Math.sin(t * 0.4 + f * 0.1) * 0.2;
            energy = Math.max(0, Math.min(1, harmonic + temporal + (rng() - 0.5) * 0.35));
          } else {
            // TTS: too smooth, unnaturally uniform, missing high-freq noise
            const harmonic = Math.exp(-((f - 5) ** 2) / 10) * 0.85
              + Math.exp(-((f - 12) ** 2) / 12) * 0.65
              + Math.exp(-((f - 22) ** 2) / 16) * 0.4;
            const temporal = Math.sin(t * 0.35 + f * 0.08) * 0.08; // less variation
            energy = Math.max(0, Math.min(1, harmonic + temporal + (rng() - 0.5) * 0.07));
          }
          // colour map: dark→blue→cyan→yellow→red
          const r = energy < 0.5 ? Math.floor(energy * 2 * 60) : Math.floor(60 + (energy - 0.5) * 2 * 195);
          const g = energy < 0.33 ? 0 : energy < 0.66 ? Math.floor((energy - 0.33) * 3 * 200) : Math.floor(200 + (energy - 0.66) * 3 * 55);
          const b = energy < 0.5 ? Math.floor(energy * 2 * 200) : Math.floor(200 - (energy - 0.5) * 2 * 200);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(ox + t * cw, 20 + (rows - 1 - f) * ch, Math.ceil(cw), Math.ceil(ch));
        }
      }
      // axis labels
      ctx.fillStyle = C.dim;
      ctx.font = "9px monospace";
      ctx.fillText("High", ox + 1, 24);
      ctx.fillText("Low",  ox + 1, 18 + rows * ch);
      ctx.fillText("Freq", ox + 1, 28 + rows * ch);
      ctx.fillText("← Time →", ox + half / 2 - 20, H - 2);

      // annotations for fake
      if (!isReal) {
        ctx.strokeStyle = "#ef444488";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 2]);
        // draw "too smooth" highlight band
        ctx.strokeRect(ox + 10, 20 + (rows - 14) * ch, half - 18, ch * 4);
        ctx.setLineDash([]);
        ctx.fillStyle = "#ef4444cc";
        ctx.font = "8px monospace";
        ctx.fillText("↑ over-smooth", ox + 14, 20 + (rows - 15) * ch);
      } else {
        ctx.strokeStyle = "#10b98188";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 2]);
        ctx.strokeRect(ox + 10, 20 + (rows - 14) * ch, half - 18, ch * 4);
        ctx.setLineDash([]);
        ctx.fillStyle = "#10b981cc";
        ctx.font = "8px monospace";
        ctx.fillText("↑ natural grain", ox + 14, 20 + (rows - 15) * ch);
      }
    }

    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    drawPanel(0,       "REAL SPEECH",  true);
    drawPanel(half + 8, "TTS / FAKE",  false);
    // divider
    ctx.fillStyle = C.border;
    ctx.fillRect(half + 1, 0, 6, H);
  }
  const ref = useCanvas(draw, []);
  return <canvas ref={ref} width={560} height={160} style={{ width:"100%", borderRadius:8, display:"block" }} />;
}

// P13–P16 → Phase / IF diagram
function PhaseDiagram() {
  function draw(ctx, W, H) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const half = W / 2 - 4;

    function drawIF(ox, label, isReal) {
      ctx.fillStyle = isReal ? C.green : C.red;
      ctx.font = "bold 11px monospace";
      ctx.fillText(label + " — Instantaneous Frequency", ox + 4, 14);

      const pts = 100;
      const dx  = half / pts;
      ctx.beginPath();
      ctx.strokeStyle = isReal ? C.cyan : C.amber;
      ctx.lineWidth = 1.5;
      for (let i = 0; i < pts; i++) {
        const x = ox + i * dx;
        let y;
        if (isReal) {
          // natural: smooth with micro-jitter and occasional discontinuity
          y = H / 2 + Math.sin(i * 0.18) * 28
            + Math.sin(i * 0.7) * 8
            + (Math.random() - 0.5) * 10;
          // occasional phase slip
          if (i === 30 || i === 65) y += (Math.random() > 0.5 ? 1 : -1) * 18;
        } else {
          // fake: too smooth, near-perfect sinusoid, NO discontinuities
          y = H / 2 + Math.sin(i * 0.18) * 28 + Math.sin(i * 0.7) * 2;
        }
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();

      if (!isReal) {
        ctx.fillStyle = "#f59e0b99";
        ctx.font = "8px monospace";
        ctx.fillText("← no phase slips (unnatural)", ox + 4, H - 6);
      } else {
        ctx.fillStyle = "#10b98199";
        ctx.font = "8px monospace";
        ctx.fillText("← natural micro-discontinuities", ox + 4, H - 6);
      }
    }

    drawIF(0,       "REAL", true);
    ctx.fillStyle = C.border;
    ctx.fillRect(half + 1, 0, 6, H);
    drawIF(half + 8, "FAKE", false);
  }
  const ref = useCanvas(draw, []);
  return <canvas ref={ref} width={560} height={110} style={{ width:"100%", borderRadius:8, display:"block" }} />;
}

// P17–P22 → F0 / Prosody diagram
function F0Diagram() {
  function draw(ctx, W, H) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const rng = seededRand(7);

    // Real F0
    ctx.fillStyle = C.green;
    ctx.font = "bold 11px monospace";
    ctx.fillText("REAL — F0 Pitch Contour", 6, 14);

    const pts = 120, dx = W / pts;
    function f0Real(i) {
      return H * 0.65 - Math.sin(i * 0.06) * 30 - Math.sin(i * 0.22) * 12
        + (rng() - 0.5) * 16      // jitter
        - (i > 55 && i < 75 ? 20 : 0); // natural dip mid-utterance
    }
    function f0Fake(i) {
      return H * 0.65 - Math.sin(i * 0.06) * 30 - Math.sin(i * 0.22) * 4; // too smooth
    }

    // Real line
    ctx.beginPath(); ctx.strokeStyle = C.green; ctx.lineWidth = 2;
    for (let i = 0; i < pts; i++) {
      const x = i * dx, y = f0Real(i);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fake line overlay
    ctx.beginPath(); ctx.strokeStyle = C.red; ctx.lineWidth = 2; ctx.setLineDash([5, 3]);
    for (let i = 0; i < pts; i++) {
      const x = i * dx, y = f0Fake(i);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke(); ctx.setLineDash([]);

    // legend
    ctx.fillStyle = C.green; ctx.font = "9px monospace";
    ctx.fillText("── Real (natural jitter)", 6, H - 20);
    ctx.fillStyle = C.red;
    ctx.fillText("╌╌ Fake (over-smooth TTS)", 6, H - 8);

    // jitter annotation
    ctx.strokeStyle = "#00e5ff55"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(30 * dx, f0Real(30), 12, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = C.cyan; ctx.font = "8px monospace";
    ctx.fillText("jitter", 30 * dx - 10, f0Real(30) - 16);
  }
  const ref = useCanvas(draw, []);
  return <canvas ref={ref} width={560} height={120} style={{ width:"100%", borderRadius:8, display:"block" }} />;
}

// P23–P26 → Glottal / harmonic spectrum
function GlottalDiagram() {
  function draw(ctx, W, H) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const half = W / 2 - 4;
    const rng = seededRand(13);

    function drawSpectrum(ox, label, isReal) {
      ctx.fillStyle = isReal ? C.green : C.red;
      ctx.font = "bold 11px monospace";
      ctx.fillText(label + " — Harmonic Spectrum", ox + 4, 14);

      const harmonics = 18;
      const bw = (half - 12) / harmonics;

      for (let h = 1; h <= harmonics; h++) {
        // H1 should be strongest for real; TTS makes H2 sometimes equal H1
        let amp;
        if (isReal) {
          amp = (1 / h) * 0.85 + (rng() - 0.5) * 0.08; // natural roll-off with jitter
        } else {
          amp = (1 / h) * 0.9 + 0.02; // too perfect roll-off, no variance
          if (h === 2) amp *= 1.15;    // H2 bump — unnatural H1-H2 ratio
        }
        amp = Math.max(0.03, Math.min(1, amp));

        const x  = ox + 6 + (h - 1) * bw;
        const bh = (H - 30) * amp;
        const y  = H - 16 - bh;

        // colour by harmonic
        const hue = isReal ? `hsl(${180 + h * 8}, 80%, 55%)` : `hsl(${30 + h * 5}, 90%, 55%)`;
        ctx.fillStyle = hue;
        ctx.fillRect(x, y, bw - 2, bh);

        if (h === 1 || h === 2) {
          ctx.fillStyle = "#fff";
          ctx.font = "8px monospace";
          ctx.fillText(`H${h}`, x + 1, H - 4);
        }
      }

      if (!isReal) {
        // draw arrow showing unnatural H2 bump
        ctx.strokeStyle = C.amber;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        const x2 = ox + 6 + bw;
        ctx.moveTo(x2 + 4, 30); ctx.lineTo(x2 + 4, 18);
        ctx.stroke();
        ctx.fillStyle = C.amber;
        ctx.font = "8px monospace";
        ctx.fillText("H2↑ anomaly", x2 - 4, 14);
      }
    }

    drawSpectrum(0,       "REAL", true);
    ctx.fillStyle = C.border;
    ctx.fillRect(half + 1, 0, 6, H);
    drawSpectrum(half + 8, "FAKE", false);
  }
  const ref = useCanvas(draw, []);
  return <canvas ref={ref} width={560} height={130} style={{ width:"100%", borderRadius:8, display:"block" }} />;
}

// P27–P30 → Formant tracks
function FormantDiagram() {
  function draw(ctx, W, H) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const half = W / 2 - 4;
    const rng = seededRand(99);

    function drawFormants(ox, label, isReal) {
      ctx.fillStyle = isReal ? C.green : C.red;
      ctx.font = "bold 11px monospace";
      ctx.fillText(label + " — Formant Tracks (F1/F2/F3)", ox + 4, 14);

      const pts = 80, dx = (half - 8) / pts;
      const formantColors = [C.cyan, C.amber, C.purple];
      const formantBase   = [0.75, 0.5, 0.28]; // normalised vertical positions

      formantColors.forEach((col, fi) => {
        ctx.beginPath();
        ctx.strokeStyle = col;
        ctx.lineWidth = isReal ? 1.8 : 1.8;
        if (!isReal) ctx.setLineDash([]);

        for (let i = 0; i < pts; i++) {
          const x = ox + 4 + i * dx;
          let y;
          if (isReal) {
            // natural: formants shift rapidly at phoneme boundaries
            const phonemeBoundary = Math.floor(i / 20);
            const baseY = formantBase[fi] * (H - 30) + 20;
            y = baseY + Math.sin(i * 0.15 + fi) * 12
              + (phonemeBoundary % 2 === 0 ? Math.sin(i * 0.5) * 8 : 0)
              + (rng() - 0.5) * 7;
          } else {
            // fake: overly smooth, transitions too gradual
            const baseY = formantBase[fi] * (H - 30) + 20;
            y = baseY + Math.sin(i * 0.15 + fi) * 12
              + Math.sin(i * 0.5) * 1.5; // almost no variation
          }
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();

        // label
        ctx.fillStyle = col;
        ctx.font = "9px monospace";
        ctx.fillText(`F${fi + 1}`, ox + half - 18, formantBase[fi] * (H - 30) + 20);
      });

      if (!isReal) {
        ctx.fillStyle = "#f59e0b88";
        ctx.font = "8px monospace";
        ctx.fillText("← transitions too smooth", ox + 4, H - 4);
      } else {
        ctx.fillStyle = "#10b98188";
        ctx.font = "8px monospace";
        ctx.fillText("← rapid natural transitions", ox + 4, H - 4);
      }
    }

    drawFormants(0,       "REAL", true);
    ctx.fillStyle = C.border;
    ctx.fillRect(half + 1, 0, 6, H);
    drawFormants(half + 8, "FAKE", false);
  }
  const ref = useCanvas(draw, []);
  return <canvas ref={ref} width={560} height={140} style={{ width:"100%", borderRadius:8, display:"block" }} />;
}

// P31–P35 → Temporal energy / modulation
function TemporalDiagram() {
  function draw(ctx, W, H) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const rng = seededRand(55);
    const pts = 140, dx = W / pts;

    ctx.fillStyle = C.muted;
    ctx.font = "bold 11px monospace";
    ctx.fillText("RMS Energy Envelope — Real vs Fake", 6, 14);

    function energyReal(i) {
      return H * 0.5 + Math.sin(i * 0.22) * 22
        + Math.sin(i * 0.08) * 14
        + (rng() - 0.5) * 18
        + (i > 50 && i < 60 ? 25 : 0);  // natural breath burst
    }
    function energyFake(i) {
      return H * 0.5 + Math.sin(i * 0.22) * 20 + Math.sin(i * 0.08) * 12;
    }

    // fill area under real
    ctx.beginPath();
    ctx.fillStyle = "rgba(16,185,129,0.12)";
    ctx.moveTo(0, H);
    for (let i = 0; i < pts; i++) ctx.lineTo(i * dx, energyReal(i));
    ctx.lineTo(W, H); ctx.closePath(); ctx.fill();

    // real line
    ctx.beginPath(); ctx.strokeStyle = C.green; ctx.lineWidth = 2;
    for (let i = 0; i < pts; i++) {
      const x = i * dx, y = energyReal(i);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // fake line
    ctx.beginPath(); ctx.strokeStyle = C.red; ctx.lineWidth = 2; ctx.setLineDash([5, 3]);
    for (let i = 0; i < pts; i++) {
      const x = i * dx, y = energyFake(i);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke(); ctx.setLineDash([]);

    // breath burst annotation
    ctx.strokeStyle = C.cyan + "88"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(55 * dx, energyReal(55), 16, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = C.cyan; ctx.font = "8px monospace";
    ctx.fillText("breath burst", 55 * dx - 28, energyReal(55) - 20);

    ctx.fillStyle = C.green; ctx.font = "9px monospace";
    ctx.fillText("── Real", 6, H - 18);
    ctx.fillStyle = C.red;
    ctx.fillText("╌╌ Fake (too uniform)", 6, H - 6);
  }
  const ref = useCanvas(draw, []);
  return <canvas ref={ref} width={560} height={120} style={{ width:"100%", borderRadius:8, display:"block" }} />;
}

// P36–P40 → Artifact signatures
function ArtifactDiagram() {
  function draw(ctx, W, H) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    // Left: vocoder cutoff in spectrogram slice
    const half = W / 2 - 4;
    ctx.fillStyle = C.red;
    ctx.font = "bold 11px monospace";
    ctx.fillText("FAKE — Vocoder Frequency Cutoff", 4, 14);

    // fake spectrogram column
    const rows = 48;
    const ch = (H - 20) / rows;
    for (let f = 0; f < rows; f++) {
      const freqNorm = f / rows;
      let energy;
      // vocoder cutoff: sharp drop above ~70% of bandwidth
      if (freqNorm > 0.72) {
        energy = 0.02 + Math.random() * 0.03; // near-zero above cutoff
      } else {
        energy = 0.6 - freqNorm * 0.4 + Math.random() * 0.15;
      }
      const r = Math.floor(energy * 60);
      const g = Math.floor(energy * 180);
      const b = Math.floor(energy * 220);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      for (let t = 0; t < half - 8; t += 6) {
        ctx.fillRect(4 + t, 20 + (rows - 1 - f) * ch, 5, Math.ceil(ch));
      }
    }
    // cutoff line
    const cutoffY = 20 + (rows - Math.floor(rows * 0.72)) * ch;
    ctx.strokeStyle = C.amber;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(4, cutoffY); ctx.lineTo(half - 4, cutoffY);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.amber; ctx.font = "8px monospace";
    ctx.fillText("← sharp cutoff (vocoder)", 6, cutoffY - 4);
    ctx.fillText("Low", 4, H - 4);
    ctx.fillText("High →", 4, 24);

    // Right: periodic grid pattern
    ctx.fillStyle = C.border;
    ctx.fillRect(half + 1, 0, 6, H);
    ctx.fillStyle = C.amber;
    ctx.font = "bold 11px monospace";
    ctx.fillText("FAKE — Vocoder Grid Pattern", half + 8, 14);

    // draw periodic artifacts
    for (let f = 0; f < rows; f++) {
      for (let t = 0; t < 80; t++) {
        const freqNorm = f / rows;
        const timeNorm = t / 80;
        // base energy
        let energy = 0.4 - freqNorm * 0.3 + Math.random() * 0.1;
        // periodic grid at harmonic intervals
        const pitchPeriod = 8;
        if (t % pitchPeriod < 1.5) energy += 0.35; // pitch pulse grid
        energy = Math.min(1, energy);

        const r = Math.floor(energy * 200);
        const g = Math.floor(energy * 120);
        const b = Math.floor(energy * 40);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        const cw = (half - 8) / 80;
        ctx.fillRect(half + 8 + t * cw, 20 + (rows - 1 - f) * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    // annotate grid lines
    ctx.strokeStyle = C.red + "aa"; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
    for (let t = 0; t < 80; t += 8) {
      const x = half + 8 + t * ((half - 8) / 80);
      ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, H - 14); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = C.red; ctx.font = "8px monospace";
    ctx.fillText("↑ periodic pitch grid (GAN artifact)", half + 10, H - 4);
  }
  const ref = useCanvas(draw, []);
  return <canvas ref={ref} width={560} height={160} style={{ width:"100%", borderRadius:8, display:"block" }} />;
}

// ════════════════════════════════════════════════════════════════════════════
// DATA
// ════════════════════════════════════════════════════════════════════════════
const GROUPS = [
  {
    id: "mel",
    label: "A",
    title: "Mel Spectrogram Core",
    color: C.cyan,
    badge: "P01–P06",
    branch: "CNN Branch",
    intro: `A Mel spectrogram is a picture of sound — time runs left to right, frequency bottom to top, and brightness shows energy. The human ear doesn't hear all frequencies equally, so Mel scale squishes high frequencies together (where we're less sensitive) and spreads low ones apart. This makes it the perfect input for a CNN treating audio like an image.`,
    diagram: <MelSpectrogramDiagram />,
    diagramCaption: "Real speech (left) has natural grain and texture at high frequencies. TTS/fake audio (right) is over-smoothed — unnaturally clean, like a painting vs a photograph.",
    params: [
      { id:"P01", name:"Log-Mel Spectrogram (80 bands)", why:"The foundation. 80 frequency bands on the Mel scale, log-compressed. Real speech has textured, noisy energy between formant peaks. TTS is suspiciously clean — the CNN learns to spot this texture difference.", impact:"★★★★★" },
      { id:"P02", name:"Delta Mel (1st derivative)", why:"How fast is the spectrum changing frame to frame? Natural speech changes abruptly at consonant-to-vowel transitions. TTS smooths these transitions — the Delta shows this as gentler slopes where real speech has sharp cliffs.", impact:"★★★★☆" },
      { id:"P03", name:"Delta-Delta Mel (2nd derivative)", why:"The acceleration of change. If Delta is velocity, Delta-Delta is acceleration. Neural vocoders regularize this too much — they can't replicate the chaotic micro-changes in a real human's vocal tract shifting between phonemes.", impact:"★★★★☆" },
      { id:"P04", name:"Mel Sub-band Variance (per band)", why:"Across time, how much does each Mel band fluctuate? Real speech variance is high and uneven across bands. TTS variance is unnaturally low and uniform — the voice sounds 'too stable'. This is the over-smoothing fingerprint.", impact:"★★★★★" },
      { id:"P05", name:"Mel Spectral Flux", why:"The total energy change between consecutive frames. Real speech has bursty, irregular flux (especially at plosives like /p/, /t/, /k/). TTS flux is regular and controlled. Think of it as measuring how 'surprised' the spectrum is frame to frame.", impact:"★★★☆☆" },
      { id:"P06", name:"Mel Sub-band Kurtosis", why:"Kurtosis measures how 'peaky' a distribution is. Real speech has spiky, non-Gaussian energy distributions — occasional very loud frames. TTS produces a bell-curve distribution that's too Gaussian. Low kurtosis = suspicious.", impact:"★★★☆☆" },
    ]
  },
  {
    id: "cepstral",
    label: "B",
    title: "Cepstral Domain",
    color: C.purple,
    badge: "P07–P12",
    branch: "CNN Branch",
    intro: `Cepstrum (spectrum of the log-spectrum) separates the vocal tract shape (formants) from the glottal source (the voice's raw buzz). MFCCs are the most famous cepstral features — they've been in speech recognition for 40 years and remain powerful deepfake detectors because TTS systems struggle to replicate the full cepstral envelope of real human speech.`,
    diagram: null,
    params: [
      { id:"P07", name:"MFCCs coefficients 1–13", why:"Lower MFCCs capture the broad shape of the vocal tract filter — basically which vowel is being spoken. TTS gets these roughly right (or speech recognition would fail), but there are subtle differences in the lower cepstral envelope that the CNN catches.", impact:"★★★★☆" },
      { id:"P08", name:"MFCCs coefficients 14–40", why:"Higher-order MFCCs capture fine spectral detail — the texture of the vocal tract. These are much harder for TTS to replicate accurately. Most TTS evaluation ignores them because they don't affect intelligibility, but they betray synthetic origin.", impact:"★★★★★" },
      { id:"P09", name:"Delta MFCCs", why:"How fast the vocal tract shape is changing. Real speech has rapid cepstral transitions during stop consonants (/b/, /d/, /g/). TTS transitions are too smooth — the Delta shows this as a smeared, gradual change instead of a sharp step.", impact:"★★★★☆" },
      { id:"P10", name:"LFCCs — Linear Frequency Cepstral Coefficients", why:"Same concept as MFCCs but on a LINEAR frequency scale instead of Mel. The linear scale gives higher resolution at high frequencies — exactly where many neural vocoder artifacts live. LFCCs and MFCCs are complementary: they catch different artifacts. LFCCs are consistently the top performer on ASVspoof benchmarks.", impact:"★★★★★" },
      { id:"P11", name:"Cepstral Peak Prominence (CPP)", why:"Measures how clearly defined the cepstral peak (corresponding to the fundamental frequency) is. Real voices have a prominent, slightly irregular cepstral peak. TTS has an unnaturally strong, perfectly consistent peak — the vocal fold model is too idealized.", impact:"★★★★☆" },
      { id:"P12", name:"RASTA-PLP Coefficients", why:"Perceptual Linear Prediction with RASTA filtering — designed to be robust to channel distortion. Perfect for phone calls where codec changes the channel. PLP captures vocal tract resonances in a way that's more stable across different recording conditions than MFCCs.", impact:"★★★☆☆" },
    ]
  },
  {
    id: "phase",
    label: "C",
    title: "Phase & Instantaneous Frequency",
    color: C.amber,
    badge: "P13–P16",
    branch: "BiMamba Branch",
    intro: `Every standard spectrogram you've seen throws away phase information. That's a huge mistake for deepfake detection. GAN-based vocoders (HiFi-GAN, WaveGlow) generate magnitude spectrograms convincingly but struggle with phase coherence. Real speech has messy, continuous phase — deepfakes have phase patterns that are too clean or show unnatural discontinuities.`,
    diagram: <PhaseDiagram />,
    diagramCaption: "Instantaneous frequency (phase derivative over time) in real speech shows natural micro-slips and irregularities. TTS/GAN audio is too smooth — the phase was reconstructed by an algorithm, not a human vocal tract.",
    params: [
      { id:"P13", name:"Instantaneous Frequency (IF)", why:"The derivative of phase over time — tells you how the frequency of each spectral component is evolving moment to moment. Real speech IF is noisy and irregular. GAN vocoders reconstruct phase algorithmically, producing IF that's too smooth. One of the most sensitive indicators of neural vocoder artifacts.", impact:"★★★★★" },
      { id:"P14", name:"Group Delay Spectrum", why:"The derivative of phase over frequency — a different angle on the same phase information. Group delay reveals formant structure very clearly (peaks at formant frequencies) and shows artifacts where the synthesis algorithm introduces non-physical phase relationships between nearby frequency components.", impact:"★★★★☆" },
      { id:"P15", name:"Modified Group Delay (MODGD)", why:"Raw group delay is very noisy. MODGD smooths it while keeping the important structure. It reveals vocal tract resonances more clearly than the magnitude spectrum in some cases, and the shape of MODGD features differs systematically between real and synthetic speech.", impact:"★★★★☆" },
      { id:"P16", name:"Phase Discontinuity at Frame Boundaries", why:"When a GAN generates audio frame by frame, there's a seam where frames join. Real speech has no such seams — the vocal tract moves continuously. These micro-discontinuities in phase are invisible to the ear but visible in the phase spectrogram. A uniquely synthetic artifact.", impact:"★★★☆☆" },
    ]
  },
  {
    id: "pitch",
    label: "D",
    title: "Pitch & Prosody",
    color: C.green,
    badge: "P17–P22",
    branch: "BiMamba Branch",
    intro: `Pitch is the musical note your voice produces — it rises when you ask a question, falls at the end of a statement. TTS systems model pitch statistically from training data, so they get the broad contour right (enough to sound natural) but miss the chaotic, biological micro-variations that human vocal folds produce in every single breath cycle.`,
    diagram: <F0Diagram />,
    diagramCaption: "Real F0 (green) shows natural jitter and an unpredictable mid-utterance dip. Fake F0 (red dashed) follows an identical smooth sinusoidal path — statistically plausible but biologically impossible.",
    params: [
      { id:"P17", name:"F0 Contour (Fundamental Frequency)", why:"The pitch track across the utterance. TTS gets the broad shape right (rising for questions, falling for statements) but is too regular. A real voice constantly deviates from any smooth prediction — due to breath pressure changes, muscle micro-tremor, and emotional state.", impact:"★★★★☆" },
      { id:"P18", name:"F0 Delta (pitch velocity)", why:"How fast pitch is changing. Real speech has sudden pitch jumps at stressed syllables. TTS pitch changes too gradually — like a smooth curve vs. a staircase. The BiMamba branch captures this over long contexts since pitch patterns span full sentences.", impact:"★★★★☆" },
      { id:"P19", name:"Jitter (local)", why:"Cycle-to-cycle variation in the pitch period. A real vocal fold never vibrates at exactly the same period twice. Jitter of ~0.5–1% is normal for healthy speech. TTS has near-zero jitter — the synthesis model produces a perfectly periodic source. This is one of the most reliable detectors.", impact:"★★★★★" },
      { id:"P20", name:"Shimmer (local dB)", why:"Cycle-to-cycle variation in amplitude. Like jitter but for loudness instead of timing. Real vocal folds vary in how hard they close each cycle. TTS shimmer is near-zero — unnaturally consistent amplitude. Combined with jitter, these two features alone achieve competitive EER on clean speech.", impact:"★★★★★" },
      { id:"P21", name:"Voiced/Unvoiced Decision Mask", why:"Binary track of which frames are voiced (vowels, voiced consonants) vs. unvoiced (fricatives, silence). TTS voicing transitions are too crisp — real speech has gradual devoicing at the end of vowels and breathy onsets. The transition pattern at boundaries is a fingerprint.", impact:"★★★☆☆" },
      { id:"P22", name:"Harmonic-to-Noise Ratio (HNR)", why:"The ratio of periodic (harmonic) energy to random (noise) energy in the voice. Real speech has moderate, variable HNR — especially lower during breathy phonation. TTS produces unrealistically high, stable HNR: the synthetic source is too pure, too clean. Real voices are never that tonal.", impact:"★★★★☆" },
    ]
  },
  {
    id: "glottal",
    label: "E",
    title: "Glottal Source",
    color: "#f97316",
    badge: "P23–P26",
    branch: "BiMamba Branch",
    intro: `The glottis is the gap between your vocal folds. Every time the folds slam shut, they create a pulse of air — the glottal pulse. The shape of that pulse determines the harmonic richness of your voice. GAN vocoders model the glottal source implicitly, and the acoustic fingerprint of that imperfect model shows up in the harmonic structure.`,
    diagram: <GlottalDiagram />,
    diagramCaption: "Real speech (left) shows natural harmonic roll-off with minor variations. Fake speech (right) has an unnatural H2 amplitude bump — a signature artifact of the GAN's simplified glottal source model.",
    params: [
      { id:"P23", name:"H1–H2 Amplitude Difference", why:"The difference in amplitude between the 1st and 2nd harmonic. In breathy voice, H1 > H2 by a large margin. In pressed voice, they're closer. TTS systems get this ratio wrong — often making H2 too prominent, which betrays a simplified vocoidal source model. Sensitive to the specific vocoder architecture.", impact:"★★★★☆" },
      { id:"P24", name:"H1–A1, H1–A2, H1–A3 ratios", why:"Comparing the first harmonic (H1) to the amplitude at each formant frequency (A1, A2, A3). These ratios encode how the glottal source interacts with the vocal tract. Real speech has complex, speaker-specific H1-Ax patterns. TTS reproduces these approximately but not exactly — the mismatch is consistent.", impact:"★★★★☆" },
      { id:"P25", name:"LPC Residual Spectrum", why:"After removing the vocal tract filter (LPC), what's left is the glottal excitation signal. For real speech, this is a complex, asymmetric pulse with fine structure. For TTS, the residual looks too smooth — it came from an idealized model. The residual spectrum directly exposes the synthetic source.", impact:"★★★★★" },
      { id:"P26", name:"Normalized Amplitude Quotient (NAQ)", why:"A measure of breathiness derived from the glottal flow waveform. Real voices show a wide range of NAQ across phonetic contexts — it varies with emphasis, emotion, and breath pressure. TTS NAQ is unnaturally constant. Particularly effective at catching voice conversion systems that clone a speaker's identity.", impact:"★★★☆☆" },
    ]
  },
  {
    id: "formant",
    label: "F",
    title: "Formant Structure",
    color: "#ec4899",
    badge: "P27–P30",
    branch: "CNN Branch",
    intro: `Formants are the resonant frequencies of your vocal tract — the peaks in the frequency spectrum that your throat, mouth, and nasal cavity create. Every vowel has a characteristic F1/F2 pattern. TTS systems track formants statistically but miss the rapid, physics-driven transitions that happen when you move from consonant to vowel. These transitions happen in milliseconds and are hard to fake.`,
    diagram: <FormantDiagram />,
    diagramCaption: "Real speech (left) shows rapid formant transitions at phoneme boundaries — F1, F2, F3 jump quickly. TTS (right) produces much smoother tracks — the synthesis model interpolates instead of replicating fast articulatory movements.",
    params: [
      { id:"P27", name:"F1 Frequency Contour", why:"First formant — primarily encodes vowel height (high vowels like /i/ have low F1, low vowels like /a/ have high F1). The transition into and out of consonants happens in <20ms in real speech. TTS formant transitions are always longer than this because the model interpolates through training statistics.", impact:"★★★★☆" },
      { id:"P28", name:"F2 Frequency Contour", why:"Second formant — encodes vowel frontness and is the most speaker-discriminative formant. Also hardest to clone in voice conversion. The F2 trajectory during consonant-vowel transitions contains rapid co-articulation effects that reflect the actual physics of tongue movement — impossible to fully model statistically.", impact:"★★★★★" },
      { id:"P29", name:"Formant Bandwidth (F1–F3)", why:"How wide each formant resonance is. Bandwidth is controlled by the acoustic absorption properties of the vocal tract walls — a complex biophysical parameter. TTS formant bandwidths are often too narrow (formants too sharp), making the voice sound slightly artificial even when the center frequencies are correct.", impact:"★★★☆☆" },
      { id:"P30", name:"LPC Coefficients (order 16)", why:"Linear Predictive Coding models the vocal tract as a filter. The 16 LPC coefficients compactly describe the formant envelope without needing explicit formant tracking. They're robust to codec compression (unlike raw spectrograms) and capture subtle vocal tract shape differences that TTS doesn't perfectly replicate.", impact:"★★★★☆" },
    ]
  },
  {
    id: "temporal",
    label: "G",
    title: "Temporal & Energy",
    color: C.green,
    badge: "P31–P35",
    branch: "BiMamba Branch",
    intro: `Real speech has chaotic energy dynamics — breath pressure fluctuates, muscles fatigue, emotions change amplitude. TTS energy is driven by a statistical model of "average" prosody, so it lacks the biological unpredictability of a real voice. BiMamba excels here because these patterns unfold over seconds, requiring long-range sequential context that CNNs can't efficiently capture.`,
    diagram: <TemporalDiagram />,
    diagramCaption: "Real energy (green) shows irregular bursts including a characteristic breath burst. TTS energy (red dashed) follows a perfectly predictable sinusoidal pattern — statistically modeled, biologically implausible.",
    params: [
      { id:"P31", name:"RMS Energy Envelope", why:"Root mean square energy per frame — the loudness curve. Real speech energy fluctuates dramatically with breath cycles, stress patterns, and emotion. TTS energy is modeled from average prosody patterns and lacks the random, biological fluctuations of real breath-driven speech.", impact:"★★★☆☆" },
      { id:"P32", name:"Zero Crossing Rate (ZCR)", why:"How often the waveform crosses zero per second. High ZCR = high-frequency content (unvoiced fricatives). Low ZCR = voiced sounds. The ZCR pattern at voiced-to-unvoiced transitions is characteristic of real speech articulation. TTS ZCR transitions are too abrupt or too gradual depending on the system.", impact:"★★★☆☆" },
      { id:"P33", name:"Temporal Modulation Spectrum (4–8 Hz)", why:"Speech amplitude modulates at the syllable rate — roughly 4–8 Hz. Real speech has natural variation in this modulation depth and rate. TTS modulation is unnaturally uniform at a system-characteristic rate. The 4–8 Hz band is critical because it survives codec compression (low frequency, unaffected by bandwidth limiting).", impact:"★★★★☆" },
      { id:"P34", name:"Speech Rate (syllables/sec)", why:"Natural speech rate varies constantly — speakers speed up during fluent passages, slow down at important words, pause unexpectedly. TTS speech rate is set by the model's duration predictor and is unnaturally constant. BiMamba captures this because rate variations play out over 2–5 seconds.", impact:"★★★☆☆" },
      { id:"P35", name:"Inter-phoneme Boundary Timing", why:"The exact timing of transitions between phonemes. In real speech, these durations are influenced by phonetic context, stress, speech rate, and co-articulation in complex, non-linear ways. TTS duration models produce timing that's statistically plausible but misses the fine-grained variability of real articulatory planning.", impact:"★★★★☆" },
    ]
  },
  {
    id: "artifacts",
    label: "H",
    title: "Artifact-Specific Signatures",
    color: C.red,
    badge: "P36–P40",
    branch: "Both Branches",
    intro: `These are the 'smoking gun' features — patterns that literally cannot exist in real speech and only appear in synthetic audio. They're not about what's missing (like jitter) but about what's wrongly present. Every GAN vocoder leaves a forensic fingerprint in the audio it generates. These features are designed to catch that fingerprint.`,
    diagram: <ArtifactDiagram />,
    diagramCaption: "Left: neural vocoders create a sharp spectral cutoff — above a certain frequency, energy drops to near-zero (a hard boundary that real speech never shows). Right: GAN vocoders create a periodic grid pattern at pitch-harmonic intervals — visible as vertical striping in the spectrogram.",
    params: [
      { id:"P36", name:"Spectral Over-smoothing Index", why:"The single most reliable indicator. Computed as the variance of spectral features across time — low variance means the spectrum is suspiciously stable. Real speech has high temporal variance because articulation is constantly changing. Any TTS system, no matter how good, produces a smoother temporal trajectory than a real human.", impact:"★★★★★" },
      { id:"P37", name:"Vocoder Frequency Cutoff Signature", why:"Neural vocoders (HiFi-GAN, WaveGlow, MelGAN) are trained on speech sampled at a fixed rate and band-limited to a specific maximum frequency. This creates a sharp spectral ceiling — energy just stops above that frequency. Real wideband speech tapers gradually. Even after phone codec compression, the shape of this cutoff region differs from real speech.", impact:"★★★★★" },
      { id:"P38", name:"Background Noise Stationarity", why:"Real phone calls have dynamic background noise — traffic changes, people move, ventilation fluctuates. TTS background is either perfectly silent (suspicious for a 'real call') or has stationary noise added as post-processing. The stationarity of the noise floor across the call is a powerful indicator.", impact:"★★★☆☆" },
      { id:"P39", name:"Sub-band Energy Ratios (8 bands)", why:"The ratio of energy in 8 frequency sub-bands. Neural vocoders don't perfectly replicate the natural energy distribution across the full spectrum — they tend to over-generate energy in certain sub-bands and under-generate in others in a system-specific, consistent pattern. This pattern is essentially the vocoder's fingerprint.", impact:"★★★★☆" },
      { id:"P40", name:"Periodic Noise Pattern (Vocoder Grid)", why:"GAN vocoders generate audio by predicting frames conditioned on the previous frame. This autoregressive process creates a subtle, periodic pattern in the generated noise floor — visible in the spectrogram as faint vertical striping at pitch-harmonic intervals. Absent in all real speech. Unique to specific vocoder architectures.", impact:"★★★★★" },
    ]
  }
];

// ════════════════════════════════════════════════════════════════════════════
// PARAM CARD
// ════════════════════════════════════════════════════════════════════════════
function ParamCard({ param, color }) {
  const [open, setOpen] = useState(false);
  const stars = param.impact.split("").filter(c => c === "★").length;
  const empty = 5 - stars;
  return (
    <div
      onClick={() => setOpen(o => !o)}
      style={{
        background: C.surf2,
        border: `1px solid ${open ? color + "55" : C.border}`,
        borderRadius: 10,
        padding: "12px 14px",
        cursor: "pointer",
        transition: "border-color .2s, transform .15s",
        transform: open ? "none" : undefined,
      }}
    >
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", gap:8 }}>
        <div style={{ flex:1 }}>
          <div style={{ fontFamily:"monospace", fontSize:10, color: C.dim, marginBottom:3 }}>{param.id}</div>
          <div style={{ fontWeight:600, fontSize:13, color: C.text, lineHeight:1.3 }}>{param.name}</div>
        </div>
        <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap:4 }}>
          <div style={{ fontSize:12, letterSpacing:1 }}>
            {"★".repeat(stars).split("").map((_,i) => (
              <span key={i} style={{ color }}>{_}</span>
            ))}
            {"☆".repeat(empty).split("").map((_,i) => (
              <span key={i} style={{ color: C.border }}>{_}</span>
            ))}
          </div>
          <div style={{ fontFamily:"monospace", fontSize:9, color: open ? color : C.dim, transition:"color .2s" }}>
            {open ? "▲ less" : "▼ why this?"}
          </div>
        </div>
      </div>
      {open && (
        <div style={{
          marginTop:10,
          fontSize:12,
          color: C.muted,
          lineHeight:1.65,
          borderTop:`1px solid ${C.border}`,
          paddingTop:10,
        }}>
          {param.why}
        </div>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// GROUP SECTION
// ════════════════════════════════════════════════════════════════════════════
function GroupSection({ g }) {
  const [open, setOpen] = useState(g.id === "mel");
  return (
    <div style={{
      marginBottom: 24,
      border: `1px solid ${C.border}`,
      borderRadius: 14,
      overflow:"hidden",
      background: C.surface,
    }}>
      {/* header */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display:"flex", alignItems:"center", gap:14,
          padding:"16px 22px",
          borderBottom: open ? `1px solid ${C.border}` : "none",
          cursor:"pointer",
          background: open ? C.surf2 : C.surface,
          transition:"background .2s",
        }}
      >
        <div style={{
          width:32, height:32, borderRadius:"50%",
          background: g.color + "22",
          border:`1px solid ${g.color}44`,
          display:"flex", alignItems:"center", justifyContent:"center",
          fontWeight:700, fontSize:13, color:g.color,
          fontFamily:"monospace", flexShrink:0,
        }}>{g.label}</div>

        <div style={{ flex:1 }}>
          <div style={{ fontWeight:700, fontSize:15, color:C.text, letterSpacing:"-0.2px" }}>{g.title}</div>
          <div style={{ fontSize:11, color:C.dim, marginTop:2 }}>
            <span style={{
              background: g.color+"18", color:g.color,
              padding:"1px 7px", borderRadius:10,
              fontFamily:"monospace", fontSize:10,
              border:`1px solid ${g.color}30`,
              marginRight:8,
            }}>{g.badge}</span>
            <span>{g.branch}</span>
          </div>
        </div>

        <div style={{ color:C.dim, fontSize:18, transition:"transform .3s", transform: open ? "rotate(180deg)" : "none" }}>▾</div>
      </div>

      {open && (
        <div style={{ padding:"20px 22px" }}>
          {/* intro */}
          <p style={{ fontSize:13, color:C.muted, lineHeight:1.7, marginBottom:20,
            borderLeft:`3px solid ${g.color}66`, paddingLeft:14 }}>
            {g.intro}
          </p>

          {/* diagram */}
          {g.diagram && (
            <div style={{ marginBottom:8 }}>
              {g.diagram}
              <p style={{ fontSize:11, color:C.dim, marginTop:6, textAlign:"center", lineHeight:1.5 }}>
                {g.diagramCaption}
              </p>
            </div>
          )}

          {/* params */}
          <div style={{
            display:"grid",
            gridTemplateColumns:"repeat(auto-fill, minmax(240px, 1fr))",
            gap:10, marginTop:16,
          }}>
            {g.params.map(p => <ParamCard key={p.id} param={p} color={g.color} />)}
          </div>
        </div>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// APP
// ════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [filter, setFilter] = useState("all");

  const filtered = filter === "all" ? GROUPS :
    GROUPS.filter(g =>
      filter === "cnn"   ? g.branch.includes("CNN") :
      filter === "mamba" ? g.branch.includes("BiMamba") :
      g.branch === "Both Branches"
    );

  return (
    <div style={{ background:C.bg, minHeight:"100vh", fontFamily:"'Inter',system-ui,sans-serif", color:C.text }}>
      {/* grid bg */}
      <div style={{
        position:"fixed", inset:0, pointerEvents:"none", zIndex:0,
        backgroundImage:`linear-gradient(${C.border}33 1px,transparent 1px),linear-gradient(90deg,${C.border}33 1px,transparent 1px)`,
        backgroundSize:"40px 40px",
      }}/>

      <div style={{ maxWidth:780, margin:"0 auto", padding:"40px 16px 80px", position:"relative", zIndex:1 }}>

        {/* header */}
        <div style={{ marginBottom:40, borderBottom:`1px solid ${C.border}`, paddingBottom:28 }}>
          <div style={{ fontFamily:"monospace", fontSize:10, color:C.cyan, letterSpacing:3, textTransform:"uppercase", marginBottom:12 }}>
            ▸ Parameter Reference Guide
          </div>
          <h1 style={{
            fontSize:"clamp(24px,5vw,40px)", fontWeight:800,
            letterSpacing:"-1.5px", lineHeight:1.1, marginBottom:12,
            background:`linear-gradient(135deg,#fff 0%,${C.cyan} 55%,${C.purple} 100%)`,
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent",
          }}>
            The 40 Parameters<br/>— Explained Visually
          </h1>
          <p style={{ fontSize:14, color:C.dim, lineHeight:1.65, maxWidth:560 }}>
            Every parameter we chose has a specific reason — a real, measurable difference between human and synthetic speech. Click any group to expand it. Click "▼ why this?" on any parameter to see the explanation. Star ratings show detection impact.
          </p>
        </div>

        {/* filter */}
        <div style={{ display:"flex", gap:8, marginBottom:28, flexWrap:"wrap" }}>
          {[
            { k:"all",   label:"All 40 Parameters" },
            { k:"cnn",   label:"CNN Branch Only" },
            { k:"mamba", label:"BiMamba Branch Only" },
            { k:"both",  label:"Both Branches" },
          ].map(f => (
            <button key={f.k} onClick={() => setFilter(f.k)} style={{
              padding:"6px 14px",
              borderRadius:20,
              border:`1px solid ${filter===f.k ? C.cyan : C.border}`,
              background: filter===f.k ? C.cyan+"18" : C.surface,
              color: filter===f.k ? C.cyan : C.dim,
              fontSize:12, fontFamily:"monospace",
              cursor:"pointer", letterSpacing:"0.5px",
              transition:"all .2s",
            }}>{f.label}</button>
          ))}
        </div>

        {/* impact legend */}
        <div style={{
          display:"flex", gap:16, alignItems:"center",
          marginBottom:24, padding:"10px 16px",
          background:C.surf2, borderRadius:10,
          border:`1px solid ${C.border}`, flexWrap:"wrap",
        }}>
          <span style={{ fontFamily:"monospace", fontSize:10, color:C.dim, letterSpacing:1 }}>IMPACT:</span>
          {[
            {stars:5, label:"Critical"},
            {stars:4, label:"High"},
            {stars:3, label:"Medium"},
          ].map(({stars,label}) => (
            <div key={stars} style={{ display:"flex", alignItems:"center", gap:5, fontSize:11 }}>
              <span style={{ color:C.amber }}>{"★".repeat(stars)}</span>
              <span style={{ color:C.dim }}>{label}</span>
            </div>
          ))}
        </div>

        {/* groups */}
        {filtered.map(g => <GroupSection key={g.id} g={g} />)}

        {/* footer note */}
        <div style={{
          marginTop:32, padding:"16px 20px",
          background:"rgba(0,229,255,0.05)",
          border:`1px solid ${C.cyan}22`,
          borderRadius:12, fontSize:12, color:C.dim, lineHeight:1.65,
        }}>
          <span style={{ color:C.cyan, fontWeight:700 }}>Honest note: </span>
          No single parameter catches all deepfakes. The power of this system is in the combination — parameters that catch GAN vocoders (P37, P40) miss voice conversion systems, while prosody features (P17–P22) catch TTS but less so voice cloning. The dual-branch architecture with mutual cross-attention is designed so the CNN catches local spectral artifacts while BiMamba catches global temporal implausibilities — together they cover the full attack space.
        </div>
      </div>
    </div>
  );
}