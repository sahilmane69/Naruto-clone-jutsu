const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const overlayButton = document.querySelector(".video-overlay-btn");
const overlayImage = document.getElementById("overlayImg");
const errorEl = document.getElementById("error");
const confidenceEl = document.querySelector(".confidence");
const detectionSound = new Audio("assets/naruto_shadow_clones.mp3");

detectionSound.preload = "auto";
detectionSound.volume = 0.9;

let clonesTriggered = false;
let cloneStartTime = null;
let mask = null;
let gestureModel = null;
let canAutoTrigger = false;
let audioUnlocked = false;
let smoothedConfidence = 0;
let detectionStreak = 0;

const DETECTION_THRESHOLD = 0.58;
const DETECTION_STREAK_FRAMES = 3;
const CONFIDENCE_SMOOTHING = 0.25;

const MODEL_CANDIDATE_PATHS = [
  "gesture-model.json",
  "./gesture-model.json",
  "assets/gesture-model.json",
  "./assets/gesture-model.json"
];

const customClones = [
  { x: -100, y: 100, scale: 0.9, delay: 1000, smokeSpawned: false },
  { x: 120, y: 100, scale: 0.85, delay: 1150, smokeSpawned: false },
  { x: -180, y: 140, scale: 0.8, delay: 1300, smokeSpawned: false },
  { x: -140, y: 140, scale: 0.45, delay: 1320, smokeSpawned: false },
  { x: 180, y: 160, scale: 0.7, delay: 1450, smokeSpawned: false },
  { x: 140, y: 160, scale: 0.4, delay: 1470, smokeSpawned: false },
  { x: -250, y: 140, scale: 0.7, delay: 1600, smokeSpawned: false },
  { x: -220, y: 140, scale: 0.35, delay: 1620, smokeSpawned: false },
  { x: 260, y: 160, scale: 0.65, delay: 1750, smokeSpawned: false },
  { x: -100, y: 150, scale: 0.6, delay: 2500, smokeSpawned: false },
  { x: 100, y: 150, scale: 0.6, delay: 2650, smokeSpawned: false },
  { x: -120, y: 70, scale: 0.55, delay: 2800, smokeSpawned: false },
  { x: 100, y: 70, scale: 0.5, delay: 2950, smokeSpawned: false },
  { x: -200, y: 85, scale: 0.55, delay: 3100, smokeSpawned: false },
  { x: 230, y: 85, scale: 0.5, delay: 3250, smokeSpawned: false },
  { x: -280, y: 100, scale: 0.4, delay: 3400, smokeSpawned: false }
];

const SMOKE_FOLDERS = ["smoke_1", "smoke_2", "smoke_3"];
const SMOKE_FRAME_COUNT = 5;
const SMOKE_DURATION = 600;
const activeSmokes = [];

const FINGER_INDICES = {
  thumb: [0, 1, 2, 3, 4],
  index: [0, 5, 6, 7, 8],
  middle: [0, 9, 10, 11, 12],
  ring: [0, 13, 14, 15, 16],
  pinky: [0, 17, 18, 19, 20]
};

function showMessage(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function hideMessage() {
  errorEl.textContent = "";
  errorEl.classList.add("hidden");
}

function resetOverlayImage() {
  overlayImage.src = "assets/state-1.png";
  overlayImage.dataset.state = "1";
}

function toggleImage() {
  if (overlayImage.dataset.state === "2") {
    return;
  }

  overlayImage.src = "assets/state-2.png";
  overlayImage.dataset.state = "2";
  overlayButton.classList.add("pop");
  setTimeout(() => overlayButton.classList.remove("pop"), 200);
}

function withMirroredContext(drawFn) {
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  drawFn();
  ctx.restore();
}

function normalizeHand(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const scale = Math.sqrt(
    (middleMcp.x - wrist.x) ** 2 +
    (middleMcp.y - wrist.y) ** 2 +
    (middleMcp.z - wrist.z) ** 2
  ) || 1;

  const normalized = [];
  for (let index = 0; index < 21; index += 1) {
    normalized.push((landmarks[index].x - wrist.x) / scale);
    normalized.push((landmarks[index].y - wrist.y) / scale);
    normalized.push((landmarks[index].z - wrist.z) / scale);
  }

  return normalized;
}

function getModelConfidence(right, left) {
  if (!gestureModel || !right || !left) {
    return 0;
  }

  const input = tf.tensor2d([[...normalizeHand(right), ...normalizeHand(left)]]);
  const probability = gestureModel.predict(input).dataSync()[0];
  input.dispose();

  return probability;
}

function predictGesture(right, left, threshold = 0.999) {
  const probability = getModelConfidence(right, left);

  confidenceEl.textContent = `${(probability * 100).toFixed(1)}%`;
  return probability > threshold;
}

function pointDistance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function isFingerExtended(landmarks, mcpIndex, pipIndex, tipIndex) {
  const mcp = landmarks[mcpIndex];
  const pip = landmarks[pipIndex];
  const tip = landmarks[tipIndex];
  return tip.y < pip.y && pip.y < mcp.y;
}

function estimateCloneSignConfidence(right, left) {
  if (!right || !left) {
    return 0;
  }

  const rightWrist = right[0];
  const leftWrist = left[0];
  const wristDistance = pointDistance(rightWrist, leftWrist);
  const palmsCloseScore = Math.max(0, Math.min(1, (0.42 - wristDistance) / 0.3));

  const rightIndexExtended = isFingerExtended(right, 5, 6, 8);
  const rightMiddleExtended = isFingerExtended(right, 9, 10, 12);
  const leftIndexExtended = isFingerExtended(left, 5, 6, 8);
  const leftMiddleExtended = isFingerExtended(left, 9, 10, 12);

  const rightRingFolded = !isFingerExtended(right, 13, 14, 16);
  const rightPinkyFolded = !isFingerExtended(right, 17, 18, 20);
  const leftRingFolded = !isFingerExtended(left, 13, 14, 16);
  const leftPinkyFolded = !isFingerExtended(left, 17, 18, 20);

  const rightIndexTip = right[8];
  const leftIndexTip = left[8];
  const rightMiddleTip = right[12];
  const leftMiddleTip = left[12];

  const indexTipDistance = pointDistance(rightIndexTip, leftIndexTip);
  const middleTipDistance = pointDistance(rightMiddleTip, leftMiddleTip);
  const tipDistanceAverage = (indexTipDistance + middleTipDistance) / 2;
  const tipClosenessScore = Math.max(0, Math.min(1, (0.22 - tipDistanceAverage) / 0.18));

  const wristHeightDelta = Math.abs(rightWrist.y - leftWrist.y);
  const wristAlignScore = Math.max(0, Math.min(1, (0.2 - wristHeightDelta) / 0.2));

  const extensionScore = (
    Number(rightIndexExtended) +
    Number(rightMiddleExtended) +
    Number(leftIndexExtended) +
    Number(leftMiddleExtended)
  ) / 4;

  const foldScore = (
    Number(rightRingFolded) +
    Number(rightPinkyFolded) +
    Number(leftRingFolded) +
    Number(leftPinkyFolded)
  ) / 4;

  return (
    0.25 * palmsCloseScore +
    0.25 * extensionScore +
    0.15 * foldScore +
    0.25 * tipClosenessScore +
    0.1 * wristAlignScore
  );
}

function detectGesture(right, left) {
  const confidence = estimateCloneSignConfidence(right, left);
  const modelConfidence = gestureModel ? getModelConfidence(right, left) : 0;
  const rawConfidence = Math.max(confidence, modelConfidence);

  smoothedConfidence =
    smoothedConfidence * (1 - CONFIDENCE_SMOOTHING) +
    rawConfidence * CONFIDENCE_SMOOTHING;

  confidenceEl.textContent = `${(smoothedConfidence * 100).toFixed(1)}%`;

  if (gestureModel && modelConfidence >= 0.995) {
    detectionStreak = DETECTION_STREAK_FRAMES;
    return true;
  }

  if (smoothedConfidence >= DETECTION_THRESHOLD) {
    detectionStreak += 1;
  } else {
    detectionStreak = Math.max(0, detectionStreak - 1);
  }

  return detectionStreak >= DETECTION_STREAK_FRAMES;
}

async function useModel(model, source) {
  if (gestureModel && gestureModel !== model) {
    gestureModel.dispose();
  }

  gestureModel = model;
  canAutoTrigger = true;
  hideMessage();
  console.log("Gesture model active from", source);
}

async function loadGestureModel() {
  for (const modelPath of MODEL_CANDIDATE_PATHS) {
    try {
      const loadedModel = await tf.loadLayersModel(modelPath);
      await useModel(loadedModel, modelPath);
      return;
    } catch (error) {
      console.warn("Failed to load model from", modelPath, error);
    }
  }

  canAutoTrigger = true;
  showMessage("Using optimized built-in gesture detection.");
}

function resetCloneState() {
  clonesTriggered = false;
  cloneStartTime = null;
  smoothedConfidence = 0;
  detectionStreak = 0;
  activeSmokes.length = 0;
  customClones.forEach((clone) => {
    clone.smokeSpawned = false;
  });
  resetOverlayImage();
}

function triggerClones() {
  if (clonesTriggered) {
    return;
  }

  clonesTriggered = true;
  cloneStartTime = performance.now();
  customClones.forEach((clone) => {
    clone.smokeSpawned = false;
  });
  detectionSound.currentTime = 0;
  detectionSound.play().catch(() => {});
  console.log("CLONE TRIGGERED");
}

function unlockDetectionSound() {
  if (audioUnlocked) {
    return;
  }

  audioUnlocked = true;
  detectionSound.muted = true;
  detectionSound.play()
    .then(() => {
      detectionSound.pause();
      detectionSound.currentTime = 0;
      detectionSound.muted = false;
    })
    .catch(() => {
      detectionSound.muted = false;
    });
}

function spawnSmoke(x, y, scale) {
  const folder = SMOKE_FOLDERS[Math.floor(Math.random() * SMOKE_FOLDERS.length)];
  const frames = [];

  for (let frame = 1; frame <= SMOKE_FRAME_COUNT; frame += 1) {
    const image = new Image();
    image.src = `assets/${folder}/${frame}.png`;
    frames.push(image);
  }

  activeSmokes.push({
    x,
    y,
    scale: scale * 1.2,
    start: performance.now(),
    frames
  });
}

function drawSmokes() {
  const now = performance.now();

  for (let index = activeSmokes.length - 1; index >= 0; index -= 1) {
    const smoke = activeSmokes[index];
    const elapsed = now - smoke.start;
    const frameDuration = SMOKE_DURATION / SMOKE_FRAME_COUNT;
    const frameIndex = Math.floor(elapsed / frameDuration);

    if (frameIndex >= smoke.frames.length) {
      activeSmokes.splice(index, 1);
      continue;
    }

    const image = smoke.frames[frameIndex];
    ctx.save();
    ctx.translate(smoke.x, smoke.y);
    ctx.scale(smoke.scale, smoke.scale);
    ctx.drawImage(image, -image.width / 2, -image.height / 2);
    ctx.restore();
  }
}

function drawClones(person) {
  const now = performance.now();
  const sortedClones = [...customClones].sort((first, second) => second.delay - first.delay);

  sortedClones.forEach((clone) => {
    if (now - cloneStartTime >= clone.delay) {
      ctx.save();
      ctx.translate(clone.x + canvas.width * (1 - clone.scale) / 2, clone.y);
      ctx.scale(clone.scale, clone.scale);
      ctx.drawImage(person, 0, 0);
      ctx.restore();
    }
  });

  ctx.drawImage(person, 0, 0);
}

function grabPerson() {
  const offscreen = document.createElement("canvas");
  offscreen.width = canvas.width;
  offscreen.height = canvas.height;
  const offscreenCtx = offscreen.getContext("2d");

  offscreenCtx.drawImage(mask, 0, 0, canvas.width, canvas.height);
  offscreenCtx.globalCompositeOperation = "source-in";
  offscreenCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
  offscreenCtx.globalCompositeOperation = "source-over";

  return offscreen;
}

function drawFingerSkeleton(landmarks) {
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;

  Object.values(FINGER_INDICES).forEach((indices) => {
    ctx.beginPath();
    indices.forEach((landmarkIndex, segmentIndex) => {
      const x = landmarks[landmarkIndex].x * canvas.width;
      const y = landmarks[landmarkIndex].y * canvas.height;
      if (segmentIndex === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });

  landmarks.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x * canvas.width, point.y * canvas.height, 3, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();
  });
}

const selfie = new SelfieSegmentation({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`
});

selfie.setOptions({ modelSelection: 1 });
selfie.onResults((results) => {
  mask = results.segmentationMask;
});

const holistic = new Holistic({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true
});

holistic.onResults((results) => {
  if (!mask) {
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const person = grabPerson();

  if (!clonesTriggered && canAutoTrigger && detectGesture(results.rightHandLandmarks, results.leftHandLandmarks)) {
    triggerClones();
  }

  withMirroredContext(() => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (clonesTriggered) {
      const now = performance.now();
      customClones.forEach((clone) => {
        if (!clone.smokeSpawned && now - cloneStartTime >= clone.delay) {
          clone.smokeSpawned = true;
          const centerX = clone.x + canvas.width / 2;
          const centerY = clone.y + canvas.height / 2 - 40;
          spawnSmoke(centerX - 15, centerY, clone.scale);
          spawnSmoke(centerX + 15, centerY, clone.scale);
        }
      });

      toggleImage();
      drawClones(person);
      drawSmokes();
    } else {
      ctx.drawImage(person, 0, 0);
    }

    if (results.rightHandLandmarks) {
      drawFingerSkeleton(results.rightHandLandmarks);
    }

    if (results.leftHandLandmarks) {
      drawFingerSkeleton(results.leftHandLandmarks);
    }
  });
});

const camera = new Camera(video, {
  width: 640,
  height: 480,
  onFrame: async () => {
    await selfie.send({ image: video });
    await holistic.send({ image: video });
  }
});

overlayButton.addEventListener("click", triggerClones);
document.addEventListener("pointerdown", unlockDetectionSound, { once: true });
document.addEventListener("keydown", unlockDetectionSound, { once: true });
window.addEventListener("load", resetCloneState);

loadGestureModel();
camera.start();
