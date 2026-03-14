const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("train-status");
const recordBadge = document.getElementById("rec-badge");
const cloneCountEl = document.getElementById("count-clone");
const otherCountEl = document.getElementById("count-other");
const trainButton = document.getElementById("btn-train");
const importInput = document.getElementById("import-data-input");
const countElements = {
  clone_sign: cloneCountEl,
  not_sign: otherCountEl
};

const holistic = new Holistic({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true });

const camera = new Camera(video, {
  width: 640,
  height: 480,
  onFrame: async () => {
    await holistic.send({ image: video });
  }
});

let samples = { clone_sign: [], not_sign: [] };
let recording = null;
let model = null;
let countdownTimer = null;
let recordTimer = null;

const COUNTDOWN = 3;
const RECORD_TIME = 4;
const HAND_SEGMENTS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20]
];

function withMirroredContext(drawFn) {
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  drawFn();
  ctx.restore();
}

function drawHand(landmarks) {
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 2;

  HAND_SEGMENTS.forEach((segment) => {
    ctx.beginPath();
    segment.forEach((landmarkIndex, segmentIndex) => {
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
    ctx.fillStyle = "#ef4444";
    ctx.fill();
  });
}

function normalize(landmarks) {
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

function extract(right, left) {
  return [...normalize(right), ...normalize(left)];
}

function updateCounts() {
  cloneCountEl.textContent = samples.clone_sign.length;
  otherCountEl.textContent = samples.not_sign.length;
}

function setBadge(text, active) {
  recordBadge.textContent = text;
  recordBadge.classList.toggle("active", active);
}

function captureFrame(right, left) {
  if (!recording || !right || !left) {
    return;
  }

  samples[recording].push(extract(right, left));
  countElements[recording].textContent = samples[recording].length;
}

function stopRec() {
  recording = null;
  clearInterval(recordTimer);
  recordTimer = null;
  setBadge("● REC", false);
}

function cancelRecording() {
  clearInterval(countdownTimer);
  clearInterval(recordTimer);
  countdownTimer = null;
  recordTimer = null;
  recording = null;
  setBadge("● REC", false);
}

function startRec(label) {
  recording = label;
  let remaining = RECORD_TIME;

  setBadge(`● REC ${remaining}s`, true);
  statusEl.textContent = `Recording "${label === "clone_sign" ? "clone sign" : "other"}" — hold your pose!`;

  recordTimer = setInterval(() => {
    remaining -= 1;
    if (remaining > 0) {
      setBadge(`● REC ${remaining}s`, true);
      return;
    }

    stopRec();
    statusEl.textContent = "Done! Captured samples. Record more or train.";
  }, 1000);
}

function startCountdown(label) {
  cancelRecording();
  let remaining = COUNTDOWN;

  setBadge(`GET READY… ${remaining}`, true);
  statusEl.textContent = `Recording "${label === "clone_sign" ? "clone sign" : "other"}" in ${remaining}s — get into position!`;

  countdownTimer = setInterval(() => {
    remaining -= 1;
    if (remaining > 0) {
      setBadge(`GET READY… ${remaining}`, true);
      statusEl.textContent = `Recording in ${remaining}s — get into position!`;
      return;
    }

    clearInterval(countdownTimer);
    countdownTimer = null;
    startRec(label);
  }, 1000);
}

function updateConf(probability) {
  document.getElementById("conf-fill").style.width = `${(probability * 100).toFixed(0)}%`;
  document.getElementById("conf-label").textContent = `${(probability * 100).toFixed(0)}%`;
}

holistic.onResults((results) => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  withMirroredContext(() => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (results.rightHandLandmarks) {
      drawHand(results.rightHandLandmarks);
    }

    if (results.leftHandLandmarks) {
      drawHand(results.leftHandLandmarks);
    }
  });

  captureFrame(results.rightHandLandmarks, results.leftHandLandmarks);

  if (model && results.rightHandLandmarks && results.leftHandLandmarks) {
    const input = tf.tensor2d([extract(results.rightHandLandmarks, results.leftHandLandmarks)]);
    const probability = model.predict(input).dataSync()[0];
    input.dispose();
    updateConf(probability);
  }
});

["btn-rec-clone", "btn-rec-other"].forEach((id) => {
  const label = id.includes("clone") ? "clone_sign" : "not_sign";
  document.getElementById(id).addEventListener("click", () => startCountdown(label));
});

document.addEventListener("keydown", (event) => {
  const keyMap = { "1": "clone_sign", "2": "not_sign" };
  if (!event.repeat && keyMap[event.key]) {
    startCountdown(keyMap[event.key]);
  }
});

trainButton.addEventListener("click", async () => {
  const positiveCount = samples.clone_sign.length;
  const negativeCount = samples.not_sign.length;

  if (positiveCount < 5 || negativeCount < 5) {
    statusEl.textContent = "Need at least 5 samples each.";
    return;
  }

  const xs = [];
  const ys = [];

  samples.clone_sign.forEach((sample) => {
    xs.push(sample);
    ys.push(1);
  });

  samples.not_sign.forEach((sample) => {
    xs.push(sample);
    ys.push(0);
  });

  for (let index = xs.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [xs[index], xs[swapIndex]] = [xs[swapIndex], xs[index]];
    [ys[index], ys[swapIndex]] = [ys[swapIndex], ys[index]];
  }

  const xTensor = tf.tensor2d(xs);
  const yTensor = tf.tensor1d(ys);

  if (model) {
    model.dispose();
  }

  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [126], units: 64, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

  trainButton.disabled = true;
  statusEl.textContent = "Training...";

  await model.fit(xTensor, yTensor, {
    epochs: 50,
    batchSize: 16,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const accuracy = logs.acc ?? logs.accuracy ?? 0;
        statusEl.textContent = `Epoch ${epoch + 1}/50 — acc: ${(accuracy * 100).toFixed(1)}%`;
      }
    }
  });

  xTensor.dispose();
  yTensor.dispose();
  trainButton.disabled = false;
  statusEl.textContent = `Done! ${positiveCount + negativeCount} samples. Model is live — test your sign above.`;
});

document.getElementById("btn-export-data").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(samples)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "gesture-data.json";
  link.click();
});

document.getElementById("btn-import-data").addEventListener("click", () => {
  importInput.click();
});

importInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();
  reader.onload = (loadEvent) => {
    const data = JSON.parse(loadEvent.target.result);
    samples.clone_sign.push(...(data.clone_sign || []));
    samples.not_sign.push(...(data.not_sign || []));
    updateCounts();
    statusEl.textContent = "Data imported.";
    importInput.value = "";
  };
  reader.readAsText(file);
});

document.getElementById("btn-save-model").addEventListener("click", async () => {
  if (!model) {
    statusEl.textContent = "Train a model first.";
    return;
  }

  await model.save("downloads://gesture-model");
  statusEl.textContent = "Model saved — you'll get gesture-model.json + gesture-model.weights.bin";
});

document.getElementById("btn-clear-data").addEventListener("click", () => {
  cancelRecording();
  samples = { clone_sign: [], not_sign: [] };
  updateCounts();
  statusEl.textContent = "Data cleared.";
});

updateCounts();
camera.start();
