let video;
let bodyPose;
let connections;
let poses = [];
let angle = 0;
let shoulderText, elbowText, wristText;
let rollTextDiv;

let detector; // For MediaPipe Hands
let socket;

let s, e, w; // Shoulder, Elbow, Wrist keypoints
let rollDeg = null; // Shared wrist roll value

function preload() {
  bodyPose = ml5.bodyPose("BlazePose");
}

function gotPoses(results) {
  poses = results;
}

function setup() {
  // Setup WebSocket
  socket = new WebSocket(`ws://${window.location.hostname}:8765`);
  socket.onopen = () => console.log("WebSocket connected");
  socket.onerror = (e) => console.error("WebSocket error", e);

  // Setup canvas
  canvas = createCanvas(640, 360, WEBGL);
  canvas.parent("sketch-holder");

  // Setup video
  video = createCapture(VIDEO);
  video.size(640, 360);
  video.parent("video-holder");
  video.style("transform", "scaleX(-1)");
  video.style("margin-top", "20px");

  video.elt.onloadeddata = () => {
    bodyPose.detectStart(video, gotPoses);
  };

  connections = bodyPose.getSkeleton();

  // Setup debug text
  shoulderText = createDiv("Shoulder: ").parent(document.body);
  elbowText = createDiv("Elbow: ").parent(document.body);
  wristText = createDiv("Wrist: ").parent(document.body);
  rollTextDiv = createDiv("Wrist Roll: ---").parent(document.body);

  // Setup hand model
  setupHandsModel();
}

async function setupHandsModel() {
  const model = handPoseDetection.SupportedModels.MediaPipeHands;
  const detectorConfig = {
    runtime: 'mediapipe',
    modelType: 'lite',
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands'
  };
  detector = await handPoseDetection.createDetector(model, detectorConfig);
}

function draw() {
  scale(height / 2);
  orbitControl();
  angle += 0.02;
  background(0);

  if (poses.length > 0) {
    let pose = poses[0];
    let kp = pose.keypoints3D;

    s = kp[11]; // LEFT_SHOULDER
    e = kp[13]; // LEFT_ELBOW
    w = kp[15]; // LEFT_WRIST

    if (s.confidence > 0.1 && e.confidence > 0.1 && w.confidence > 0.1) {
      shoulderText.html(`LEFT SHOULDER: x=${nf(s.x, 1, 3)} y=${nf(s.y, 1, 3)} z=${nf(s.z, 1, 3)}`);
      elbowText.html(`LEFT ELBOW: x=${nf(e.x, 1, 3)} y=${nf(e.y, 1, 3)} z=${nf(e.z, 1, 3)}`);
      wristText.html(`LEFT WRIST: x=${nf(w.x, 1, 3)} y=${nf(w.y, 1, 3)} z=${nf(w.z, 1, 3)}`);

      // Draw limbs with custom colors
      drawArmSegment(s, e, [0, 255, 200]); // Shoulder to Elbow: Cyan-ish
      drawArmSegment(e, w, [0, 200, 255]); // Elbow to Wrist: Blue-ish

      // Draw joints with different colors
      drawJoint(s, [255, 50, 50], 0.07);   // Shoulder: Red
      drawJoint(e, [50, 255, 50], 0.06);   // Elbow: Green
      drawJoint(w, [50, 50, 255], 0.05);   // Wrist: Blue

      // ✅ Send data if rollDeg is available
      if (rollDeg !== null && socket && socket.readyState === WebSocket.OPEN) {
        const dataToSend = {
          shoulder: { x: s.x, y: s.y, z: s.z },
          elbow: { x: e.x, y: e.y, z: e.z },
          wrist: { x: w.x, y: w.y, z: w.z },
          roll: parseFloat(rollDeg)
        };
        socket.send(JSON.stringify(dataToSend));
      }
    }
  }

  detectHand(); // async roll detection
}

async function detectHand() {
  if (detector && video.loadedmetadata) {
    const hands = await detector.estimateHands(video.elt, { flipHorizontal: true });

    if (hands.length > 0) {
      const hand = hands[0];
      const landmarks = hand.keypoints;

      const index = landmarks.find(pt => pt.name === 'index_finger_mcp');
      const pinky = landmarks.find(pt => pt.name === 'pinky_finger_mcp');

      if (index && pinky) {
        const dx = pinky.x - index.x;
        const dy = pinky.y - index.y;
        const rollRad = Math.atan2(dy, dx);
        rollDeg = degrees(rollRad).toFixed(2); // update global rollDeg

        rollTextDiv.html(`Wrist Roll: ${rollDeg} °`);
        console.log("Wrist Roll (deg):", rollDeg);
      }
    }
  }
}

function drawJoint(point, color, size) {
  if (point.confidence > 0.1) {
    push();
    stroke(color);
    strokeWeight(1);
    fill(color[0], color[1], color[2], 150);
    translate(point.x, point.y, point.z);
    rotateZ(angle);
    box(size);
    pop();
  }
}

// Updated function to draw arm segments with custom colors
function drawArmSegment(a, b, color = [0, 255, 255]) {
  if (a.confidence > 0.1 && b.confidence > 0.1) {
    stroke(color);
    strokeWeight(4);
    beginShape();
    vertex(a.x, a.y, a.z);
    vertex(b.x, b.y, b.z);
    endShape();
  }
}

function mousePressed() {
  console.log(poses);
}