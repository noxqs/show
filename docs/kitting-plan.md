# Kitting lane: perception and timing plan

**The cell.** A conveyor carries empty boxes past two arms. Arm 1 picks a box off
the moving belt; arm 2 places a smaller box inside it. The belt does not stop.

**The question this document answers.** Whether scene understanding has to run in
real time for that to work, and if so, what "real time" has to mean.

---

## 1. The reframe

The problem is not *understand the scene fast*. It is **know where a known object
will be at time T, with bounded error**. Those are different engineering
problems, and the second has a deterministic answer that beats any perception
speed.

**We do not need low latency. We need *known* latency.**

At a belt speed of 0.25 m/s, 50 ms is 12.5 mm of travel:

* if that 50 ms is **constant**, predict 12.5 mm forward and the error is ~zero;
* if it is 50 ms **± 20 ms**, there is ±5 mm of irreducible error, and no faster
  model removes it.

This is why industrial conveyor picking has used encoder synchronisation for
thirty years. FANUC's Visual Line Tracking fuses three streams: conveyor-encoder
pulses, a vision-reported part offset, and the calibrated camera-to-robot
transform. Vision runs **once, upstream**; the encoder dead-reckons position from
there. Published figures worth holding on to:

* EtherNet/IP RPI above 10 ms already costs ~1 mm at typical belt speeds;
* a 2000 PPR encoder gives 1 mm resolution on a 200 mm pulley (4000+ PPR gives
  the tracking filter smoother velocity);
* standard tracking runs to ~1500 mm/min, high-speed options to ~3000 mm/min;
  SCARA cells reach ~120 parts/min.

That one design choice moves the "can SAM 3 answer in 50 ms" question from
critical to irrelevant.

---

## 2. The landscape, in the order it constrains us

### Tier 0 — Physics. No model fixes this.

Motion blur, in pixels, is `speed x exposure / mm-per-pixel`. At 1 m/s with
0.1 mm per pixel, staying under half a pixel of blur means a **50 microsecond**
exposure. Separately, a **rolling shutter skews a moving box**: the mask is
geometrically wrong no matter how good the model is.

**Global shutter plus strobed lighting is a prerequisite, not an optimisation.**
It is also the cheapest item on this list.

### Tier 1 — Deterministic tracking. The standard answer.

Trigger a detection upstream (photoeye, or an encoder position), timestamp it,
and hand the result to the robot controller's conveyor-tracking mode.

The payoff for arm 2: in belt-following mode **the box is stationary in the arm's
frame**, so insertion becomes a static problem rather than a moving-target
problem.

### Tier 2 — The perception, which is smaller than it looks.

Empty boxes on a kitting lane are a **closed set**: a few SKUs, known geometry,
lying flat, known approach direction. That is `(x, y, theta)` plus a size class —
not open-vocabulary segmentation.

* a small YOLO-seg or oriented-box model at 640x640: **5-8 ms** on an RTX 4070
  Super (measured class of performance; our own yolo26m-seg runs 12 ms at 20 Hz);
* classical template or edge fitting: sub-millisecond, and it is what Cognex and
  Keyence actually ship for this.

SAM 3 solves a problem this station does not have.

### Tier 3 — 3D, only if the cavity matters.

If arm 2 drops into an open-top box, the **inner cavity pose** is the thing to
measure, and that is where 3D earns its cost.

* **Photoneo MotionCam-3D** is purpose-built for it: parallel structured light in
  a single snapshot, sub-millimetre accuracy at up to 20 fps *on moving objects*,
  rated to objects moving 144 km/h.
* Conventional structured light (Zivid class) needs a static scene. Excellent at
  an indexed station; wrong on a moving belt.

### Tier 4 — Foundation models, out of the control loop.

* FoundationPose-class 6D pose exceeds **200 ms per frame**. Recent bin-picking
  work reports sub-300 ms at under 1 mm error — a per-**pick** budget, not a
  per-**frame** one.
* SAM 3 on our own hardware is far worse than that (see appendix).

They earn their place in three roles, all off the critical path:

1. **Commissioning and auto-labelling** — SAM 3 text prompt produces training
   data for the small fast model. Days become minutes.
2. **Exception handling at 2-5 Hz on a second thread** — crushed box, misfeed,
   foreign object, "something is on the belt that should not be".
3. **Changeover** — a new SKU without re-teaching.

---

## 3. Latency budget

```
exposure (strobed)          0.2 ms
readout + transfer          1-5  ms    USB3 / CoaXPress vs GigE
detect (small model)        5-10 ms
pose + decision             2-5  ms
comms to robot              1-10 ms    RPI dominates; keep <= 10 ms
robot plan / servo update   4-12 ms
                           --------
total                      ~15-40 ms, and bounded
```

Comfortably under 50 ms end to end, **with no foundation model in it**.

**Jitter is the number to hold, not the mean.** A budget that averages 25 ms and
spikes to 90 ms is worse than one that always takes 40 ms.

---

## 4. The unglamorous option, to price first

A **mechanical funnel or lead-in chamfer** on the insert removes the last few
millimetres of accuracy requirement. Combined with compliance — force control, or
simply a compliant tool — it turns "place precisely into a moving target" into
"get close and let the geometry finish".

This routinely beats six months of perception work. Price it before any camera.

---

## 5. Four facts decide which branch we take

1. **Belt speed and cycle rate.** Sets the entire budget, and whether indexing is
   even on the table.
2. **Can the belt index (stop-and-go) at the insert station?** If yes, the hard
   problem disappears and Tier 3 gets much cheaper.
3. **Which robots and controllers?** FANUC, ABB and Omron ship conveyor tracking
   as a controller option. On ROS 2 we build the equivalent ourselves.
4. **Open-top box, or flaps closed?** Decides whether we need 3D at all.

---

## 6. Where `show` fits

**Not in the control loop.** The control loop is a separate deterministic
pipeline: trigger -> detect -> timestamp -> encoder -> robot, with a hard
real-time path and bounded jitter.

`show` is where we *see* what the line sees — tune thresholds, review failures,
replay a misfeed frame by frame, and run SAM 3 at low Hz on the exception path,
where 200 ms costs nothing. The inference work already in the viewer pays off
there; it is simply not the thing the arm waits on.

---

## Appendix: why SAM 3 is not in the loop

Measured and published latency for SAM 3, worst to best:

| Setup | Latency | Hardware |
|---|---|---|
| Our own build, CPU | 6.1 s encode + 1.0 s decode | 32-thread CPU |
| Ultralytics measurement | 2921 ms / image | RTX PRO 6000 |
| ONNX Runtime, CUDA EP | ~1.4 s / image (600x500) | RTX 4090 |
| ONNX Runtime, TensorRT EP | ~1.1 s / image | RTX 4090 |
| Native TensorRT FP16 | 23.6 ms (text prompt, 612x421) | RTX 5090 |
| DART, TensorRT FP16, 1008px | 61-72 ms (1-4 classes) | RTX 4080 |
| DART + RepViT-M2.3 student | 22 ms (45 FPS), 38.7 AP | RTX 4080 |
| Meta's own figure | 30 ms, 100+ objects | H200 |
| SAM 3.1, object multiplexing | 32 fps video, <=16 objects/pass | H100 |

Three things follow:

* **ONNX Runtime is a dead end for this model.** Its TensorRT EP bought only
  ~20 % over CUDA EP, from graph partitioning and operator fallback. Reaching the
  fast numbers means native TensorRT engines, which means leaving ORT for SAM 3.
* **On our RTX 4070 Super** (roughly 0.7x an RTX 4080) the full PE backbone
  projects to ~75 ms and misses; only a distilled student backbone (~30-35 ms,
  38.7 AP) fits, and it gives up accuracy to get there.
* **Even at 30 ms, it would be the wrong tool here** — see section 2, Tier 2. The
  object set is closed.

If we ever do want SAM 3 near real time, the shape is: **identity at 2-5 Hz,
position at 30-60 Hz**. The Perception Encoder backbone is class-agnostic and
dominates the cost (53 ms of DART's 61 ms), so one pass serves every prompt, and
a cheap tracker carries position between passes.

---

## Sources

* [FANUC Visual Line Tracking setup and calibration](https://industrialmonitordirect.com/blogs/knowledgebase/fanuc-visual-line-tracking-conveyor-pick-setup-and-calibration)
* [FANUC iRPickTool conveyor tracking](https://industrialmonitordirect.com/blogs/knowledgebase/fanuc-irpicktool-conveyor-tracking-setup-in-roboguide)
* [Line tracking for robotic pick and place](https://motioncontrolsrobotics.com/resources/tech-talk-articles/line-tracking-robotic-pick-and-place/)
* [Photoneo MotionCam-3D](https://www.photoneo.com/motioncam-3d/)
* [Photoneo: 3D scanning on an overhead conveyor](https://www.photoneo.com/case-study/3d-scanning-of-objects-moving-on-an-overhead-conveyor/)
* [How to avoid motion blur](https://va-imaging.com/en-us/blogs/machine-vision-knowledge-center/how-to-avoid-motion-blur)
* [Global shutter vs rolling shutter](https://www.overview.ai/blog/global-shutter-vs-rolling-shutter/)
* [Triggering and strobing machine vision lighting](https://www.machine-vision-shop.com/blog/triggering-strobing-machine-vision-lighting)
* [Real-time 6D pose estimation for bin picking (IEEE)](https://ieeexplore.ieee.org/document/11233319)
* [Geo6DPose: fast zero-shot 6D pose estimation](https://arxiv.org/html/2512.10674)
* [SAM 3.1 — Meta AI](https://ai.meta.com/blog/segment-anything-model-3/)
* [SAM 3: Segment Anything with Concepts (arXiv)](https://arxiv.org/html/2511.16719v2)
* [DART: Detect Anything in Real Time (arXiv)](https://arxiv.org/html/2603.11441v1) — [code](https://github.com/mkturkcan/DART)
* [SAM 3 TensorRT inference](https://yimin-pan.github.io/sam3-trt/)
* [SAM 3 TensorRT EP latency, issue #424](https://github.com/facebookresearch/sam3/issues/424)
* [SAM 3 — Ultralytics docs](https://docs.ultralytics.com/models/sam-3)
* [EfficientSAM3 (arXiv)](https://arxiv.org/html/2511.15833v1)
