"""
Smart Parking System — Streamlit App (FIXED - No CSS in UI)
Supports: live laptop camera, image upload, and video upload.
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
from datetime import datetime

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════
# CSS STYLING (Fixed - No rendering in UI)
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background: #0b1120; color: #f1f5f9; }
.block-container { padding: 1rem 1.2rem 2rem; max-width: 960px; }

.stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
.stat-card {
    flex:1; min-width:110px;
    background:#1e293b; border:0.5px solid #334155;
    border-radius:14px; padding:14px 10px; text-align:center;
}
.stat-num { font-size:28px; font-weight:600; margin:0; }
.stat-lbl { font-size:11px; color:#94a3b8; margin:4px 0 0; }
.green { color:#22c55e; } .red { color:#ef4444; }
.blue  { color:#38bdf8; } .amber{ color:#f59e0b; }

.slot-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(72px,1fr)); gap:7px; margin-top:10px; }
.slot { border-radius:10px; padding:9px 4px; text-align:center; font-size:12px; font-weight:500; border:0.5px solid transparent; }
.slot.free  { background:#14532d; border-color:#22c55e; color:#86efac; }
.slot.taken { background:#450a0a; border-color:#ef4444; color:#fca5a5; }

.sec-head { font-size:15px; font-weight:500; color:#f8fafc;
    border-left:3px solid #38bdf8; padding-left:10px; margin:18px 0 8px; }

.live-badge {
    display:inline-block; background:#ef4444; color:white;
    border-radius:999px; padding:2px 10px; font-size:11px; font-weight:600;
    animation: pulse 1.4s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

.camera-frame { border:1.5px solid #334155; border-radius:14px; overflow:hidden; }

[data-testid="stFileUploader"] {
    background:#1e293b !important; border:1px dashed #334155 !important;
    border-radius:14px !important; padding:10px !important;
}
.stButton > button {
    background:linear-gradient(135deg,#38bdf8,#6366f1) !important;
    color:white !important; border:none !important;
    border-radius:12px !important; padding:10px 20px !important;
    font-weight:500 !important; width:100%;
}
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# LOAD AI MODEL
# ════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI model…")
def load_detector():
    from ultralytics import YOLO
    path = "model_data/parking_detector.pt"
    return YOLO(path) if os.path.exists(path) else YOLO("yolov8n.pt")


# ════════════════════════════════════════════════════════════════════
# DETECTION FUNCTIONS
# ════════════════════════════════════════════════════════════════════
def detect_vehicles(model, frame_rgb: np.ndarray, conf: float = 0.35):
    """Detect vehicles in frame using YOLO"""
    results = model(frame_rgb, conf=conf, iou=0.45, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(box.conf[0]), "cls": cls
                })
    return boxes


def build_slots(h, w, rows, cols):
    """Create parking slot grid"""
    rh, cw = h // rows, w // cols
    return [(c*cw, r*rh, (c+1)*cw, (r+1)*rh)
            for r in range(rows) for c in range(cols)]


def analyze(frame_rgb: np.ndarray, slots: list, boxes: list):
    """Analyze which slots are occupied"""
    img = frame_rgb.copy()
    occupancy = []
    
    for i, (sx1, sy1, sx2, sy2) in enumerate(slots):
        occupied = False
        
        # Check if any vehicle overlaps this slot
        for b in boxes:
            ox1, oy1 = max(b["x1"], sx1), max(b["y1"], sy1)
            ox2, oy2 = min(b["x2"], sx2), min(b["y2"], sy2)
            inter = max(0, ox2 - ox1) * max(0, oy2 - oy1)
            slot_area = (sx2 - sx1) * (sy2 - sy1)
            
            if slot_area > 0 and inter / slot_area > 0.25:
                occupied = True
                break
        
        # Draw slot
        color = (239, 68, 68) if occupied else (34, 197, 94)  # Red or Green
        label = f"S{i+1} {'TAKEN' if occupied else 'FREE'}"
        
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, 2)
        cv2.rectangle(img, (sx1, max(sy1-22, 0)), (sx1 + len(label)*9, sy1), color, -1)
        cv2.putText(img, label, (sx1+3, max(sy1-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        
        occupancy.append({"slot": i+1, "occupied": occupied})
    
    # Draw detected vehicles
    for b in boxes:
        cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (56, 189, 248), 1)
    
    return img, occupancy


# ════════════════════════════════════════════════════════════════════
# UI RENDERING FUNCTIONS
# ════════════════════════════════════════════════════════════════════
def render_stat_cards(total, free, taken, vehicles):
    """Display KPI stat cards"""
    pct = int(free / total * 100) if total else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Slots", total)
    with col2:
        st.metric("Available", free)
    with col3:
        st.metric("Occupied", taken)
    with col4:
        st.metric("Free %", f"{pct}%")
    with col5:
        st.metric("Vehicles", vehicles)


def render_slot_map(occupancy):
    """Display slot occupancy grid"""
    cols = st.columns(8)
    col_idx = 0
    
    for slot in occupancy:
        col = cols[col_idx % 8]
        with col:
            if slot["occupied"]:
                st.markdown("🔴 **S" + str(slot["slot"]) + "**\nOccupied")
            else:
                st.markdown("🟢 **S" + str(slot["slot"]) + "**\nFree")
        col_idx += 1


# ════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "cam_running" not in st.session_state:
    st.session_state.cam_running = False
if "cam_paused" not in st.session_state:
    st.session_state.cam_paused = False

# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
st.title("🅿️ Smart Parking AI")
st.caption("Real-time vehicle detection & slot occupancy analysis")

try:
    model = load_detector()
    model_ok = True
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")
    st.info("Make sure you have ultralytics installed: `pip install ultralytics`")
    model_ok = False

# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════
tab_live, tab_upload, tab_history, tab_info = st.tabs(
    ["📹 Live Camera", "📷 Upload Image", "📊 History", "ℹ️ How it works"]
)


# ════════════════════════════════════════════════════════════════════
# TAB 1: LIVE CAMERA
# ════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("📹 Live Camera Detection")
    
    # Settings
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cam_source = st.selectbox(
            "Camera",
            ["Laptop camera (0)", "Camera 1", "Camera 2"],
            index=0
        )
        cam_idx = int(cam_source.split("(")[-1].replace(")", ""))
    
    with col2:
        live_conf = st.slider("Confidence", 0.1, 0.9, 0.35, 0.05, key="live_conf")
    
    with col3:
        live_rows = st.slider("Slot rows", 1, 8, 3, key="live_rows")
    
    with col4:
        live_cols = st.slider("Slot cols", 1, 12, 4, key="live_cols")
    
    # Control buttons
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("▶ Start Camera", key="start_cam"):
            st.session_state.cam_running = True
            st.session_state.cam_paused = False
    
    with b2:
        if st.button("⏹ Stop Camera", key="stop_cam"):
            st.session_state.cam_running = False
    
    with b3:
        snap_btn = st.button("📸 Snapshot", key="snap_cam")
    
    # Status
    if st.session_state.cam_running:
        st.warning("🔴 **LIVE - Camera is running**")
    else:
        st.info("Camera is off. Click 'Start Camera' to begin.")
    
    # Placeholders
    frame_ph = st.empty()
    stats_ph = st.empty()
    slotmap_ph = st.empty()
    
    # ─── CAMERA LOOP ───────────────────────────────────────────────────────
    if st.session_state.cam_running and model_ok:
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            st.error("❌ Camera not found. Try another camera or close other apps using it.")
            st.session_state.cam_running = False
        else:
            frame_count = 0
            last_boxes = []
            last_occ = []
            
            while st.session_state.cam_running:
                ok, frame_bgr = cap.read()
                if not ok:
                    st.warning("Camera read failed")
                    time.sleep(0.1)
                    continue
                
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                slots = build_slots(h, w, live_rows, live_cols)
                
                # Run detection every 3 frames
                frame_count += 1
                if frame_count % 3 == 0:
                    last_boxes = detect_vehicles(model, frame_rgb, conf=live_conf)
                
                annotated, last_occ = analyze(frame_rgb, slots, last_boxes)
                
                # Timestamp
                ts = datetime.now().strftime("%H:%M:%S")
                cv2.putText(annotated, f"Smart Parking AI  {ts}",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (56, 189, 248), 1)
                
                # Display
                frame_ph.image(annotated, channels="RGB", use_container_width=True)
                
                # Stats (update every 6 frames)
                if frame_count % 6 == 0 and last_occ:
                    total = len(last_occ)
                    taken = sum(1 for s in last_occ if s["occupied"])
                    free = total - taken
                    
                    with stats_ph.container():
                        render_stat_cards(total, free, taken, len(last_boxes))
                    
                    with slotmap_ph.container():
                        st.subheader("Slot Map")
                        render_slot_map(last_occ)
                
                # Snapshot
                if snap_btn and last_occ:
                    total = len(last_occ)
                    taken = sum(1 for s in last_occ if s["occupied"])
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "file": f"snapshot_{ts}",
                        "total": total,
                        "free": total - taken,
                        "taken": taken,
                        "vehicles": len(last_boxes),
                    })
                    st.success("✅ Snapshot saved!")
                
                time.sleep(0.04)
            
            cap.release()
    
    elif not st.session_state.cam_running:
        frame_ph.info("📹 Click **▶ Start Camera** to begin live detection")


# ════════════════════════════════════════════════════════════════════
# TAB 2: UPLOAD IMAGE
# ════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("📷 Upload Image")
    
    with st.expander("⚙️ Settings", expanded=False):
        ca, cb, cc = st.columns(3)
        with ca:
            conf_thresh = st.slider("Confidence", 0.1, 0.9, 0.35, 0.05, key="up_conf")
        with cb:
            grid_rows = st.slider("Slot rows", 1, 8, 3, key="up_rows")
        with cc:
            grid_cols = st.slider("Slot cols", 1, 12, 5, key="up_cols")
    
    uploaded = st.file_uploader(
        "Upload a parking lot image",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded and model_ok:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        with st.spinner("Analyzing image…"):
            boxes = detect_vehicles(model, image_rgb, conf=conf_thresh)
            h, w = image_rgb.shape[:2]
            slots = build_slots(h, w, grid_rows, grid_cols)
            annotated, occupancy = analyze(image_rgb, slots, boxes)
        
        total = len(slots)
        taken = sum(1 for s in occupancy if s["occupied"])
        free = total - taken
        
        # Display results
        render_stat_cards(total, free, taken, len(boxes))
        
        st.subheader("Annotated View")
        st.image(annotated, use_container_width=True)
        
        st.subheader("Slot Map")
        render_slot_map(occupancy)
        
        # Save to history
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "file": uploaded.name,
            "total": total,
            "free": free,
            "taken": taken,
            "vehicles": len(boxes),
        })
    
    else:
        st.info("📸 Upload an image to analyze parking occupancy")


# ════════════════════════════════════════════════════════════════════
# TAB 3: HISTORY
# ════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("📊 Detection History")
    
    if st.session_state.history:
        for idx, h in enumerate(reversed(st.session_state.history)):
            pct = int(h["free"] / h["total"] * 100) if h["total"] else 0
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{h['file']}**")
                st.caption(f"{h['date']} {h['time']}")
            
            with col2:
                st.metric("Free", f"{pct}%")
            
            with col3:
                st.write(
                    f"Slots: {h['total']} | Free: {h['free']} | "
                    f"Taken: {h['taken']} | Vehicles: {h['vehicles']}"
                )
            
            st.divider()
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    
    else:
        st.info("No detection history yet")


# ════════════════════════════════════════════════════════════════════
# TAB 4: HOW IT WORKS
# ════════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("ℹ️ How It Works")
    
    st.markdown("""
    ### 📹 Live Camera Mode
    Click **▶ Start Camera** to open your laptop webcam. YOLOv8 detects vehicles in real-time 
    and marks parking slots as:
    - 🟢 **GREEN** = Free (available)
    - 🔴 **RED** = Occupied
    
    Click **📸 Snapshot** to save the current state to history.
    
    ### 🖼️ Image Upload
    Upload a parking lot photo and the AI will:
    1. Detect all vehicles (cars, bikes, buses, trucks)
    2. Divide the image into a grid of parking slots
    3. Check if each slot is occupied or free
    
    ### 🔍 Detection Algorithm
    1. **Vehicle Detection** - YOLOv8 finds all vehicles
    2. **Slot Grid** - Frame divided into rows × columns
    3. **Occupancy Check** - If vehicle overlaps >25% of slot, mark as occupied
    
    ### 🎯 Custom Model
    Place a fine-tuned `parking_detector.pt` in the `model_data/` folder 
    for better parking-specific detection.
    
    ### 📊 Statistics
    The app tracks:
    - Total parking slots
    - Available spaces
    - Occupied spaces
    - Detection history
    """)
    
    st.markdown("---")
    st.markdown("**Made with** 🅿️ YOLOv8 + Streamlit")