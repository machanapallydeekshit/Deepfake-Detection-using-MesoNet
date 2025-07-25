#app.py
import streamlit as st
import os
import re
import tempfile
import mediapipe as mp
import cv2
import shutil
import atexit
import stat

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from test import detect_deepfake, load_meso4_model
from blur import process_video as detect_blur
from halo_scan import process_video as halo_scan
from color_scan import process_video as color_scan
from blinkrate import process_video as blink_scan
from mtcnn import MTCNN

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Configure Streamlit page
st.set_page_config(page_title="Deepfake Detection System", layout="centered")

# --- HEADER SECTION ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a video to detect face-swap-based tampering and view detailed results.</div>', unsafe_allow_html=True)


# --- SESSION STATE INITIALIZATION ---
if "processed" not in st.session_state:
    st.session_state.processed = False
if "deepfake_score" not in st.session_state:
    st.session_state.deepfake_score = None
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = ""
if "temp_file" not in st.session_state:
    st.session_state.temp_file = ""
if "anomaly_detection" not in st.session_state:
    st.session_state.anomaly_detection = False

# --- Threshold ---
BLINK_THRESHOLD = 17
BLUR_THRESHOLD = 0.70
HALO_THRESHOLD = 1.5
COLOR_THRESHOLD = 12
FRAME_RATE = 30


tab1, tab2, tab3, tab4= st.tabs(["📤 Upload & Detect", "🧠 Frame Analysis", "📄 Summary","⬇️ Save Report"])
with tab1:
    # --- STEP 1: UPLOAD VIDEO ---
    st.subheader("Upload Your Video")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if video_file:
        # New upload: reset processing
        if video_file.name != st.session_state.uploaded_video_path:
            st.session_state.uploaded_video_path = video_file.name
            st.session_state.processed = False
            st.session_state.log_lines.clear()
            atexit.register(lambda: os.remove(tmp.name) if os.path.exists(tmp.name) else None)
            for key in ["blur_score", "halo_result", "color_result", "blink_score"]:
                st.session_state.pop(key, None)
            for folder in ["gradcam_outputs", "original_frames"]:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder, onerror=remove_readonly)
                    except PermissionError as e:
                        st.warning(f"⚠️ Could not delete folder {folder}: {e}")
                else:
                    print(f"[INFO] Folder '{folder}' does not exist. Skipping.")
        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
        tmp.write(video_file.read())
        tmp.close()
        st.session_state.temp_file = tmp.name
        st.success(f"Uploaded: {video_file.name}")

    # --- STEP 2: PROCESS VIDEO ---
    if video_file:
        st.subheader("Process the Video")
        show_logs = st.checkbox("Show log messages during processing", value=True)
        if st.button("Start Processing") and not st.session_state.processed:
            st.session_state.log_lines.clear()
            st.info("Starting deepfake analysis...")
            def log_callback(msg):
                st.session_state.log_lines.append(msg)
            with st.spinner("Analyzing video. Please wait..."):
                model = load_meso4_model("weights/MesoInception_DF.h5")
                detector = MTCNN()
                with mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ) as face_mesh:
                    score = detect_deepfake(
                        st.session_state.temp_file,
                        model,
                        face_mesh,
                        detector,
                        log_callback=log_callback
                    )
                    if score is None:
                        st.session_state.processed = False
                        st.session_state.deepfake_score = None
                        st.error("❌ No faces were detected in the video. Please upload another one.")
                        st.session_state.anomaly_detection = False
                    else:
                        st.session_state.deepfake_score = score
                        st.session_state.processed = True
                        st.success("✅ Video analyzed successfully!")
                        st.session_state.anomaly_detection = True

with tab2:
    # --- LOG DISPLAY ---
    st.subheader("LOG DETAILS")
    if st.session_state.processed and st.session_state.log_lines:
        with st.expander("📄 Logs", expanded=show_logs):
            for line in st.session_state.log_lines:
                if "Deepfake score" in line:
                    st.markdown(f"**🧠 {line}**")
                elif "Feedback:" in line:
                    fb = line.split("Feedback: ", 1)[-1]
                    st.markdown(f"🧪 **Feedback:** {fb}")
                elif line.startswith("->"):
                    region = line.split("is", 1)[-1].strip()
                    st.markdown(f"🧭 **Most activated region:** {region}")
                else:
                    st.text(line)
    else:
        st.info("No logs available yet. Please start the analysis first.")



with tab3:

    # --- STEP 3: ANOMALY DETECTION  ---
    if st.session_state.processed and st.session_state.anomaly_detection:
        st.subheader("Anomaly Detection ")

        # --- Blink Detection ---
        if "blink_score" not in st.session_state:
            st.session_state.blink_score = blink_scan(st.session_state.temp_file)

        blink_score = st.session_state.blink_score
        if blink_score:
            if blink_score["bpmrate"] < BLINK_THRESHOLD:
                st.write(f"👁️ Blink Result: Blink count - {blink_score['blink_count']}, BPM - {blink_score['bpmrate']:0.2f} bp/min - Suspicious")
            else:
                st.write(f"👁️ Blink Result: Blink count - {blink_score['blink_count']}, BPM - {blink_score['bpmrate']:0.2f} bp/min - Normal")
        else:
            st.warning("Could not process video or not enough facial landmarks.")

        # --- Blur Detection ---
        if "blur_score" not in st.session_state:
            st.session_state.blur_score = detect_blur(st.session_state.temp_file)

        blur_score = st.session_state.blur_score
        if blur_score < BLUR_THRESHOLD:
            st.write(f" 🌫️ Blur Result: Blur Score - {blur_score:0.2f} - BLUR-SUSPECT")
        elif blur_score > BLUR_THRESHOLD:
            st.write(f" 🌫️ Blur Result: Blur Score - {blur_score:0.2f} - BLUR-NORMAL")
        else:
            st.warning("Could not process blur detection due to insufficient data.")

        # --- Halo Detection ---
        if "halo_result" not in st.session_state:
            st.session_state.halo_result = halo_scan(st.session_state.temp_file)

        halo_result = st.session_state.halo_result
        if halo_result is not None:
            if halo_result < HALO_THRESHOLD:
                st.write(f"👼 Halo Detected: Halo Ratio - {halo_result:0.2f} - Halo Normal")
            else:
                st.write(f"👼 Halo Detected: Halo Ratio - {halo_result:0.2f} - Halo Suspect")
        else:
            st.warning("Could not process halo detection due to insufficient data.")

        # --- Color Scan ---
        if "color_result" not in st.session_state:
            st.session_state.color_result = color_scan(st.session_state.temp_file)

        color_results = st.session_state.color_result
        if color_results > COLOR_THRESHOLD:
            st.write(f" 🌈 Color scan: Color Ratio - {color_results:5.1f} - Color Suspect")
        elif color_results < COLOR_THRESHOLD:
            st.write(f" 🌈 Color scan: Color Ratio - {color_results:5.1f} - Color Normal")
        else:
            st.warning("Could not process color detection due to insufficient data.")

                        
    # --- RESULTS AND HEATMAPS ---
    if st.session_state.processed and st.session_state.deepfake_score is not None:
        st.subheader("Deepfake Results ")
        st.write(f"**Deepfake Score:** `{st.session_state.deepfake_score:.4f}`")
        st.subheader("Heatmap Results")
        gradcam_folder = "gradcam_outputs"
        original_folder = "original_frames"
        if os.path.exists(gradcam_folder) and os.path.exists(original_folder):
            heatmap_files = sorted([
                f for f in os.listdir(gradcam_folder)
                if f.endswith(('.jpg', '.png'))
            ], key=lambda x: int(x.split('_')[1]))
            for i in range(0, len(heatmap_files), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(heatmap_files):
                        fname = heatmap_files[idx]
                        num = int(fname.split('_')[1])
                        heat_path = os.path.join(gradcam_folder, fname)
                        orig_name = fname.replace("_annotated", "_original")
                        orig_path = os.path.join(original_folder, orig_name)
                        with col:
                            chk = st.toggle("Toggle Heatmap", value=False, key=f"toggle_{num}")
                            img_path = heat_path if chk else orig_path
                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                                secs = num / 30  # assume 30 FPS
                                mins = int(secs // 60)
                                secs_int = int(secs % 60)
                                st.caption(f"Frame {num} | Timestamp: {mins:02d}:{secs_int:02d}")
                            else:
                                st.warning("Missing image")
        else:
            st.warning("Heatmap or original frame folder not found.")
    else:
        st.info("No results available. Please analyze a video first.")

with tab4:
    def generate_pdf_report():
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Text settings
        textobject = c.beginText(40, height - 40)
        textobject.setFont("Helvetica", 11)
        line_height = 14  # spacing between lines
        bottom_margin = 40  # margin before starting new page

        def add_line(text):
            nonlocal textobject
            nonlocal c
            y = textobject.getY()
            if y <= bottom_margin:
                c.drawText(textobject)
                c.showPage()
                textobject = c.beginText(40, height - 40)
                textobject.setFont("Helvetica", 11)
            textobject.textLine(text)

        # Header
        add_line("Deepfake Detection Report")
        add_line("-" * 50)

        if "deepfake_score" in st.session_state:
            add_line(f"Deepfake Score: {st.session_state.deepfake_score:.4f}")

        if "blink_score" in st.session_state:
            b = st.session_state.blink_score
            result = "Suspicious" if b['bpmrate'] < BLINK_THRESHOLD else "Normal"
            add_line(f"Blink Detection: {b['blink_count']} blinks, {b['bpmrate']:.2f} bpm - {result}")

        if "blur_score" in st.session_state:
            b = st.session_state.blur_score
            if b is not None:
                result = "BLUR-SUSPECT" if b < BLUR_THRESHOLD else "BLUR-NORMAL"
                add_line(f"Blur Detection: {b:.2f} - {result}")
            else:
                add_line("Blur Detection: Not Available")

        if "halo_result" in st.session_state:
            h = st.session_state.halo_result
            if h is not None:
                result = "Halo Suspect" if h >= HALO_THRESHOLD else "Halo Normal"
                add_line(f"Halo Detection: {h:.2f} - {result}")
            else:
                add_line("Halo Detection: Not Available")

        if "color_result" in st.session_state:
            c_val = st.session_state.color_result
            if c_val is not None:
                result = "Color Suspect" if c_val > COLOR_THRESHOLD else "Color Normal"
                add_line(f"Color Detection: {c_val:.2f} - {result}")
            else:
                add_line("Color Detection: Not Available")

        # Logs
        add_line("")
        add_line("Log Messages:")
        add_line("-" * 50)

        if "log_lines" in st.session_state and st.session_state.log_lines:
            for log in st.session_state.log_lines:
                # Wrap each log line to max 90 chars
                wrapped = re.findall('.{1,90}', log)
                for part in wrapped:
                    add_line(part)
        else:
            add_line("No logs available.")

        # Finalize and return
        c.drawText(textobject)
        c.showPage()
        c.save()

        buffer.seek(0)
        return buffer

    if st.session_state.processed:
        pdf = generate_pdf_report()
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf,
            file_name="deepfake_report.pdf",
            mime="application/pdf"
        )
    else:
            st.info("No Results available yet. Please start the analysis first.")




