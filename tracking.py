#!/usr/bin/env python3
# tracking.py — Face Auto-Tracking (OpenCV)
# Uso:
#   python tracking.py --video caminho.mp4 --out saida.mp4 --auto-face
#   python tracking.py --url "https://www.youtube.com/watch?v=G7h5fix9Ny4" --auto-face --out saida.mp4
# Controles: [r] re-detectar • [p] pausar • [q] sair

import argparse, os, sys, tempfile, time, uuid, subprocess
from pathlib import Path
import cv2
import numpy as np

# ---------- Utils de Tracker ----------
def create_tracker(tracker_name="CSRT"):
    name = tracker_name.upper()
    legacy = getattr(cv2, "legacy", None)
    def legacy_make(attr):
        return getattr(legacy, attr)() if legacy is not None and hasattr(legacy, attr) else None
    if name == "CSRT":       return legacy_make("TrackerCSRT_create") or cv2.TrackerCSRT_create()
    if name == "KCF":        return legacy_make("TrackerKCF_create") or cv2.TrackerKCF_create()
    if name == "MOSSE":      return legacy_make("TrackerMOSSE_create") or cv2.TrackerMOSSE_create()
    if name == "MIL":        return legacy_make("TrackerMIL_create") or cv2.TrackerMIL_create()
    if name == "TLD":        return legacy_make("TrackerTLD_create") or cv2.TrackerTLD_create()
    if name == "MEDIANFLOW": return legacy_make("TrackerMedianFlow_create") or cv2.TrackerMedianFlow_create()
    if name == "BOOSTING":   return legacy_make("TrackerBoosting_create") or cv2.TrackerBoosting_create()
    raise ValueError(f"Tracker desconhecido: {tracker_name}")

def new_multitracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "MultiTracker_create"):
        return cv2.legacy.MultiTracker_create()
    if hasattr(cv2, "MultiTracker_create"):
        return cv2.MultiTracker_create()
    return None

def add_to_multitracker(mt, tracker, frame, box):
    if mt is None: return tracker.init(frame, box)
    return mt.add(tracker, frame, box)

def update_multitracker(mt, frame):
    if mt is None: return False, []
    return mt.update(frame)

# ---------- Desenho / util ----------
def draw_fancy_box(img, box, color, label=None):
    x, y, w, h = [int(v) for v in box]
    x2, y2 = x + w, y + h
    lw = max(1, int(0.002 * (img.shape[0] + img.shape[1])))
    cv2.rectangle(img, (x, y), (x2, y2), color, lw)
    if label:
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(img, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def random_color(seed=None):
    rng = np.random.default_rng(seed)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))

def human_fps(t0, t1):
    dt = max(1e-6, t1 - t0)
    return 1.0 / dt

# ---------- Download YouTube ----------
def _run(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def download_youtube(url, out_dir):
    out_tpl = str(Path(out_dir) / "yt_%(id)s.%(ext)s")  # deixa o yt-dlp escolher a extensão
    base_flags = ["-o", out_tpl, "-S", "res,ext:mp4:m4a", "-f", "bv*+ba/b", "--no-part", "--no-playlist", "--merge-output-format", "mp4", url]

    # 1) Tenta yt-dlp com cookies do Chrome (ajuda em vídeos age/region)
    try:
        _run(["yt-dlp", "--cookies-from-browser", "chrome", *base_flags])
    except FileNotFoundError:
        # yt-dlp não no PATH → usa módulo
        try:
            _run([sys.executable, "-m", "yt_dlp", "--cookies-from-browser", "chrome", *base_flags])
        except subprocess.CalledProcessError as e:
            # 2) Fallback sem cookies via módulo
            try:
                _run([sys.executable, "-m", "yt_dlp", *base_flags])
            except subprocess.CalledProcessError as e2:
                # Mostra erro do yt-dlp para diagnóstico
                sys.stderr.write((e2.stderr or "") + "\n")
                raise RuntimeError("yt-dlp falhou ao baixar o vídeo.")
    except subprocess.CalledProcessError:
        # 2) Fallback sem cookies com binário
        try:
            _run(["yt-dlp", *base_flags])
        except Exception as e:
            # tenta módulo sem cookies
            try:
                _run([sys.executable, "-m", "yt_dlp", *base_flags])
            except subprocess.CalledProcessError as e2:
                sys.stderr.write((e2.stderr or "") + "\n")
                raise RuntimeError("yt-dlp falhou ao baixar o vídeo.")

    # encontra o arquivo baixado (mp4/mkv/webm) mais recente
    candidates = sorted(Path(out_dir).glob("yt_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("Falha ao baixar vídeo do YouTube (nenhum arquivo gerado).")
    media = candidates[0]

    # se não for .mp4, tenta remuxar para mp4 (requer ffmpeg)
    if media.suffix.lower() != ".mp4":
        mp4_path = media.with_suffix(".mp4")
        try:
            _run(["ffmpeg", "-y", "-i", str(media), "-c", "copy", str(mp4_path)])
            media = mp4_path
        except Exception as e:
            print(f"Aviso: remux para mp4 falhou ({e}); usando {media.name}.", file=sys.stderr)

    if not media.exists() or media.stat().st_size < 1024 * 50:
        raise RuntimeError("Falha ao baixar vídeo do YouTube (arquivo muito pequeno/ inválido).")
    return str(media)

# ---------- Face Detection ----------
def build_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face = cv2.CascadeClassifier(cascade_path)
    if face.empty():
        raise RuntimeError("Não foi possível carregar o Haar Cascade de faces.")
    return face

def detect_faces(frame_bgr, detector, min_size=60):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE, minSize=(min_size, min_size)
    )
    return [tuple(map(float, f)) for f in faces]

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah; bx2, by2 = bx+bw, by+bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = aw*ah + bw*bh - inter + 1e-6
    return inter / union

def match_boxes(prev_boxes, new_boxes, thr=0.3):
    M = len(prev_boxes); N = len(new_boxes)
    if M == 0 or N == 0: return {}, set(range(N)), set(range(M))
    iou_mat = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            iou_mat[i, j] = iou(prev_boxes[i], new_boxes[j])
    matched_prev, matched_new = set(), set()
    pairs = {}
    while True:
        i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[i, j] < thr: break
        if i in matched_prev or j in matched_new:
            iou_mat[i, j] = -1; continue
        pairs[i] = j
        matched_prev.add(i); matched_new.add(j)
        iou_mat[i, :] = -1; iou_mat[:, j] = -1
    new_only = set(range(N)) - matched_new
    prev_only = set(range(M)) - matched_prev
    return pairs, new_only, prev_only

# ---------- Interação manual (fallback) ----------
def select_rois_interactively(frame, win_name="Selecione ROIs (ENTER para confirmar)"):
    cloneshow = frame.copy()
    cv2.putText(cloneshow, "Selecione multiplas ROIs. ENTER confirma, ESC cancela.",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(cloneshow, "Selecione multiplas ROIs. ENTER confirma, ESC cancela.",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow(win_name, cloneshow); cv2.waitKey(300)
    rois = cv2.selectROIs(win_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win_name)
    return rois

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Face auto-tracking com OpenCV (CSRT/KCF/MOSSE...).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Caminho do arquivo de vídeo local.")
    src.add_argument("--url", type=str, help="URL do YouTube (requer yt-dlp).")
    ap.add_argument("--auto-face", action="store_true", help="Ativar detecção automática de rostos.")
    ap.add_argument("--redetect-every", type=int, default=15, help="Re-detectar faces a cada N frames (default: 15).")
    ap.add_argument("--min-face", type=int, default=60, help="Tamanho mínimo da face em pixels (default: 60).")
    ap.add_argument("--tracker", type=str, default="CSRT", help="CSRT|KCF|MOSSE|MIL|TLD|MEDIANFLOW|BOOSTING (default: CSRT)")
    ap.add_argument("--out", type=str, help="Arquivo de saída (ex.: saida.mp4). Se omitido, não grava.")
    ap.add_argument("--display", action="store_true", help="Forçar exibição de janela.")
    ap.add_argument("--no-display", action="store_true", help="Não exibir janela.")
    ap.add_argument("--resize", type=int, default=0, help="Redimensionar largura para N px (0=sem).")
    ap.add_argument("--label", type=str, default="face", help="Rótulo base (default: face).")
    ap.add_argument("--seed", type=int, default=42, help="Seed p/ cores.")
    args = ap.parse_args()

    temp_dir = None
    video_path = args.video
    if args.url:
        temp_dir = tempfile.mkdtemp(prefix="ytvid_")
        print("Baixando video do YouTube... (yt-dlp)")
        video_path = download_youtube(args.url, temp_dir)
        print(f"Vídeo salvo em: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Não foi possível abrir o vídeo.", file=sys.stderr); sys.exit(2)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize_w = args.resize if args.resize and args.resize > 0 else None
    def maybe_resize(frame):
        if resize_w is None: return frame
        h, w = frame.shape[:2]
        if w == resize_w: return frame
        scale = resize_w / float(w)
        return cv2.resize(frame, (resize_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = resize_w if resize_w else w_in
        out_h = int(h_in * (out_w / w_in)) if resize_w else h_in
        writer = cv2.VideoWriter(args.out, fourcc, fps_in if fps_in > 1 else 30.0, (out_w, out_h))
        if not writer.isOpened():
            print("Aviso: não foi possível abrir o gravador de vídeo. Prosseguindo sem gravar.", file=sys.stderr)
            writer = None

    show_win = not args.no_display
    if args.display: show_win = True

    ok, frame0 = cap.read()
    if not ok:
        print("Vídeo vazio.", file=sys.stderr); sys.exit(3)
    frame0 = maybe_resize(frame0)

    # Estado
    multitracker = new_multitracker()
    ids, colors, boxes_prev = [], [], []
    id_counter = 0
    base = args.label.strip() or "face"
    face_detector = build_face_detector() if args.auto_face else None

    # Inicialização (auto-face ou manual)
    if args.auto_face:
        faces = detect_faces(frame0, face_detector, min_size=args.min_face)
        if len(faces) == 0:
            h, w = frame0.shape[:2]; w0, h0 = int(w*0.2), int(h*0.2)
            x0, y0 = (w-w0)//2, (h-h0)//2
            faces = [(float(x0), float(y0), float(w0), float(h0))]
            print("Nenhuma face detectada no 1º frame. Usando ROI central por padrão.")
        for f in faces:
            trk = create_tracker(args.tracker)
            add_to_multitracker(multitracker, trk, frame0, f)
            id_counter += 1
            ids.append(f"{base}_{id_counter}")
            colors.append(random_color(args.seed + id_counter))
        boxes_prev = faces[:]
    else:
        if show_win:
            rois = select_rois_interactively(frame0)
        else:
            rois = np.array([], dtype=np.int32)
        if rois is None or len(rois)==0:
            h, w = frame0.shape[:2]; w0, h0 = int(w*0.2), int(h*0.2); x0, y0 = (w-w0)//2, (h-h0)//2
            rois = np.array([[x0, y0, w0, h0]], dtype=np.int32)
            print("Nenhuma ROI selecionada. Usando ROI central por padrão.")
        for (x,y,w,h) in rois:
            trk = create_tracker(args.tracker)
            add_to_multitracker(multitracker, trk, frame0, tuple(map(float,(x,y,w,h))))
            id_counter += 1
            ids.append(f"{base}_{id_counter}")
            colors.append(random_color(args.seed + id_counter))
        boxes_prev = [tuple(map(float, r)) for r in rois]

    win = "Face Tracking (q: sair, p: pausar, r: re-detectar)"
    paused = False
    prev_t = time.time()
    frame_idx = 0
    if show_win: cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok: break
            frame = maybe_resize(frame)
            frame_idx += 1

            okup, boxes = update_multitracker(multitracker, frame)
            t1 = time.time(); fps = human_fps(prev_t, t1); prev_t = t1
            if not okup: boxes = []

            # Re-detecção periódica (apenas se auto-face)
            if args.auto_face and (frame_idx % max(1, args.redetect_every) == 0 or len(boxes)==0):
                detected = detect_faces(frame, face_detector, min_size=args.min_face)
                pairs, new_only, _prev_only = match_boxes(boxes, detected, thr=0.3)

                # Recria MultiTracker preservando IDs e adiciona novos
                new_ids, new_colors, new_boxes = [], [], []
                for i_prev, j_new in pairs.items():
                    new_boxes.append(detected[j_new])
                    new_ids.append(ids[i_prev] if i_prev < len(ids) else f"{base}_X")
                    new_colors.append(colors[i_prev] if i_prev < len(colors) else (0,255,0))
                for j in sorted(list(new_only)):
                    new_boxes.append(detected[j])
                    id_counter += 1
                    new_ids.append(f"{base}_{id_counter}")
                    new_colors.append(random_color(args.seed + id_counter))

                if len(new_boxes) > 0:
                    multitracker = new_multitracker()
                    for b in new_boxes:
                        trk = create_tracker(args.tracker)
                        add_to_multitracker(multitracker, trk, frame, b)
                    ids, colors = new_ids, new_colors
                    boxes = new_boxes
                boxes_prev = boxes[:]

            for i, box in enumerate(boxes):
                label = ids[i] if i < len(ids) else f"{base}_{i+1}"
                color = colors[i] if i < len(colors) else (0,255,0)
                draw_fancy_box(frame, box, color, label)

            cv2.putText(frame, f"FPS: {fps:.1f} | Tracker: {args.tracker.upper()} | Faces: {len(boxes)}",
                        (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            if writer is not None: writer.write(frame)

        if show_win:
            cv2.imshow(win, frame if not paused else frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): paused = not paused
            elif key == ord('r') and args.auto_face:
                detected = detect_faces(frame, face_detector, min_size=args.min_face)
                if len(detected) > 0:
                    multitracker = new_multitracker()
                    ids, colors = [], []
                    for b in detected:
                        trk = create_tracker(args.tracker)
                        add_to_multitracker(multitracker, trk, frame, b)
                        id_counter += 1
                        ids.append(f"{base}_{id_counter}")
                        colors.append(random_color(args.seed + id_counter))
                    boxes_prev = detected[:]

    cap.release()
    if writer is not None: writer.release()
    if show_win: cv2.destroyAllWindows()
    if temp_dir:
        try:
            for p in Path(temp_dir).glob("*"): p.unlink(missing_ok=True)
            Path(temp_dir).rmdir()
        except Exception: pass

if __name__ == "__main__":
    main()
