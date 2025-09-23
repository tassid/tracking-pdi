#!/usr/bin/env python3
# tracking.py
# Uso:
#   python tracking.py --video caminho.mp4 --out saida.mp4
#   python tracking.py --url "https://www.youtube.com/watch?v=G7h5fix9Ny4" --out saida.mp4
# Controles: [r] re-selecionar ROIs • [p] pausar • [q] sair

import argparse, os, sys, tempfile, time, uuid, subprocess
from pathlib import Path

import cv2
import numpy as np

# ---------- Utils ----------

def create_tracker(tracker_name="CSRT"):
    """Cria um tracker compatível com diferentes versões do OpenCV."""
    name = tracker_name.upper()
    legacy = getattr(cv2, "legacy", None)

    def legacy_make(attr):
        return getattr(legacy, attr)() if legacy is not None and hasattr(legacy, attr) else None

    if name == "CSRT":
        return legacy_make("TrackerCSRT_create") or cv2.TrackerCSRT_create()
    if name == "KCF":
        return legacy_make("TrackerKCF_create") or cv2.TrackerKCF_create()
    if name == "MOSSE":
        return legacy_make("TrackerMOSSE_create") or cv2.TrackerMOSSE_create()
    if name == "MIL":
        return legacy_make("TrackerMIL_create") or cv2.TrackerMIL_create()
    if name == "TLD":
        return legacy_make("TrackerTLD_create") or cv2.TrackerTLD_create()
    if name == "MEDIANFLOW":
        return legacy_make("TrackerMedianFlow_create") or cv2.TrackerMedianFlow_create()
    if name == "BOOSTING":
        return legacy_make("TrackerBoosting_create") or cv2.TrackerBoosting_create()
    raise ValueError(f"Tracker desconhecido: {tracker_name}")

def new_multitracker():
    """Cria um MultiTracker compatível com versões novas/antigas."""
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "MultiTracker_create"):
        return cv2.legacy.MultiTracker_create()
    if hasattr(cv2, "MultiTracker_create"):
        return cv2.MultiTracker_create()
    # Fallback simples: estrutura própria
    return None

def add_to_multitracker(mt, tracker, frame, box):
    if mt is None:
        return tracker.init(frame, box)
    return mt.add(tracker, frame, box)

def update_multitracker(mt, frame):
    if mt is None:
        return False, []
    return mt.update(frame)

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

def download_youtube(url, out_dir):
    """
    Tenta baixar com yt-dlp para .mp4. Requer yt-dlp instalado.
    Retorna caminho do arquivo ou lança exceção.
    """
    out = Path(out_dir) / f"yt_{uuid.uuid4().hex}.mp4"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-o", str(out),
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        url
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Tenta binário 'yt-dlp' se módulo falhar
        cmd = ["yt-dlp", "-o", str(out), "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", url]
        subprocess.run(cmd, check=True)
    if not out.exists() or out.stat().st_size < 1024:
        raise RuntimeError("Falha ao baixar vídeo do YouTube (arquivo inválido).")
    return str(out)

# ---------- Lógica principal ----------

def select_rois_interactively(frame, win_name="Selecione ROIs (ENTER para confirmar)"):
    cloneshow = frame.copy()
    cv2.putText(cloneshow, "Selecione multiplas ROIs. ENTER para confirmar, ESC para cancelar.",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(cloneshow, "Selecione multiplas ROIs. ENTER para confirmar, ESC para cancelar.",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow(win_name, cloneshow)
    cv2.waitKey(300)
    rois = cv2.selectROIs(win_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win_name)
    return rois

def main():
    ap = argparse.ArgumentParser(description="Tracking multiobjeto com OpenCV (CSRT/KCF/MOSSE...).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Caminho do arquivo de vídeo local.")
    src.add_argument("--url", type=str, help="URL do YouTube (requer yt-dlp).")
    ap.add_argument("--tracker", type=str, default="CSRT", help="Tipo de tracker: CSRT|KCF|MOSSE|MIL|TLD|MEDIANFLOW|BOOSTING (default: CSRT)")
    ap.add_argument("--out", type=str, help="Arquivo de saída (ex.: saida.mp4). Se omitido, não grava.")
    ap.add_argument("--display", action="store_true", help="Força exibição de janela (habilitado por padrão se houver GUI).")
    ap.add_argument("--no-display", action="store_true", help="Não exibir janela.")
    ap.add_argument("--resize", type=int, default=0, help="Redimensionar largura do vídeo para N px (mantém proporção). 0 = sem resize.")
    ap.add_argument("--label", type=str, default="", help='Rótulo base p/ objetos (ex.: "obj"). IDs serão obj_1, obj_2...')
    ap.add_argument("--seed", type=int, default=42, help="Seed p/ cores aleatórias.")
    args = ap.parse_args()

    # Fonte do vídeo
    temp_dir = None
    video_path = args.video
    if args.url:
        temp_dir = tempfile.mkdtemp(prefix="ytvid_")
        print("Baixando video do YouTube... (yt-dlp)")
        video_path = download_youtube(args.url, temp_dir)
        print(f"Vídeo salvo em: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Não foi possível abrir o vídeo.", file=sys.stderr)
        sys.exit(2)

    # Obter props
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize opcional
    resize_w = args.resize if args.resize and args.resize > 0 else None

    def maybe_resize(frame):
        if resize_w is None:
            return frame
        h, w = frame.shape[:2]
        if w == resize_w:
            return frame
        scale = resize_w / float(w)
        new_size = (resize_w, int(h * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # Saída (VideoWriter)
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # use "avc1" se tiver FFmpeg/H.264 disponível
        out_w = resize_w if resize_w else w_in
        out_h = int(h_in * (out_w / w_in)) if resize_w else h_in
        writer = cv2.VideoWriter(args.out, fourcc, fps_in if fps_in > 1 else 30.0, (out_w, out_h))
        if not writer.isOpened():
            print("Aviso: não foi possível abrir o gravador de vídeo. Prosseguindo sem gravar.", file=sys.stderr)
            writer = None

    # Exibição
    show_win = not args.no_display
    if args.display:
        show_win = True

    # Lê primeiro frame e seleciona ROIs
    ok, frame0 = cap.read()
    if not ok:
        print("Vídeo vazio.", file=sys.stderr)
        sys.exit(3)
    frame0 = maybe_resize(frame0)

    rois = select_rois_interactively(frame0) if show_win else np.array([], dtype=np.int32)
    if rois is None:
        rois = np.array([], dtype=np.int32)

    # Se usuário não selecionou nada, tenta iniciar com uma ROI central (exemplo)
    if len(rois) == 0:
        h, w = frame0.shape[:2]
        w0, h0 = int(w * 0.2), int(h * 0.2)
        x0, y0 = (w - w0)//2, (h - h0)//2
        rois = np.array([[x0, y0, w0, h0]], dtype=np.int32)
        print("Nenhuma ROI selecionada. Usando ROI central por padrão.")

    # Estado do tracking
    multitracker = new_multitracker()
    ids = []
    colors = []
    base = args.label if args.label.strip() else "obj"

    for i, (x, y, w, h) in enumerate(rois):
        trk = create_tracker(args.tracker)
        add_to_multitracker(multitracker, trk, frame0, tuple(map(float, (x, y, w, h))))
        uid = f"{base}_{i+1}"
        ids.append(uid)
        colors.append(random_color(args.seed + i))

    # Loop principal
    win = "Tracking (q: sair, p: pausar, r: reselecionar)"
    paused = False
    prev_t = time.time()

    if show_win:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame = maybe_resize(frame)

            t0 = time.time()
            okup, boxes = update_multitracker(multitracker, frame)
            t1 = time.time()
            fps = human_fps(prev_t, t1)
            prev_t = t1

            if not okup:
                # Se falhar geral, apenas avisa
                cv2.putText(frame, "Tracking falhou (alguns objetos podem ter sido perdidos).",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                boxes = []

            # Desenha resultados
            for i, box in enumerate(boxes):
                label = ids[i] if i < len(ids) else f"{base}_{i+1}"
                color = colors[i] if i < len(colors) else (0, 255, 0)
                draw_fancy_box(frame, box, color, f"{label}")

            # HUD
            cv2.putText(frame, f"FPS: {fps:.1f} | Tracker: {args.tracker.upper()} | Objetos: {len(boxes)}",
                        (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Grava se solicitado
            if writer is not None:
                writer.write(frame)

        if show_win:
            cv2.imshow(win, frame if not paused else frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('r'):
                # Re-seleciona ROIs no frame atual
                paused = True
                rois_new = select_rois_interactively(frame)
                if rois_new is not None and len(rois_new) > 0:
                    multitracker = new_multitracker()
                    ids, colors = [], []
                    for i, (x, y, w, h) in enumerate(rois_new):
                        trk = create_tracker(args.tracker)
                        add_to_multitracker(multitracker, trk, frame, tuple(map(float, (x, y, w, h))))
                        ids.append(f"{base}_{i+1}")
                        colors.append(random_color(args.seed + i))
                paused = False
        else:
            # Sem display, apenas processa até o fim
            pass

    # Libera recursos
    cap.release()
    if writer is not None:
        writer.release()
    if show_win:
        cv2.destroyAllWindows()

    # Limpa temp
    if temp_dir:
        try:
            for p in Path(temp_dir).glob("*"):
                p.unlink(missing_ok=True)
            Path(temp_dir).rmdir()
        except Exception:
            pass

if __name__ == "__main__":
    """
    Requisitos:
      - Python 3.8+
      - opencv-python (ou opencv-contrib-python para mais trackers): pip install opencv-contrib-python
      - (Opcional) yt-dlp para --url: pip install yt-dlp
    Exemplo:
      1) Baixe manualmente com yt-dlp: yt-dlp -f mp4 "URL" -o input.mp4
         python tracking.py --video input.mp4 --out saida.mp4
      2) Baixe automático:
         python tracking.py --url "https://www.youtube.com/watch?v=XXXX" --out saida.mp4
      3) Selecione múltiplas ROIs com o mouse e confirme com ENTER.
    """
    main()
