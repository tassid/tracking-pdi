# tracking-pdi — Auto Face Tracking (OpenCV) + yt-dlp (opcional)

Rastreamento multiobjeto com **auto face tracking** usando OpenCV (Haar Cascade + MultiTracker), com suporte a **download automático de vídeos do YouTube via yt-dlp** (ou uso de arquivo local). IDs estáveis por IoU, re-detecção periódica, seleção manual de ROIs (fallback), e **gravação em MP4** quando o FFmpeg está disponível no PATH.

---

## Requisitos
- **Python 3.8+**
- **OpenCV**: `pip install opencv-contrib-python` (recomendado) ou `pip install opencv-python`
- **yt-dlp** (opcional, só para `--url`): `pip install yt-dlp`
- **FFmpeg no PATH** (necessário para mux/remux MP4 e gravação estável)
  - Windows: `winget install Gyan.FFmpeg` ou `choco install ffmpeg`
  - Verifique: `ffmpeg -version`

> Dica: Em Windows, reabra o PowerShell/Prompt após instalar o FFmpeg para atualizar o PATH.

---

## Instalação rápida
```bash
# (opcional) crie e ative um venv
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# dependências
pip install -U opencv-contrib-python yt-dlp
```

---

## Uso rápido

### 1) Arquivo local (recomendado se tiver o vídeo)
```bash
python tracking.py --video input.mp4 --out saida.mp4 --auto-face
```
- Sem GUI (headless): `--no-display`
- Redimensionar para 960px de largura: `--resize 960`

### 2) YouTube automático (yt-dlp)
```bash
python tracking.py --url "https://www.youtube.com/watch?v=XXXX" --out saida.mp4 --auto-face
```
> Requer `yt-dlp` + **FFmpeg** no PATH. Se houver restrição de região/idade, adicione `--cookies-from-browser chrome` ao yt-dlp (ver troubleshooting).

### 3) Baixar antes com yt-dlp (garante MP4 com vídeo)
```bash
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/22/18" --merge-output-format mp4 --recode-video mp4 "URL" -o input.mp4
python tracking.py --video input.mp4 --out saida.mp4 --auto-face
```

---

## Controles (GUI)
- **[r]**: re-detectar rostos imediatamente
- **[p]**: pausar/continuar
- **[q]**: sair

---

## Parâmetros principais
```text
--auto-face                 Ativa detecção automática de rostos (Haar Cascade)
--redetect-every 15         Re-detecta a cada N frames (default: 15)
--min-face 60               Tamanho mínimo de face (px) para detectar
--tracker CSRT|KCF|...      Tipo do tracker (padrão: CSRT)
--resize 0                  Redimensiona largura do vídeo para N px (0=sem)
--no-display                Desliga a janela de visualização (headless)
--label face                Prefixo de rótulo (IDs estáveis)
--out saida.mp4             Arquivo de saída (MP4 recomendado)
```

> Dicas: Para rostos pequenos, use `--min-face 40`. Se o tracking oscilar, tente `--tracker KCF` e `--redetect-every 30`.

---

## Como funciona (resumo técnico)
- **Detecção**: `haarcascade_frontalface_default.xml` (OpenCV) em escala de cinza + equalização de histograma.
- **Rastreamento**: `cv2.MultiTracker` (CSRT/KCF/MOSSE etc.).
- **IDs estáveis**: associação por **IoU** entre caixas antigas e novas (greedy matching).
- **Re-detecção periódica**: substitui/atualiza trackers para lidar com oclusões, novas faces e drift.
- **YouTube**: `yt-dlp` com seleção de formatos `bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/22/18` e **remux** para MP4 quando necessário.

---

## Troubleshooting

### Só baixou **áudio** (`.m4a`) ou `VideoWriter` falhou
- Instale e valide o **FFmpeg** (`ffmpeg -version`).
- Baixe explicitamente **vídeo+áudio** em MP4:
  ```bash
  yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/22/18" --merge-output-format mp4 --recode-video mp4 "URL" -o input.mp4
  ```
- No script, a função `download_youtube` já tenta remuxar para `.mp4` se necessário.

### Erro de permissão/idade/região no YouTube
- Use cookies do navegador (Chrome):
  ```bash
  yt-dlp --cookies-from-browser chrome -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/22/18" --merge-output-format mp4 --recode-video mp4 "URL"
  ```

### Baixo FPS / performance
- Use `--resize 960` (ou menor), `--tracker KCF`, aumente `--redetect-every`.
- Feche apps pesados; prefira vídeo local a streaming por URL.

---

## Exemplo completo
```bash
# Baixar + processar automaticamente (com auto-face)
python tracking.py --url "https://www.youtube.com/watch?v=G7h5fix9Ny4" --out saida.mp4 --auto-face --resize 960 --redetect-every 20 --min-face 50

# Arquivo local (recomendado)
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/22/18" --merge-output-format mp4 --recode-video mp4 "https://www.youtube.com/watch?v=G7h5fix9Ny4" -o input.mp4
python tracking.py --video input.mp4 --out saida.mp4 --auto-face --tracker KCF --redetect-every 30 --min-face 40
```

---

## Licença
MIT — use à vontade, cite a origem quando possível.
