
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
