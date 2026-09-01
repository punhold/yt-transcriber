# 🎙️ YT Transcriber

Descarga audio de YouTube y lo convierte a texto usando Whisper AI (100% offline).

## Requisitos previos

- Python 3.8 o superior
- **FFmpeg** instalado en el sistema

### Instalar FFmpeg

**Windows:**
Descargá desde https://ffmpeg.org/download.html y agregalo al PATH.
O con Chocolatey: `choco install ffmpeg`

**Mac:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install ffmpeg
```

---

## Instalación

```bash
# 1. Instalá las dependencias de Python
pip install -r requirements.txt

# 2. Corré la app
python app.py
```

Luego abrí tu navegador en: **http://localhost:5000**

---

## Uso

1. Pegá URLs de YouTube en el textarea (una por línea, o mezcladas en texto)
2. Elegí el modelo Whisper:
   - `tiny` → muy rápido, menos preciso
   - `base` → ideal para uso general ✓
   - `small` → más preciso, un poco más lento
   - `medium` / `large` → máxima calidad (requieren más RAM y tiempo)
3. Hacé clic en **Iniciar transcripción**
4. Esperá y descargá los `.txt` al finalizar

---

## Estructura de archivos generados

```
downloads/          ← audios temporales (se borran automáticamente)
transcriptions/     ← transcripciones en .txt
  job_123_0_Titulo.txt     (por video)
  job_123_COMPLETO.txt     (todos juntos)
```

---

## Notas

- La primera vez que usés un modelo Whisper se descarga automáticamente.
- El modelo `base` pesa ~140MB, `large` ~2.9GB.
- Funciona con cualquier video público de YouTube.
- No necesita API key ni conexión a servicios externos (salvo para descargar de YT).
