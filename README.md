# PCB Bauteilerkennung

Python + OpenCV ile FireBeetle/ESP32 tabanli PCB uzerinde su parcalari tespit eder:

- `BOARD`
- `ESP32`
- `USB_PORT`
- `JST_CONNECTOR`
- `RESET_BUTTON`

Proje klasik goruntu isleme icin tasarlandi ve Raspberry Pi 5 / Pi AI Camera tarafina tasinabilecek sekilde hafif tutuldu. Ana yaklasim hybrid:

- Once PCB bulunur ve kanonik `900 x 460` board gorunumune warp edilir.
- Board dogrulama, hiz icin kucultulmus verify kopyasinda yapilir.
- Kucuk component'ler once template matching ile aranir.
- Template skoru dusuk ama board guvenilir ise sabit PCB layout ROI fallback'i kullanilir.

## Kurulum

```bash
cd /home/emrahtek/Schreibtisch/CodeLab/PCB_Bauteilerkennung
.venv/bin/python -m pytest -q
```

Bu projede sisteminde `python` komutu olmayabilir; bu yuzden komutlarda `.venv/bin/python` kullanmak daha guvenli.

## Template Veri Seti

Aktif template bankasi `pcb_template_tools` altindan okunur:

```text
pcb_template_tools/data/preparation_output/warped
pcb_template_tools/data/generated_templates
pcb_template_tools/data/generated_templates/templates_metadata.json
```

Kanonik yon kuralimiz:

- board uzun kenar yatay
- ESP32 / metal modul solda
- USB-C ve JST sagda

Raw telefon fotograflari portre veya yatay olabilir; onemli olan warp sonucunun bu kurala uymasi.

## Image Test

Headless hizli regression:

```bash
.venv/bin/python main.py \
  --source images \
  --images-dir pcb_template_tools/test_images \
  --headless \
  --debug \
  --wait-ms 1 \
  --proc-resize-width 960
```

GUI ile tek tek gormek icin:

```bash
.venv/bin/python main.py \
  --source images \
  --images-dir pcb_template_tools/test_images \
  --debug \
  --wait-ms 1500 \
  --proc-resize-width 960
```

Tek resim:

```bash
.venv/bin/python main.py \
  --source image \
  --image-path pcb_template_tools/test_images/IMG_9688.JPG \
  --debug \
  --loop \
  --wait-ms 30 \
  --proc-resize-width 960
```

## Video Test

Detayli kontrol:

```bash
.venv/bin/python main.py \
  --source video \
  --video-path pcb_template_tools/test_video/WIN_20260420_11_54_52_Pro.mp4 \
  --debug \
  --video-resize-width 720 \
  --proc-resize-width 720
```

Daha hizli test icin frame atlama:

```bash
.venv/bin/python main.py \
  --source video \
  --video-path pcb_template_tools/test_video/WIN_20260420_11_54_52_Pro.mp4 \
  --headless \
  --debug \
  --max-frames 80 \
  --video-resize-width 720 \
  --video-stride 2 \
  --proc-resize-width 720
```

## Live Webcam Test

Baslangic komutu:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-index 0 \
  --debug \
  --width 1280 \
  --height 720 \
  --proc-resize-width 720
```

Eger FPS cok dusukse:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-index 0 \
  --debug \
  --width 1280 \
  --height 720 \
  --proc-resize-width 540
```

Eger board cok uzaktaysa veya kucuk gorunuyorsa kameraya biraz yaklastir. Yeni ayarlarda kucuk board icin `0.08` template scale destegi var, ama cok uzak ve bulanikhsa component ROI'leri dogru oturmaz.

## Log Kontrolu

```bash
tail -n 80 logs/app.log
```

Ornek basarili satir:

```text
labels=BOARD, ESP32, JST_CONNECTOR, RESET_BUTTON, USB_PORT
```

## Tuning Notlari

Ana ayarlar `config/default.yaml` icindedir.

- `board_template.scales`: PCB'nin frame icindeki boyut araligini belirler. Uzak/kucuk board icin `0.08-0.16` kritik.
- `board.min_objectness_score`: TV, yuz, tisort, dolap gibi yanlis board adaylarini elemek icin kullanilir.
- `board.verify_resize_width`: board verify hizini belirler. Daha kucuk deger hizli, ama biraz daha az hassastir.
- `components.*.layout_fallback_score`: board guvenilir ama template match zayifsa sabit layout kutusunun skorudur.
- `source_profiles.video` ve `source_profiles.webcam`: live kullanim icin daha hafif ayarlari override eder.

## Raspberry Pi 5 Icin

Baslangic stratejisi:

- `--proc-resize-width 540` veya `720` ile basla.
- `source_profiles.webcam.board.verify_resize_width` degerini `160-220` araliginda tut.
- `source_profiles.webcam.board.max_reference_templates` degerini `4-5` araliginda tut.
- Kamera sabit ise board tracking daha stabil olur; elde tutulan board icin daha fazla isik ve daha az motion blur gerekir.

GPU zorunlu degil. Bu pipeline OpenCV CPU uzerinde calisacak sekilde tasarlandi. Pi tarafinda asil kazanc, dogru resize, az referans, iyi isik ve sabit kamera/board mesafesinden gelir.
