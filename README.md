# PCB Component Detection

Bu proje, FireBeetle / ESP32 tabanli bir PCB uzerindeki ana parcalari klasik
goruntu isleme ile tespit eder.

Tespit edilen etiketler:

- `BOARD`
- `ESP32`
- `USB_PORT`
- `JST_CONNECTOR`
- `RESET_BUTTON`

Ana kullanim hedefi Raspberry Pi 5 + Sony/Pi camera ile canli goruntude
stabil component detection calistirmaktir. iPhone ile hazirlanmis eski template
bankasi korunur; Pi camera icin ayri template bankasi kullanilir.

## Kisa Ozet

Pipeline su sekilde calisir:

1. Kamera veya resimden frame alinir.
2. Once buyuk PCB board bulunur.
3. Board `900 x 460` kanonik gorunume warp edilir.
4. Component'ler board uzerindeki beklenen ROI alanlarinda aranir.
5. Canli goruntude board pozu ve component kutulari takip edilerek flicker azaltilir.
6. Sonuc OpenCV GUI uzerine renkli kutularla cizilir.

Kanonik board yonu her zaman aynidir:

- ESP32 / metal modul solda
- USB-C ve JST sagda
- board uzun kenari yatay

## Proje Klasorleri

```text
config/default.yaml                         Ana detector ve source ayarlari
main.py                                     Uygulama giris noktasi
src/                                       Kamera, pipeline, detection ve render kodu
tests/                                     Unit/smoke testler
logs/app.log                               Runtime log dosyasi
pcb_template_tools/tools/warp_and_rank_boards.py
pcb_template_tools/tools/extract_templates.py
pcb_template_tools/data/pcb_iphone_raw     iPhone raw fotograflari
pcb_template_tools/data/raw_pi             Pi camera raw fotograflari
pcb_template_tools/data/preparation_output iPhone warp/mask/preview ciktilari
pcb_template_tools/data/preparation_output_pi
pcb_template_tools/data/generated_templates
pcb_template_tools/data/generated_templates_pi
```

Template ayrimi:

- Default/image/webcam/video/IDS profilinde iPhone bankasi kullanilir:
  `pcb_template_tools/data/generated_templates`
- `--source picamera` profilinde Pi bankasi kullanilir:
  `pcb_template_tools/data/generated_templates_pi`

Bu ayrim `config/default.yaml` icindeki `source_profiles.picamera` bolumunden
gelir.

## Kurulum

Bu makinedeki proje yolu:

```bash
cd /home/emrahtek/codelab/PCB_Bildverarbeitung
```

Normal Linux/PC ortami icin sanal ortam:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Kodun import edilebilir oldugunu hizli kontrol:

```bash
.venv/bin/python -m compileall main.py src tests
```

Testler icin:

```bash
.venv/bin/python -m pytest -q
```

Not: Raspberry Pi tarafinda `picamera2` apt paketi olarak geldiginden Pi camera
komutlarinda genellikle `.venv/bin/python` yerine `/usr/bin/python3` kullanmak
daha sorunsuzdur.

## Raspberry Pi Camera Canli Calistirma

Pi tarafinda gerekli apt paketleri:

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-yaml
```

Kamerayi ve ilk frame'i test et:

```bash
cd /home/emrahtek/codelab/PCB_Bildverarbeitung

PYTHONPATH=. /usr/bin/python3 main.py \
  --source picamera \
  --camera-index 0 \
  --camera-open-check \
  --save-first-frame /tmp/picamera-first-frame.png \
  --debug \
  --width 1280 \
  --height 720 \
  --camera-fps 30
```

Canli Pi camera detection icin ana komut:

```bash
cd /home/emrahtek/codelab/PCB_Bildverarbeitung

PYTHONPATH=. /usr/bin/python3 main.py \
  --source picamera \
  --camera-index 0 \
  --debug \
  --width 1280 \
  --height 720 \
  --camera-fps 30 \
  --proc-resize-width 720
```

`--debug`, kutularin uzerinde skor/oran yazilarini gosterir ve detay log uretir.
Bu proje icin en stabil Pi camera modu budur; kalibrasyon ve kontrol yaparken
bu komutla calistir.

Skor yazilari gerekmediginde `--debug` kaldirilabilir. Detection ayarlari ayni
kalir, sadece ekrandaki skor/oran yazilari gizlenir:

```bash
PYTHONPATH=. /usr/bin/python3 main.py \
  --source picamera \
  --camera-index 0 \
  --width 1280 \
  --height 720 \
  --camera-fps 30 \
  --proc-resize-width 720
```

Not: `--proc-resize-width 720` Pi camera icin secilen stabil ayardir. Daha dusuk
degerler FPS'i artirabilir ama RESET_BUTTON gibi kucuk komponentlerde kutu
hassasiyetini bozabilir.

GUI penceresinde cikmak icin `q` tusuna bas.

## Tek Resim, Klasor ve Video Komutlari

Tek resim GUI:

```bash
.venv/bin/python main.py \
  --source image \
  --image-path pcb_template_tools/test_images/IMG_9688.JPG \
  --debug \
  --loop \
  --wait-ms 30 \
  --proc-resize-width 960
```

Klasordeki resimleri headless test et:

```bash
.venv/bin/python main.py \
  --source images \
  --images-dir pcb_template_tools/test_images \
  --headless \
  --debug \
  --wait-ms 1 \
  --proc-resize-width 960
```

Klasordeki resimleri GUI ile sirayla goster:

```bash
.venv/bin/python main.py \
  --source images \
  --images-dir pcb_template_tools/test_images \
  --debug \
  --wait-ms 1500 \
  --proc-resize-width 960
```

Video GUI:

```bash
.venv/bin/python main.py \
  --source video \
  --video-path pcb_template_tools/test_video/WIN_20260420_11_54_52_Pro.mp4 \
  --debug \
  --video-resize-width 720 \
  --proc-resize-width 720
```

Video headless hizli test:

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

## Webcam ve IDS Komutlari

Video cihazlarini listele:

```bash
.venv/bin/python main.py --list-video-devices
```

Webcam acilis testi:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-device 0 \
  --camera-backend any \
  --camera-open-check \
  --debug
```

Webcam canli detection:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-device 0 \
  --camera-backend any \
  --debug \
  --width 1280 \
  --height 720 \
  --proc-resize-width 720
```

IDS kamera acilis testi:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device /dev/video0 \
  --camera-backend auto \
  --camera-open-check \
  --debug \
  --width 1600 \
  --height 1200 \
  --disable-mjpg
```

IDS canli detection:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device /dev/video0 \
  --camera-backend auto \
  --debug \
  --width 1600 \
  --height 1200 \
  --proc-resize-width 960 \
  --disable-mjpg
```

IDS uEye SDK / pyueye yolu gerekiyorsa:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device 0 \
  --camera-backend pyueye \
  --debug \
  --width 1600 \
  --height 1200 \
  --proc-resize-width 960 \
  --disable-mjpg
```

## Template Hazirlama Akisi

Template hazirlama iki adimlidir:

1. Raw board fotograflarini kanonik board gorunumune warp et.
2. En iyi warped board'lar uzerinden component ROI/template bankasini cikar.

### Pi Camera Template Bankasini Yeniden Uretmek

Raw Pi fotograflari buraya koy:

```text
pcb_template_tools/data/raw_pi
```

Warp ve kalite siralama:

```bash
.venv/bin/python pcb_template_tools/tools/warp_and_rank_boards.py \
  --input-dir pcb_template_tools/data/raw_pi \
  --output-dir pcb_template_tools/data/preparation_output_pi
```

Component template cikarma:

```bash
.venv/bin/python pcb_template_tools/tools/extract_templates.py \
  --report pcb_template_tools/data/preparation_output_pi/board_quality_report.json \
  --top-k 4 \
  --output-dir pcb_template_tools/data/generated_templates_pi
```

ROI pencereleri acildiginda sirasiyla su kutulari sec:

1. `esp32`
2. `usb_port`
3. `jst_connector`
4. `reset_button`

Kutuyu cizdikten sonra `ENTER` veya `SPACE` ile onayla. Yanlis cizimde `c`
ile tekrar secim yapabilirsin.

Daha once secilmis ROI'leri tekrar kullanmak icin:

```bash
.venv/bin/python pcb_template_tools/tools/extract_templates.py \
  --report pcb_template_tools/data/preparation_output_pi/board_quality_report.json \
  --top-k 4 \
  --output-dir pcb_template_tools/data/generated_templates_pi \
  --roi-file pcb_template_tools/data/generated_templates_pi/component_rois.json
```

### iPhone Template Bankasini Yeniden Uretmek

iPhone raw fotograflari buradadir:

```text
pcb_template_tools/data/pcb_iphone_raw
```

Warp:

```bash
.venv/bin/python pcb_template_tools/tools/warp_and_rank_boards.py \
  --input-dir pcb_template_tools/data/pcb_iphone_raw \
  --output-dir pcb_template_tools/data/preparation_output
```

Template cikarma:

```bash
.venv/bin/python pcb_template_tools/tools/extract_templates.py \
  --report pcb_template_tools/data/preparation_output/board_quality_report.json \
  --top-k 3 \
  --output-dir pcb_template_tools/data/generated_templates \
  --roi-file pcb_template_tools/data/generated_templates/component_rois.json
```

## Log Kontrolu

Son log satirlarini gormek:

```bash
tail -n 80 logs/app.log
```

Canli calismada basarili bir debug satirinda tipik olarak su etiketler gorulur:

```text
labels=BOARD, ESP32, JST_CONNECTOR, RESET_BUTTON, USB_PORT
```

Pi camera loglarinda `source=picamera:0` ve `labels=...` satirlari runtime'in
dogru kaynakla calistigini gosterir.

## Onemli Ayarlar

Ana ayar dosyasi:

```text
config/default.yaml
```

Sik kullanilan ayarlar:

- `runtime.processing_width`: varsayilan islem genisligi.
- `source_profiles.picamera`: Pi camera icin template, board ve component override'lari.
- `source_profiles.picamera.templates`: Pi template bankasinin yollarini belirler.
- `tracking.board_bbox_pad_right`: sadece goruntulenen `BOARD` kutusunun sag kenarini acar.
- `components.USB_PORT.output_bbox_pad_right`: USB kutusunun sag kenarini cikista buyutur.
- `components.JST_CONNECTOR.output_bbox_pad_right`: JST kutusunun sag kenarini cikista buyutur.
- `components.*.layout_anchor`: board ve ROI kaniti guvenliyse component'i beklenen layout ROI'sine sabitler.
- `components.*.layout_fallback_score`: layout fallback kullanildiginda gosterilecek skor.
- `components.*.search_roi_expansion`: component arama alanini genisletir.
- `components.*.layout_roi_left_trim`: layout kutusunun sol tarafini trim eder; USB/JST kutusunu saga toplamak icin kullanilir.

Pi profilinde son canli ayar, USB ve JST'nin sag dis kenarlarini kutu icinde
tutacak sekilde yapilmistir. Bu ayar goruntu overlay'ini iyilestirir; template
matching mantigini agresif sekilde degistirmez.

## Sorun Giderme

Picamera2 venv icinde bulunamiyorsa:

```bash
PYTHONPATH=. /usr/bin/python3 main.py --source picamera --camera-open-check --debug
```

Kamera acilmiyorsa:

```bash
ls /dev/video*
.venv/bin/python main.py --list-video-devices
```

Pi camera ilk frame kaydetme:

```bash
PYTHONPATH=. /usr/bin/python3 main.py \
  --source picamera \
  --camera-index 0 \
  --camera-open-check \
  --save-first-frame /tmp/picamera-first-frame.png \
  --debug \
  --width 1280 \
  --height 720 \
  --camera-fps 30
```

Detection titriyorsa:

- Daha sabit isik kullan.
- Board'u frame icinde orta-buyuk tut.
- Kamera ile board arasindaki mesafeyi sabit tut.
- Pi camera icin `--proc-resize-width 720` kullan; daha dusuk degerler kucuk
  RESET_BUTTON kutusunu bozabilir.
- Pi camera icin yeni raw fotograflar cekip `generated_templates_pi` bankasini yenile.

## Gelistirici Kontrol Komutlari

Syntax/import kontrolu:

```bash
python3 -m compileall main.py src tests
```

Pytest kuruluysa:

```bash
python3 -m pytest -q
```

Kisa headless goruntu kontrolu icin:

```bash
.venv/bin/python main.py \
  --source images \
  --images-dir pcb_template_tools/test_images \
  --headless \
  --debug \
  --wait-ms 1 \
  --proc-resize-width 960
```
