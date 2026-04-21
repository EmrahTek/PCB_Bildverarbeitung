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
- Kucuk component'ler ROI icinde class-specific preprocessing, template/edge matching ve local visibility skoru ile aranir.
- Template skoru dusuk ama board/warp/ROI kaniti guvenilir ise class-specific layout ROI fallback'i kullanilir.
- Live modlarda board pozu/homography stabilize edilir, component'ler kanonik board uzayinda kilitlenir ve acquire/keep hysteresis ile flicker azaltilir.

Detector mantigi kamera markasina bagli degildir. Ayni klasik-CV akisi image, video, webcam, IDS ve ileride Raspberry Pi 5 + Pi AI Camera icin kullanilacak sekilde tasarlanmistir.

## Kurulum

```bash
cd /home/emrahtek/Schreibtisch/CodeLab/PCB_Bauteilerkennung
PYTHONPATH=. .venv/bin/python -m pytest -q
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

Tek IDS test resmi:

```bash
.venv/bin/python main.py \
  --source image \
  --image-path pcb_template_tools/test_images/IDS_Kamera.bmp \
  --headless \
  --debug \
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

Once kamera acilisini test et:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-device 0 \
  --camera-backend any \
  --camera-open-check \
  --debug
```

Canli calistirma:

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

Linux/V4L2 ile:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-device 0 \
  --camera-backend v4l2 \
  --debug \
  --width 1280 \
  --height 720 \
  --proc-resize-width 720
```

Eger FPS cok dusukse:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-device 0 \
  --camera-backend any \
  --debug \
  --width 1280 \
  --height 720 \
  --proc-resize-width 540
```

Eger board cok uzaktaysa veya kucuk gorunuyorsa kameraya biraz yaklastir. Yeni ayarlarda kucuk board icin `0.08` template scale destegi var, ama cok uzak ve bulanikhsa component ROI'leri dogru oturmaz.

## IDS Kamera Testi

Desteklenen hedef kamera:

- IDS UI-3250CP-M-GL rev.2
- Lens: SV-1614H

Kod tarafinda IDS icin ayri `--source ids` profili vardir ve detector kapatilmaz. Akis yine normal GUI pipeline'idir:

```text
IDS frame -> board localization -> canonical warp -> component detection -> GUI overlay
```

`--source ids` detector'u kapatmadan normal GUI pipeline'ini calistirir. Elle verilen `--camera-device 0` veya `--camera-device /dev/videoX` hedefleri kullanici tercihi kabul edilir: Linux cihaz adi IDS/uEye gibi gorunmese bile acilmaya calisilir, sadece WARNING loglanir. Yalnizca OpenCV cihazi gercekten acamazsa hata alirsiniz.

Dogru `/dev/videoX` cihazini bulmak icin:

```bash
.venv/bin/python main.py --list-video-devices
```

Once kamera acilisini test et:

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

Canli IDS calistirma:

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

`--camera-device 0` de kullanilabilir:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device 0 \
  --camera-backend auto \
  --debug \
  --width 1600 \
  --height 1200 \
  --proc-resize-width 960 \
  --disable-mjpg
```

Eger IDS kamera gercekten `/dev/video2` ise:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device /dev/video2 \
  --camera-backend v4l2 \
  --debug \
  --width 1600 \
  --height 1200 \
  --proc-resize-width 960 \
  --disable-mjpg
```

Eger IDS kamera V4L2 olarak gorunmuyorsa ve IDS Software Suite/uEye SDK + `pyueye` kuruluysa:

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

OpenCV uEye backend'i sadece OpenCV runtime'iniz gercekten `CAP_UEYE` backend'ini destekliyorsa kullanilabilir. Mevcut test runtime'inda `cv.videoio_registry.hasBackend(cv.CAP_UEYE)` false donuyor; bu durumda su komut temiz bir hata mesaji ile durur:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device 0 \
  --camera-backend ueye \
  --camera-open-check \
  --debug
```

GStreamer pipeline gerekirse `--camera-device` string olarak verilebilir:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device "v4l2src device=/dev/video0 ! video/x-raw,width=1600,height=1200,framerate=30/1 ! videoconvert ! appsink" \
  --camera-backend gstreamer \
  --debug \
  --proc-resize-width 960 \
  --disable-mjpg
```

IDS notlari:

- UI-3250CP-M-GL global shutter oldugu icin motion blur webcam'e gore daha az olmali; yine de exposure cok uzunsa component ROI matching zayiflar.
- SV-1614H ile once board'u frame icinde orta-buyuk boyutta tut. Board cok kucukse warp dogru olsa bile RESET/JST guvenilir olmaz.
- IDS SDK/uEye daemon kurulu olsa bile OpenCV build'inizde uEye video backend aktif degilse `--camera-backend ueye` calismaz.
- `--camera-backend pyueye` icin IDS Software Suite/uEye SDK ve Python `pyueye` paketi gerekir. Paket yoksa komut net bir hata ile durur.
- `--camera-device 0` ve `--camera-index 0` OpenCV'ye integer index `0` olarak verilir. `--camera-device /dev/video0` da V4L2 kullaniminda integer index `0` olarak yorumlanir; boylece OpenCV'nin "capture by name" hatasina dusulmez.
- `--source ids` elle secilen IDS/uEye gibi gorunmeyen V4L2 cihazlarini artik reddetmez; warning loglar ve acmayi dener.
- `--ids-allow-unverified-opencv` geriye donuk uyumluluk icin kaldi, ama elle secilen OpenCV hedefleri zaten warning ile acilmaya calisilir.
- GStreamer gibi pipeline kullaniminda `--camera-device "v4l2src ... ! appsink"` string olarak kalir ve `--camera-backend gstreamer` ile denenir.
- `source_profiles.ids` ayarlari webcam'den biraz farkli exposure/kontrast varsayimi ile gelir.

Backend notlari:

- `--camera-backend auto`: webcam icin `CAP_ANY`; IDS icin OpenCV `v4l2,any` denenir. Elle cihaz verilirse o hedef acilmaya calisilir.
- `--camera-backend any`: OpenCV'nin varsayilan backend secimine birakir.
- `--camera-backend v4l2`: Linux video cihazlari icin tercih edilir.
- `--camera-backend pyueye`: IDS uEye SDK Python yolu; V4L2/OpenCV cihazina gerek duymaz.
- `--camera-backend ueye`: yalnizca OpenCV runtime'iniz uEye backend'ini destekliyorsa kullanilir; aksi halde net hata verir.

Kamera acilis sorunu giderme:

```bash
ls /dev/video*
.venv/bin/python main.py --list-video-devices
.venv/bin/python main.py --source webcam --camera-device 0 --camera-backend any --camera-open-check --debug
.venv/bin/python main.py --source webcam --camera-device /dev/video0 --camera-backend v4l2 --camera-open-check --debug
.venv/bin/python main.py --source ids --camera-device /dev/video0 --camera-backend auto --camera-open-check --debug --disable-mjpg
.venv/bin/python main.py --source ids --camera-device /dev/video0 --camera-backend auto --camera-open-check --save-first-frame /tmp/ids-first-frame.png --debug --disable-mjpg
.venv/bin/python main.py --source ids --camera-device 0 --camera-backend pyueye --camera-open-check --debug
```

Loglarda IDS icin `IDS using OpenCV/V4L2 path`, `ids-opencv`, `verified_ids=True/False`, `IDS using pyueye path` veya hata mesaji gorunur. `verified_ids=False` artik bloklayici degildir; sadece secilen cihazin Linux adinin IDS/uEye gibi gorunmedigini soyler. Eger hata `can't open camera by index` ise hedef artik dogru integer index olarak gidiyor demektir; cihaz izinleri, indeks, baska uygulamanin kamerayi kullanmasi veya driver/IDS ayarlari kontrol edilmelidir.

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
- `board.min_pcb_structure_score`: pin/header benzeri ic yapi, edge dagilimi ve sol/sag PCB yapisini kontrol eder.
- `board.min_canonical_structure_score`: canonical warp icinde beklenen sol ESP32 / sag konnektor yapisinin minimum kanitidir.
- `board.min_edge_grid_score`: monitor/duvar/kasa gibi buyuk ama ic yapisi zayif dikdortgenleri elemek icin edge dagilimini kontrol eder.
- `board.min_tightness_score`: board warp'inin PCB'yi ne kadar sikica sardigini kontrol eder.
- `board.max_skin_ratio`: el/yuz bolgelerinden gelen false board adaylarini azaltir.
- `board.verify_resize_width`: board verify hizini belirler. Daha kucuk deger hizli, ama biraz daha az hassastir.
- `components.*.layout_fallback_score`: board guvenilir ama template match zayifsa sabit layout kutusunun skorudur.
- `components.*.preprocess_mode`: ROI icin class-specific local contrast/edge enhancement secimidir.
- `components.*.min_visibility_score`: ROI icindeki lokal gorunurluk kanitini kontrol eder.
- `components.*.warp_quality_weight`: warp kalitesi cok iyiyse ve ROI kaniti de varsa component skoruna kucuk bir destek verir.
- `components.*.keep_score_threshold`: live tracking sirasinda onceki component kilidini korumak icin gereken daha dusuk keep esigidir.
- `components.*.keep_min_visibility_score`: locked local search veya persistence icin gereken minimum ROI gorunurlugudur.
- `components.*.local_search_expansion`: onceki kanonik component kutusu etrafindaki local search penceresinin buyuklugudur.
- `components.*.track_max_missing`: component'in kac kare dusuk kanit ile kisa sure korunabilecegini belirler.
- `components.*.position_prior_weight`: template/edge skoruna layout veya onceki track pozisyonundan gelen kucuk, kaynak-bagimsiz destek verir.
- `tracking.board_smoothing_alpha`: live modda yeni board pozu ile onceki guvenilir board pozunun karisim oranidir.
- `tracking.board_smoothing_min_quality`: board pozu stabilize edilmeden once gereken minimum warp kalitesidir.
- `tracking.board_smoothing_max_shift`: ani buyuk hareketlerde smoothing'i kapatip yeni pozu oldugu gibi kullanmak icin limitdir.
- `source_profiles.video`, `source_profiles.webcam` ve `source_profiles.ids`: live kullanim icin daha hafif/uygun ayarlari override eder.

Live stabilization kaynak-bagimsizdir:

- Board smoothing sadece hem onceki hem yeni warp kaliteli ve hareket kucukse uygulanir.
- Component locking, frame uzayinda degil kanonik board uzayinda yapilir; bu nedenle IDS, webcam, video ve Pi kamera icin ayni mantik calisir.
- ESP32 ve USB daha esnek tutulur; JST daha fazla local visibility ister; RESET_BUTTON en guclu pozisyon onceligi ve temporal persistence kullanir.

## Raspberry Pi 5 Icin

Baslangic stratejisi:

- `--proc-resize-width 540` veya `720` ile basla.
- `source_profiles.webcam.board.verify_resize_width` degerini `160-220` araliginda tut.
- `source_profiles.webcam.board.max_reference_templates` degerini `4-5` araliginda tut.
- Kamera sabit ise board tracking daha stabil olur; elde tutulan board icin daha fazla isik ve daha az motion blur gerekir.

GPU zorunlu degil. Bu pipeline OpenCV CPU uzerinde calisacak sekilde tasarlandi. Pi tarafinda asil kazanc, dogru resize, az referans, iyi isik ve sabit kamera/board mesafesinden gelir.
