import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import sys
from segment_anything import SamPredictor, sam_model_registry

# -------------------------------
# ZOOM FONKSİYONU
# -------------------------------
def zoom_factory(ax, base_scale=1.2):
    def zoom_fun(event):
        # Sadece harita üzerindeyken çalışsın
        if event.inaxes != ax:
            return
        
        # Mevcut eksen limitlerini al
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xdata = event.xdata # Farenin X konumu
        ydata = event.ydata # Farenin Y konumu

        if xdata is None or ydata is None:
            return 

        # Tekerlek yukarı mı aşağı mı?
        if event.button == 'up':
            # Zoom in (Yakınlaş)
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # Zoom out (Uzaklaş)
            scale_factor = base_scale
        else:
            scale_factor = 1

        # Yeni limitleri hesapla (Fare imlecini merkez alarak)
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
        
        # Çizimi güncelle
        ax.figure.canvas.draw_idle()

    # Scroll olayını bağla
    fig = ax.get_figure()
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    return zoom_fun

# -------------------------------
# SEÇİM VE MASKELEME
# -------------------------------
def onselect(eclick, erelease):
    "Fare ile seçim bitince çalışacak fonksiyon"
    global box_coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    
    # Koordinatları sırala (min-max)
    box_coords = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
    print(f"Seçim yapıldı: {box_coords} -> Segmentasyon için ENTER'a basın.")

def on_keypress(event):
    "Klavye tuşlarını dinler"
    global box_coords, current_ax
    
    # Çıkış
    if event.key == 'escape':
        plt.close()
        return

    # Onaylama
    if event.key in ['enter', ' ']:
        if box_coords is None:
            print("Önce bir kutu seçin!")
            return
            
        print("Segmentasyon hesaplanıyor...")
        masks, _, _ = predictor.predict(box=box_coords.reshape(1, 4), multimask_output=False)
        
        # Maskeyi göster
        show_mask(masks[0], current_ax)
        plt.draw()
        print("Maske çizildi! Yeni seçim yapabilirsiniz.")

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6]) # Mavi renk, 0.6 saydamlık
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# -------------------------------
# ANA KOD
# -------------------------------
image_path = "car.jpeg"
print("Model yükleniyor...")

# Model Ayarları
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
except Exception as e:
    print("Model yüklenemedi. Dosya yolunu kontrol edin.")
    sys.exit()

# Resmi Yükle
image = cv2.imread(image_path)
if image is None:
    print("Resim bulunamadı!")
    sys.exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

print("\n--- KULLANIM ---")
print("1. TEKERLEK: Zoom yapın (Yakınlaş/Uzaklaş).")
print("2. SOL TIK SÜRÜKLE: Kutu seçin.")
print("3. ENTER: Seçimi onayla ve maskeyi çıkar.")
print("4. PAN (Kaydırma): Alt menüdeki 'Ok' işaretine basıp sürükleyin.")
print("----------------\n")

# Matplotlib Penceresi
fig, current_ax = plt.subplots(figsize=(10, 8))
current_ax.imshow(image)
current_ax.set_title("Zoom: Tekerlek | Secim: Sol Tik | Onay: ENTER")

box_coords = None

# Zoom Özelliğini Aktif Et
zoom_factory(current_ax)

# Seçim Aracını Aktif Et
selector = RectangleSelector(
    current_ax, onselect,
    useblit=True,
    button=[1], # Sadece sol tık ile seçim
    minspanx=5, minspany=5,
    spancoords='data',
    interactive=True
)

fig.canvas.mpl_connect('key_press_event', on_keypress)
plt.show()
