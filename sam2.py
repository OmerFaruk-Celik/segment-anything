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
        if event.inaxes != ax: return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None: return 

        if event.button == 'up': scale_factor = 1 / base_scale
        elif event.button == 'down': scale_factor = base_scale
        else: scale_factor = 1

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
        ax.figure.canvas.draw_idle()

    fig = ax.get_figure()
    fig.canvas.mpl_connect('scroll_event', zoom_fun)
    return zoom_fun

# -------------------------------
# SEÇİM VE MASKELEME
# -------------------------------
def onselect(eclick, erelease):
    global box_coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    box_coords = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
    print(f"\nSeçim yapıldı: {box_coords} -> Segmentasyon için ENTER'a basın.")

def on_keypress(event):
    global box_coords, current_ax
    if event.key == 'escape':
        plt.close()
        return

    if event.key in ['enter', ' ']:
        if box_coords is None:
            print("Önce bir kutu seçin!")
            return
            
        print("Segmentasyon hesaplanıyor...")
        masks, scores, _ = predictor.predict(box=box_coords.reshape(1, 4), multimask_output=False)
        
        # Seçilen en iyi maske (True/False matrisi)
        binary_mask = masks[0] 
        
        # ---------------------------------------------------------
        # 🟢 KOORDİNATLARI ALMA KISMI 🟢
        # ---------------------------------------------------------
        # np.where, maskenin True olduğu yerlerin (satır, sütun) indekslerini verir.
        # Resim dünyasında: Satır = Y, Sütun = X demektir.
        ys, xs = np.where(binary_mask)
        
        # Koordinatları (x, y) çiftleri haline getirelim
        pixel_coords = np.column_stack((xs, ys))
        
        print(f"\n--- SONUÇLAR ---")
        print(f"Maske Skoru: {scores[0]:.2f}")
        print(f"Toplam Piksel Sayısı: {len(pixel_coords)}")
        
        # Terminale ilk ve son 5 pikseli yazdıralım (Hepsini yazarsak terminal donar)
        print("\nÖrnek Koordinatlar [X, Y]:")
        if len(pixel_coords) > 10:
            print(pixel_coords[:5])
            print("...")
            print(pixel_coords[-5:])
        else:
            print(pixel_coords)
            
        # Dosyaya kaydetme (İsteğe bağlı)
        np.savetxt("koordinatlar.txt", pixel_coords, fmt='%d', header="X Y", comments='')
        print(f"\nTüm koordinatlar 'koordinatlar.txt' dosyasına kaydedildi!")
        # ---------------------------------------------------------

        # Maskeyi göster
        show_mask(binary_mask, current_ax)
        plt.draw()

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6]) 
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# -------------------------------
# ANA KOD
# -------------------------------
image_path = "car.jpeg"
print("Model yükleniyor...")

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
except Exception as e:
    print("Model hatası:", e)
    sys.exit()

image = cv2.imread(image_path)
if image is None: sys.exit("Resim yok!")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

print("\nHazır! Kutu çizip ENTER'a basın.")
fig, current_ax = plt.subplots(figsize=(10, 8))
current_ax.imshow(image)
box_coords = None
zoom_factory(current_ax)
selector = RectangleSelector(current_ax, onselect, useblit=True, button=[1], interactive=True)
fig.canvas.mpl_connect('key_press_event', on_keypress)
plt.show()
