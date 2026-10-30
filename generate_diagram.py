import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up canvas
fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1100)
ax.set_aspect('equal')
ax.axis('off')

# Outer Boundary Frame (System Boundary)
rect = patches.Rectangle((40, 40), 920, 1000, linewidth=1.5, edgecolor='black', facecolor='white')
ax.add_patch(rect)

# Top Left Title Box
title_poly = patches.Polygon([[40, 1040], [400, 1040], [420, 1010], [40, 1010]], 
                             closed=True, linewidth=1.5, edgecolor='black', facecolor='white')
ax.add_patch(title_poly)
ax.text(60, 1022, "Sistem Analisis Sentimen BPJS", fontsize=12, fontweight='bold', va='center')

# Function to draw stick figure actor
def draw_actor(x, y, label):
    # Head
    head = patches.Circle((x, y + 35), 18, linewidth=1.5, edgecolor='black', facecolor='white')
    ax.add_patch(head)
    # Body
    ax.plot([x, x], [y + 17, y - 25], color='black', lw=1.5)
    # Arms
    ax.plot([x - 25, x + 25], [y + 5, y + 5], color='black', lw=1.5)
    # Leg Left
    ax.plot([x, x - 20], [y - 25, y - 65], color='black', lw=1.5)
    # Leg Right
    ax.plot([x, x + 20], [y - 25, y - 65], color='black', lw=1.5)
    # Label
    ax.text(x, y - 85, label, fontsize=12, ha='center', va='top')

draw_actor(90, 600, "Pengguna")

# Function to draw Use Case oval
def draw_oval(x, y, rx, ry, text):
    ellipse = patches.Ellipse((x, y), rx * 2, ry * 2, linewidth=1.2, edgecolor='black', facecolor='white')
    ax.add_patch(ellipse)
    ax.text(x, y, text, fontsize=11, ha='center', va='center')

# Main Use Cases (Middle Column)
# 1. Dashboard
draw_oval(380, 920, 80, 35, "Dashboard")
# 2. Prediksi
draw_oval(380, 700, 80, 35, "Prediksi")
# 3. Data Set
draw_oval(380, 470, 80, 35, "Data Set")
# 4. Riwayat Analisis
draw_oval(380, 200, 100, 35, "Riwayat Analisis")

# Sub Use Cases (Right Column)
# Dashboard sub
draw_oval(740, 920, 120, 35, "Lihat grafik statistik")

# Prediksi subs
draw_oval(740, 830, 125, 35, "Input teks / upload CSV")
draw_oval(740, 740, 135, 35, "Jalankan model SVM & KNN")
draw_oval(740, 650, 120, 35, "Lihat hasil prediksi")
draw_oval(740, 560, 120, 35, "Simpan otomatis ke DB")

# Data Set sub
draw_oval(740, 470, 120, 35, "Lihat tabel dataset")

# Riwayat Analisis subs
draw_oval(740, 280, 125, 35, "Lihat Riwayat Analisis")
draw_oval(740, 190, 130, 35, "Lihat Statistik Database")
draw_oval(740, 100, 125, 35, "Download Laporan PDF")

# Helper for solid arrow from Actor to Main Use Cases
def draw_solid_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))

# Solid arrows from Pengguna (x=115, y=600 approx)
draw_solid_arrow(115, 620, 300, 920)
draw_solid_arrow(115, 600, 300, 700)
draw_solid_arrow(115, 580, 300, 470)
draw_solid_arrow(115, 560, 280, 200)

# Helper for dashed arrow <<include>>
def draw_include_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', linestyle='--', lw=1.2, color='black'))
    # Label midpoint
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    ax.text(mx, my + 10, "«include»", fontsize=9, ha='center', va='bottom', backgroundcolor='white')

# Include arrows
# Dashboard
draw_include_arrow(460, 920, 620, 920)

# Prediksi
draw_include_arrow(450, 725, 615, 815)
draw_include_arrow(460, 710, 605, 735)
draw_include_arrow(460, 690, 620, 660)
draw_include_arrow(450, 675, 620, 580)

# Data Set
draw_include_arrow(460, 470, 620, 470)

# Riwayat Analisis (3 Includes)
draw_include_arrow(465, 225, 615, 270)
draw_include_arrow(480, 200, 610, 190)
draw_include_arrow(465, 175, 615, 110)

plt.tight_layout()
plt.savefig("use_case_diagram_fixed.png", bbox_inches='tight', dpi=300)
print("Diagram successfully saved as use_case_diagram_fixed.png")

