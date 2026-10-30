import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean sans-serif font
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# ----------------- DIAGRAM 1: MENU RIWAYAT WITH SUB-FEATURES -----------------
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
ax.set_xlim(0, 900)
ax.set_ylim(0, 600)
ax.set_aspect('equal')
ax.axis('off')

# Outer Boundary Frame
rect = patches.Rectangle((40, 40), 820, 520, linewidth=1.5, edgecolor='black', facecolor='white')
ax.add_patch(rect)

# Top Left Title Box
title_poly = patches.Polygon([[40, 560], [420, 560], [440, 530], [40, 530]], 
                             closed=True, linewidth=1.5, edgecolor='black', facecolor='white')
ax.add_patch(title_poly)
ax.text(60, 542, "Sistem Analisis Sentimen BPJS", fontsize=11, fontweight='bold', va='center')

# Draw Actor Function
def draw_actor(ax, x, y, label):
    head = patches.Circle((x, y + 35), 18, linewidth=1.5, edgecolor='black', facecolor='white')
    ax.add_patch(head)
    ax.plot([x, x], [y + 17, y - 25], color='black', lw=1.5)
    ax.plot([x - 25, x + 25], [y + 5, y + 5], color='black', lw=1.5)
    ax.plot([x, x - 20], [y - 25, y - 65], color='black', lw=1.5)
    ax.plot([x, x + 20], [y - 25, y - 65], color='black', lw=1.5)
    ax.text(x, y - 85, label, fontsize=12, ha='center', va='top')

draw_actor(ax, 110, 300, "Pengguna")

# Main Use Case
ellipse_main = patches.Ellipse((360, 300), 200, 80, linewidth=1.5, edgecolor='black', facecolor='white')
ax.add_patch(ellipse_main)
ax.text(360, 300, "Riwayat Analisis", fontsize=12, fontweight='bold', ha='center', va='center')

# Sub Use Cases
def draw_sub_oval(ax, x, y, text):
    ellipse = patches.Ellipse((x, y), 230, 75, linewidth=1.2, edgecolor='black', facecolor='white')
    ax.add_patch(ellipse)
    ax.text(x, y, text, fontsize=10.5, ha='center', va='center')

draw_sub_oval(ax, 690, 440, "Lihat Riwayat Analisis")
draw_sub_oval(ax, 690, 300, "Lihat Statistik Database")
draw_sub_oval(ax, 690, 160, "Download Laporan PDF")

# Arrow: Actor to Main Use Case
ax.annotate('', xy=(260, 300), xytext=(140, 300),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Include Arrows
def draw_inc(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', linestyle='--', lw=1.2, color='black'))
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    ax.text(mx, my + 8, "«include»", fontsize=9, ha='center', va='bottom', backgroundcolor='white')

draw_inc(ax, 460, 330, 575, 415)
draw_inc(ax, 460, 300, 575, 300)
draw_inc(ax, 460, 270, 575, 185)

plt.tight_layout()
plt.savefig("use_case_riwayat_detail.png", bbox_inches='tight', dpi=300)
plt.close()

# ----------------- DIAGRAM 2: SINGLE OVAL SIMPLE -----------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.set_xlim(0, 800)
ax.set_ylim(0, 500)
ax.set_aspect('equal')
ax.axis('off')

# Outer Boundary Frame
rect = patches.Rectangle((40, 40), 720, 420, linewidth=1.5, edgecolor='black', facecolor='white')
ax.add_patch(rect)

# Top Left Title Box
title_poly = patches.Polygon([[40, 460], [380, 460], [400, 430], [40, 430]], 
                             closed=True, linewidth=1.5, edgecolor='black', facecolor='white')
ax.add_patch(title_poly)
ax.text(60, 442, "Sistem Analisis Sentimen BPJS", fontsize=11, fontweight='bold', va='center')

draw_actor(ax, 120, 250, "Pengguna")

# Use Case Oval
ellipse = patches.Ellipse((480, 250), 280, 100, linewidth=1.5, edgecolor='black', facecolor='white')
ax.add_patch(ellipse)
ax.text(480, 250, "Riwayat Analisis", fontsize=13, fontweight='bold', ha='center', va='center')

# Solid arrow
ax.annotate('', xy=(340, 250), xytext=(155, 250),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

plt.tight_layout()
plt.savefig("use_case_riwayat_simpel.png", bbox_inches='tight', dpi=300)
plt.close()

print("Generated single riwayat diagrams successfully.")

