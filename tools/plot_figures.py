"""Generate the MoE teaching figure with the Python standard library."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def moe_svg():
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720" role="img" aria-labelledby="title desc">
<title id="title">Mixture-of-Experts: router і два вибрані FFN</title>
<desc id="desc">Після RMSNorm стан токена z надходить у router і вибрані експерти. Router обирає експерти 1 і 3. Їхні виходи змішуються з вагами 0.625 і 0.375 та додаються до residual x. Експерт 2 не обчислюється.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="context-stroke"/></marker></defs>
<rect width="1200" height="720" rx="16" fill="#f8fafc"/>
<g font-family="Arial, sans-serif" fill="#0f172a">''']
    def text(x, y, value, size=18, color='#0f172a', anchor='start'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')
    def box(x,y,w,h,title,subtitle,fill='#ffffff',stroke='#cbd5e1'):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text(x+w/2,y+30,title,20,anchor='middle')
        text(x+w/2,y+55,subtitle,16,'#475569','middle')
    def path(d,color='#334155',dashed=False):
        dash=' stroke-dasharray="7 5"' if dashed else ''
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"{dash} marker-end="url(#arrow)"/>')
    text(36,45,'Mixture-of-Experts: новий router, уже знайомі FFN',28)
    text(36,76,'Один токен · 3 експерти · top-2 · заміна FFN-гілки Transformer-блока',19,'#475569')
    path('M85 405 V135 H1080 V376')
    text(480,125,'Residual: переносимо x без змін',18,'#475569')
    text(48,414,'x',25)
    path('M65 405 H135')
    box(135,368,150,74,'RMSNorm','z = norm(x)')
    path('M285 405 H350')
    parts.append('<circle cx="350" cy="405" r="5" fill="#2563eb"/>')
    text(325,435,'z',21,'#2563eb')
    path('M350 405 V265 H405','#2563eb')
    box(405,218,255,90,'Router: Linear(D, 3)','softmax → [0.50, 0.20, 0.30]','#eff6ff','#60a5fa')
    box(405,368,255,90,'Top-2 + нормування','1 і 3 → [0.625, 0.375]','#eff6ff','#60a5fa')
    # Routing is a control path; token states travel along separate solid paths.
    path('M532 308 V368','#7c3aed',True)
    path('M660 405 H695 V205 H740','#7c3aed',True)
    path('M695 405 V505 H740','#7c3aed',True)
    path('M350 405 V185 H715 V240 H740','#2563eb')
    path('M350 405 V565 H715 V540 H740','#2563eb')
    box(740,185,185,80,'Експерт 1','FFN₁(z)','#ecfdf5','#34d399')
    box(740,335,185,80,'Експерт 2','не викликаємо','#f1f5f9','#cbd5e1')
    box(740,485,185,80,'Експерт 3','FFN₃(z)','#ecfdf5','#34d399')
    path('M925 225 H985 V345','#059669')
    text(943,210,'× 0.625',16,'#047857')
    path('M925 525 H985 V441','#059669')
    text(943,555,'× 0.375',16,'#047857')
    box(942,345,86,96,'Σ','суміш','#ecfdf5','#34d399')
    path('M1028 393 H1056')
    parts.append('<circle cx="1080" cy="400" r="24" fill="#ffffff" stroke="#334155" stroke-width="2"/>')
    text(1080,408,'+',27,anchor='middle')
    path('M1104 400 H1160')
    text(1134,435,'новий x',16,anchor='middle')
    text(36,625,'y = 0.625 · FFN₁(z) + 0.375 · FFN₃(z)       новий x = x + y',23)
    text(36,661,'Суцільні стрілки: стани токена / виходи. Пунктир: вибір експертів і ваги router.',18,'#475569')
    text(36,692,'Вибір повторюється для кожного токена; інший токен може обрати інші експерти.',18,'#475569')
    parts.append('</g></svg>')
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    destination = ROOT / 'figures' / 'moe.svg'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(moe_svg(), encoding='utf-8')
