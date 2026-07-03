from pathlib import Path
import matplotlib
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

CANDIDATE_FAMILIES = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
    'Noto Sans CJK', 'Source Han Sans SC', 'WenQuanYi Zen Hei',
    'Arial Unicode MS', 'PingFang SC'
]
CANDIDATE_FILES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/System/Library/Fonts/PingFang.ttc',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
]

def get_chinese_font():
    """Return a FontProperties that supports Chinese when possible, without crashing on missing fonts."""
    # 1) direct file candidates
    for fp in CANDIDATE_FILES:
        try:
            p = Path(fp)
            if p.exists():
                return FontProperties(fname=str(p))
        except Exception:
            pass
    # 2) installed font family candidates
    try:
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for fam in CANDIDATE_FAMILIES:
            if fam in installed:
                matplotlib.rcParams['font.sans-serif'] = [fam] + matplotlib.rcParams.get('font.sans-serif', [])
                matplotlib.rcParams['axes.unicode_minus'] = False
                return FontProperties(family=fam)
    except Exception:
        pass
    # 3) fallback: will not guarantee Chinese glyphs, but should not crash.
    matplotlib.rcParams['axes.unicode_minus'] = False
    return FontProperties()
