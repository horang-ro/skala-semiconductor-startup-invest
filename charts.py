"""투자 보고서에 넣을 그래프를 생성합니다."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from config import AREA_WEIGHTS  # noqa: E402

KOREAN_FONTS = ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]


def set_korean_font() -> None:
    """설치된 한글 글꼴을 찾아 적용합니다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def draw_summary_chart(evaluations: list[dict], forecasts: list[dict], top_ids: list[str], path: Path) -> Path:
    """통과 기업의 영역별 점수(왼쪽)와 예상 수익률(오른쪽)을 한 장에 그립니다."""
    set_korean_font()
    by_id = {e["company"]["id"]: e for e in evaluations}
    rows = sorted(forecasts, key=lambda f: -f["total"])
    names = [f["company"]["name_ko"] + (" ★" if f["company"]["id"] in top_ids else "") for f in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))

    # 왼쪽: 영역별 점수 누적 막대
    left = [0.0] * len(rows)
    colors = {"tech": "#3b6fb6", "market": "#6aa84f", "revenue": "#e69138"}
    labels = {"tech": "기술", "market": "시장", "revenue": "매출"}
    for area in AREA_WEIGHTS:
        values = [by_id[f["company"]["id"]]["scores"][area] for f in rows]
        ax1.barh(names, values, left=left, color=colors[area], label=f"{labels[area]} ({AREA_WEIGHTS[area]})")
        left = [a + b for a, b in zip(left, values)]
    for y, total in enumerate(left):
        ax1.text(total + 1, y, f"{total:.1f}", va="center", fontsize=8)
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()
    ax1.set_title("통과 기업 평가 점수 (100점)", fontsize=10)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.tick_params(labelsize=8)

    # 오른쪽: 예상 수익률
    rois = [(f["roi"] or 0) * 100 for f in rows]
    bar_colors = ["#3b6fb6" if r >= 0 else "#cc4125" for r in rois]
    ax2.barh(names, rois, color=bar_colors)
    for y, roi in enumerate(rois):
        ax2.text(roi + (4 if roi >= 0 else -4), y, f"{roi:+.1f}%", va="center",
                 ha="left" if roi >= 0 else "right", fontsize=8)
    ax2.axvline(0, color="#333", linewidth=0.8)
    span = max(abs(min(rois)), abs(max(rois)), 10)
    ax2.set_xlim(-span * 1.5, span * 1.5)
    ax2.invert_yaxis()
    ax2.set_title("예상 수익률 (★ = 추천 TOP3)", fontsize=10)
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
