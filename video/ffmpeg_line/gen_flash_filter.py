"""
gen_flash_filter.py — 讀 thunder_events.json,生成 ffmpeg overlay enable 閃電表達式

閃電設計:
  主閃  80ms (t_flash 起): 白色 overlay alpha=0.35
  回閃  40ms (t_flash + 120ms 起): 白色 overlay alpha=0.18
  兩階段 overlay:main_flash layer + echo_flash layer

輸出:
  video/ffmpeg_line/assets/flash_filter.txt  — 包含兩個 enable= 表達式(逐行)
  video/ffmpeg_line/assets/flash_times.json  — 供 verify_m6.py 驗收

用法:
  python video/ffmpeg_line/gen_flash_filter.py [--events PATH] [--out-dir PATH]
"""

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.parent
DEFAULT_EVENTS = ROOT / "out/long/thunder_events.json"
DEFAULT_OUT = ROOT / "video/ffmpeg_line/assets"

MAIN_FLASH_DUR   = 0.080   # 80ms
ECHO_FLASH_DELAY = 0.120   # 主閃後 120ms 才開始
ECHO_FLASH_DUR   = 0.040   # 40ms

# overlay alpha(白色疊加不透明度)
MAIN_ALPHA = 0.35
ECHO_ALPHA = 0.18


def build_enable_exprs(events: list):
    """
    為每個 thunder event 產生 ffmpeg overlay enable 表達式。
    overlay 的 enable= 接受標準 ffmpeg eval 表達式,逗號不需跳脫。

    回傳:
      main_enable  -- 所有主閃時段的 enable 表達式(小寫 t)
      echo_enable  -- 所有回閃時段的 enable 表達式
      flash_windows -- 所有時窗列表(供 verify_m6.py)
    """
    main_terms = []
    echo_terms = []
    flash_windows = []

    for ev in events:
        t_flash = ev["t_flash"]

        # 主閃
        t0 = t_flash
        t1 = t_flash + MAIN_FLASH_DUR
        main_terms.append(f"between(t,{t0:.4f},{t1:.4f})")
        flash_windows.append({"t_start": t0, "t_end": t1, "type": "main",
                               "alpha": MAIN_ALPHA})

        # 回閃
        t2 = t_flash + ECHO_FLASH_DELAY
        t3 = t2 + ECHO_FLASH_DUR
        echo_terms.append(f"between(t,{t2:.4f},{t3:.4f})")
        flash_windows.append({"t_start": t2, "t_end": t3, "type": "echo",
                               "alpha": ECHO_ALPHA})

    main_enable = "+".join(main_terms) if main_terms else "0"
    echo_enable = "+".join(echo_terms) if echo_terms else "0"

    return main_enable, echo_enable, flash_windows


def main():
    parser = argparse.ArgumentParser(description="生成閃電 ffmpeg overlay filter")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    events_path = pathlib.Path(args.events)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(events_path) as f:
        data = json.load(f)

    events = data.get("events", [])
    print(f"[gen_flash_filter] {len(events)} thunder events loaded from {events_path}")

    main_enable, echo_enable, flash_windows = build_enable_exprs(events)

    # 輸出到 flash_filter.txt:第1行=main enable, 第2行=echo enable
    flash_filter_path = out_dir / "flash_filter.txt"
    with open(flash_filter_path, "w") as f:
        f.write(f"MAIN_ENABLE={main_enable}\n")
        f.write(f"ECHO_ENABLE={echo_enable}\n")
        f.write(f"MAIN_ALPHA={MAIN_ALPHA}\n")
        f.write(f"ECHO_ALPHA={ECHO_ALPHA}\n")

    print(f"[gen_flash_filter] flash_filter.txt saved -> {flash_filter_path}")

    # flash_times.json 供 verify_m6.py 驗收
    flash_times_path = out_dir / "flash_times.json"
    main_flashes = [w for w in flash_windows if w["type"] == "main"]
    with open(flash_times_path, "w") as f:
        json.dump({
            "n_events": len(events),
            "main_flashes": main_flashes,
            "all_windows": flash_windows,
            "main_alpha": MAIN_ALPHA,
            "echo_alpha": ECHO_ALPHA,
        }, f, indent=2)
    print(f"[gen_flash_filter] flash_times.json saved -> {flash_times_path}")
    print(f"[gen_flash_filter] {len(main_flashes)} main flashes, "
          f"{len(flash_windows)} total windows")


if __name__ == "__main__":
    main()
