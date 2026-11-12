# -*- coding: utf-8 -*-
"""
EHS 违规台账导出工具
=====================================
从 output/violations/ 的违规截图中提取信息，汇总成 Excel 台账。

功能：
  1. 扫描所有违规截图，从文件名解析时间/帧号/类型
  2. 按日期、按区域、按类型统计
  3. 导出 Excel（含汇总表、明细表、图表）
  4. 输出到 output/EHS违规台账.xlsx

用法：
    python export_ledger.py
    python export_ledger.py --dir output/violations --out output/EHS违规台账.xlsx
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIOLATION_DIR = PROJECT_ROOT / "output" / "violations"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "EHS违规台账.xlsx"

# 违规类型中文名
TYPE_NAMES = {
    "ppe": "PPE未佩戴安全帽",
    "area": "危险区域闯入",
    "combined": "PPE违规+区域闯入",
}

# 文件名格式: {kind}_{YYYYmmdd_HHMMSS}_f{帧号:06d}[_{区域名}].jpg
FNAME_PATTERN = re.compile(
    r"^(?P<kind>ppe|area|combined)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})_"
    r"f(?P<frame>\d+)(?:_(?P<region>.+))?\.jpg$"
)


def parse_filename(fname):
    """从截图文件名解析违规信息"""
    m = FNAME_PATTERN.match(fname)
    if not m:
        return None

    d = m.group("date")
    t = m.group("time")
    try:
        dt = datetime.strptime(d + t, "%Y%m%d%H%M%S")
    except ValueError:
        return None

    kind = m.group("kind")
    return {
        "违规时间": dt,
        "日期": dt.strftime("%Y-%m-%d"),
        "时刻": dt.strftime("%H:%M:%S"),
        "违规类型": TYPE_NAMES.get(kind, kind),
        "区域": m.group("region") or "默认区域",
        "帧号": int(m.group("frame")),
        "截图文件": fname,
    }


def collect_records(vdir):
    """扫描目录，收集所有违规记录"""
    vdir = Path(vdir)
    if not vdir.exists():
        print(f"[ERROR] 目录不存在: {vdir}")
        return pd.DataFrame()

    records = []
    for f in sorted(vdir.glob("*.jpg")):
        rec = parse_filename(f.name)
        if rec is None:
            print(f"[WARN] 文件名不符合格式，跳过: {f.name}")
            continue
        rec["截图路径"] = str(f)
        records.append(rec)

    if not records:
        print("[WARN] 未找到任何违规截图")
        return pd.DataFrame()

    return pd.DataFrame(records)


def build_summary(df):
    """生成汇总统计"""
    print("\n" + "=" * 60)
    print("违规统计汇总")
    print("=" * 60)

    # 1. 按类型
    print("\n【按违规类型统计】")
    by_type = df["违规类型"].value_counts()
    for k, v in by_type.items():
        print(f"  {k}: {v} 次")

    # 2. 按日期
    print("\n【按日期统计】")
    by_date = df["日期"].value_counts().sort_index()
    for k, v in by_date.items():
        print(f"  {k}: {v} 次")

    # 3. 按区域
    print("\n【按区域统计】")
    by_region = df["区域"].value_counts()
    for k, v in by_region.items():
        print(f"  {k}: {v} 次")

    # 4. 总计
    print(f"\n总计违规: {len(df)} 次")
    if len(df) > 0:
        print(f"时间范围: {df['违规时间'].min()} ~ {df['违规时间'].max()}")


def export_excel(df, out_path):
    """导出 Excel（多 sheet）"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("[WARN] 无数据可导出")
        return None

    # 明细表：按时间排序，去掉内部字段
    detail = df.sort_values("违规时间")[
        ["违规时间", "日期", "时刻", "违规类型", "区域", "帧号", "截图文件"]
    ].copy()
    detail["违规时间"] = detail["违规时间"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # 汇总表1：按类型
    type_summary = (
        df.groupby("违规类型").size().reset_index(name="次数")
        .sort_values("次数", ascending=False)
    )

    # 汇总表2：按日期
    date_summary = (
        df.groupby("日期").size().reset_index(name="次数")
        .sort_values("日期")
    )

    # 汇总表3：按区域
    region_summary = (
        df.groupby("区域").size().reset_index(name="次数")
        .sort_values("次数", ascending=False)
    )

    # 交叉表：日期 x 类型
    pivot = pd.crosstab(df["日期"], df["违规类型"], margins=True, margins_name="合计")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        type_summary.to_excel(writer, sheet_name="按类型汇总", index=False)
        date_summary.to_excel(writer, sheet_name="按日期汇总", index=False)
        region_summary.to_excel(writer, sheet_name="按区域汇总", index=False)
        pivot.to_excel(writer, sheet_name="日期类型交叉")
        detail.to_excel(writer, sheet_name="违规明细", index=False)

        # 调整列宽
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(
                    (len(str(cell.value)) for cell in col if cell.value is not None),
                    default=10,
                )
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)

    print(f"\n[OK] Excel 已导出: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="EHS 违规台账导出")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_VIOLATION_DIR),
                        help="违规截图目录")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT),
                        help="输出 Excel 路径")
    args = parser.parse_args()

    print("=" * 60)
    print("EHS 违规台账导出")
    print("=" * 60)
    print(f"扫描目录: {args.dir}\n")

    df = collect_records(args.dir)
    if df.empty:
        print("\n[提示] 没有违规记录。")
        print("先运行检测脚本产生违规截图，再执行本脚本。")
        return

    build_summary(df)
    export_excel(df, args.out)


if __name__ == "__main__":
    main()
