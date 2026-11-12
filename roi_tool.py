# -*- coding: utf-8 -*-
"""
禁区标注工具
=====================================
用法：
    python roi_tool.py --video test_videos/veo3_construction.mp4
    python roi_tool.py --image area_dangerous/1.jpg

操作：
    鼠标左键  : 添加一个顶点
    鼠标右键  : 撤销上一个顶点
    Enter     : 保存并退出（保存到 area_dangerous/area_labels/<文件名>.json）
    R         : 清空重画
    Q / Esc   : 退出不保存

保存的 json 格式与原项目 Smart_Construction 完全一致，
ehs_detect.py 会自动加载同名 json。
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
LABEL_DIR = PROJECT_ROOT / "area_dangerous" / "area_labels"

points = []      # 已点击的顶点
img_display = None
base_img = None
window_name = "ROI Tool"


def redraw():
    """重绘画面"""
    global img_display
    img_display = base_img.copy()

    if points:
        # 画已点击的点
        for i, pt in enumerate(points):
            cv2.circle(img_display, pt, 5, (0, 255, 255), -1)
            cv2.putText(
                img_display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

        # 画连线（点 >= 2 时才连）
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(img_display, points[i], points[i + 1], (0, 255, 0), 2)

        # 预览闭合多边形（点 >= 3）
        if len(points) >= 3:
            poly = np.array(points, dtype=np.int32)
            overlay = img_display.copy()
            cv2.fillPoly(overlay, [poly], (0, 0, 180))
            img_display = cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0)

    # 顶部提示
    hints = [
        "Left-click: add point | Right-click: undo | Enter: save | R: reset | Q: quit",
        f"Points: {len(points)}",
    ]
    for i, text in enumerate(hints):
        cv2.putText(
            img_display, text, (10, 25 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

    cv2.imshow(window_name, img_display)


def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"  添加顶点 {len(points)}: ({x}, {y})")
        redraw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            removed = points.pop()
            print(f"  撤销顶点: {removed}")
            redraw()


def save_roi(out_json):
    """按原项目格式保存 json"""
    if len(points) < 3:
        print("[WARN] 至少需要 3 个点才能构成多边形，当前 %d 个" % len(points))
        return False

    polygon = {}
    for i, (x, y) in enumerate(points):
        idx = str(i + 1)
        polygon["x" + idx] = x
        polygon["y" + idx] = y

    data = {
        "outputs": {
            "object": [
                {
                    "name": "dangerous_area",
                    "polygon": polygon,
                }
            ]
        }
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已保存: {out_json}")
    print(f"     共 {len(points)} 个顶点")
    return True


def main():
    global base_img, img_display

    parser = argparse.ArgumentParser(description="禁区多边形标注工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="视频路径（取第一帧标注）")
    group.add_argument("--image", type=str, help="图片路径")
    parser.add_argument("--out", type=str, default="",
                        help="输出 json 路径，默认 area_dangerous/area_labels/<名字>.json")
    args = parser.parse_args()

    # 读取画面
    if args.image:
        base_img = cv2.imread(args.image)
        stem = Path(args.image).stem
        if base_img is None:
            print(f"[ERROR] 无法读取图片: {args.image}")
            return
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {args.video}")
            return
        ret, base_img = cap.read()
        cap.release()
        stem = Path(args.video).stem
        if not ret:
            print(f"[ERROR] 无法读取视频帧: {args.video}")
            return

    print("=" * 60)
    print("禁区标注工具")
    print("=" * 60)
    print(f"画面尺寸: {base_img.shape[1]} x {base_img.shape[0]}")
    print("操作说明:")
    print("  鼠标左键  → 添加顶点")
    print("  鼠标右键  → 撤销上一个顶点")
    print("  Enter     → 保存 json 并退出")
    print("  R         → 清空重画")
    print("  Q / Esc   → 退出不保存")
    print("-" * 60)

    # 输出路径
    out_json = args.out if args.out else (LABEL_DIR / f"{stem}.json")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):      # Enter
            save_roi(out_json)
            break
        elif key in (ord("r"), ord("R")):
            points.clear()
            print("  已清空，重新开始")
            redraw()
        elif key in (ord("q"), ord("Q"), 27):   # q / Esc
            print("  退出，未保存")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
