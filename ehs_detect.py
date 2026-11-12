# -*- coding: utf-8 -*-
"""
EHS 智慧工地安全监测脚本
=====================================
功能：
  1. PPE 违规判定：检测 head（未戴帽的头）+ helmet（安全帽），
     用 IoU 匹配判断"有头无帽"→ 标记 PPE 违规
  2. 危险区域闯入：多边形 ROI 判定，person 进入禁区 → 报警

技术方案（双模型协同）：
  - ppe_model   : 自训练 YOLOv11n，识别 head / helmet（mAP50: helmet 0.970）
  - person_model: COCO 预训练 YOLOv11n，识别 person（解决自建数据集 person 标注不足）

用法：
  python ehs_detect.py --source 图片或视频路径
  python ehs_detect.py --source 0                          # 摄像头
  python ehs_detect.py --source area_dangerous/1.jpg --roi "100,300,600,300,600,700,100,700"
"""
import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

# ============================================================
# 默认配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PPE_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train-4" / "weights" / "best.pt"
DEFAULT_PERSON_WEIGHTS = PROJECT_ROOT / "yolo11n.pt"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIOLATION_DIR = OUTPUT_DIR / "violations"

# PPE 模型类别索引（data.yaml: names: ['head', 'helmet', 'person']）
CLS_HEAD = 0
CLS_HELMET = 1

# COCO 中 person 的类别索引
COCO_PERSON_IDX = 0

# 颜色 (BGR)
COLOR_HELMET_OK = (0, 200, 0)      # 绿色：戴了安全帽
COLOR_VIOLATION = (0, 0, 255)      # 红色：PPE 违规 / 禁区闯入
COLOR_PERSON = (255, 160, 0)       # 蓝色：检测到的人
COLOR_ROI = (0, 0, 255)            # 红色：禁区边框


# ============================================================
# 声音报警
# ============================================================
class AlarmPlayer:
    """
    非阻塞报警音（Windows winsound）
    极端区分方案 —— 按"播放次数"区分违规类型，一耳就能分辨：
      PPE 违规      → Windows Ding.wav  响 1 次   （轻轻的"叮"一声）
      禁区闯入      → Alarm05.wav       响 5 次   （嘟——嘟——嘟——嘟——嘟）
      两者同时      → Alarm05.wav       响 8 次   （连续不断的嘟嘟嘟嘟嘟嘟嘟嘟）
    """

    SOUND_PPE = r"C:\Windows\Media\Windows Ding.wav"   # 叮（1.09s）
    SOUND_AREA = r"C:\Windows\Media\Alarm05.wav"       # 长警报（8.26s）

    REPEAT_PPE = 1
    REPEAT_AREA = 5
    REPEAT_BOTH = 8

    def __init__(self, enabled=True, cooldown=2.0):
        self.enabled = bool(enabled) and _HAS_WINSOUND
        self.cooldown = cooldown
        self._last = {}
        self._lock = threading.Lock()
        self._playing = False
        if enabled and not _HAS_WINSOUND:
            print("[WARN] 当前系统不支持 winsound，报警音已禁用")
        if self.enabled:
            import os
            missing = [
                p for p in (self.SOUND_PPE, self.SOUND_AREA)
                if not os.path.exists(p)
            ]
            if missing:
                print(f"[WARN] 缺少声音文件 {missing}")

    def play(self, kind="ppe"):
        """kind: 'ppe' | 'area' | 'combined'"""
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            if self._playing:                      # 上一段报警还没播完，不叠加
                return
            if now - self._last.get(kind, 0) < self.cooldown:
                return
            self._last[kind] = now
            self._playing = True
        threading.Thread(target=self._play_sound, args=(kind,), daemon=True).start()

    def _play_sound(self, kind):
        try:
            if kind == "ppe":
                wav, repeat = self.SOUND_PPE, self.REPEAT_PPE
            elif kind == "area":
                wav, repeat = self.SOUND_AREA, self.REPEAT_AREA
            else:
                wav, repeat = self.SOUND_AREA, self.REPEAT_BOTH

            for i in range(repeat):
                winsound.PlaySound(wav, winsound.SND_FILENAME)   # 同步播放，等播完
                if i < repeat - 1:
                    time.sleep(0.05)
        except Exception:
            pass
        finally:
            with self._lock:
                self._playing = False


# ============================================================
# 工具函数
# ============================================================
def load_roi(roi_str):
    """把 "x1,y1,x2,y2,..." 解析成 numpy 多边形数组"""
    if not roi_str:
        return None
    try:
        nums = [int(v) for v in roi_str.split(",")]
    except ValueError:
        print("[WARN] ROI 格式错误，应为 'x1,y1,x2,y2,...'")
        return None
    if len(nums) < 6 or len(nums) % 2 != 0:
        print(f"[WARN] ROI 至少需要 3 个点（6 个数），当前 {len(nums)} 个数")
        return None
    return np.array(nums, dtype=np.int32).reshape(-1, 2)


def load_roi_from_json(json_path):
    """
    复用原项目 Smart_Construction 的禁区标注格式：
    area_dangerous/area_labels/*.json 里的 polygon 点集
    """
    if not Path(json_path).exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        area_poly = []
        for area_info in info["outputs"]["object"]:
            if "polygon" not in area_info:
                return None
            pts_len = len(area_info["polygon"])
            if pts_len % 2 != 0:
                return None
            for i in range(pts_len // 2):
                idx = str(i + 1)
                area_poly.append(
                    [area_info["polygon"]["x" + idx], area_info["polygon"]["y" + idx]]
                )
        return np.array(area_poly, dtype=np.int32) if area_poly else None
    except Exception as e:
        print(f"[WARN] 读取 {json_path} 失败: {e}")
        return None


def box_iou(box_a, box_b):
    """计算两个 xyxy 框的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def point_in_polygon(point, polygon):
    """判断点是否在多边形内"""
    return cv2.pointPolygonTest(
        polygon, (float(point[0]), float(point[1])), False
    ) >= 0


def draw_box(img, box, color, label, thickness=2):
    """画框 + 标签"""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )


def save_violation(frame, kind, frame_idx, region_name=""):
    """保存违规截图"""
    VIOLATION_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{region_name}" if region_name else ""
    fname = VIOLATION_DIR / f"{kind}_{ts}_f{frame_idx:06d}{tag}.jpg"
    cv2.imwrite(str(fname), frame)
    return fname


# ============================================================
# PPE 合规判定
# ============================================================
def judge_ppe(head_boxes, helmet_boxes, iou_thr=0.25):
    """
    对每个 head 框检查是否有 helmet 框重叠：
      有重叠 → 戴了安全帽（合规）
      无重叠 → 未戴安全帽（违规）
    """
    ok_heads = []
    violation_heads = []

    for head in head_boxes:
        best_iou = 0.0
        best_helmet = None
        for helmet in helmet_boxes:
            iou = box_iou(head, helmet)
            if iou > best_iou:
                best_iou = iou
                best_helmet = helmet
        if best_iou >= iou_thr:
            ok_heads.append((head, best_helmet))
        else:
            violation_heads.append(head)

    return ok_heads, violation_heads


# ============================================================
# 主流程
# ============================================================
def run(args):
    print("=" * 60)
    print("EHS 智慧工地安全监测系统")
    print("=" * 60)

    if not Path(args.ppe_weights).exists():
        print(f"[ERROR] PPE 模型不存在: {args.ppe_weights}")
        return

    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 设备: {device}")

    print(f"[INFO] 加载 PPE 模型: {Path(args.ppe_weights).name}")
    ppe_model = YOLO(str(args.ppe_weights))

    print(f"[INFO] 加载 person 检测模型: {Path(args.person_weights).name}")
    person_model = YOLO(str(args.person_weights))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIOLATION_DIR.mkdir(parents=True, exist_ok=True)

    roi = load_roi(args.roi)

    # 打开输入
    source = args.source
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开: {source}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 画面: {width}x{height} @ {fps_in:.1f}fps")

    # 没给 ROI 时，尝试从原项目 json 加载
    if roi is None and not is_webcam:
        json_path = PROJECT_ROOT / "area_dangerous" / "area_labels" / (
            Path(source).stem + ".json"
        )
        if json_path.exists():
            roi = load_roi_from_json(json_path)
            if roi is not None:
                print(f"[INFO] 从 {json_path.name} 加载禁区 ({len(roi)} 个点)")

    if roi is not None:
        print(f"[INFO] 禁区已启用，{len(roi)} 个顶点")

    writer = None
    out_path = None
    if args.save_video:
        out_path = OUTPUT_DIR / f"result_{Path(source).stem if not is_webcam else 'webcam'}.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height)
        )

    frame_idx = 0
    total_ppe = 0
    total_area = 0
    t_start = time.time()

    # 声音报警器
    alarm = AlarmPlayer(enabled=not args.no_alarm, cooldown=args.alarm_cooldown)
    if alarm.enabled:
        print(f"[INFO] 声音报警已启用（间隔 {args.alarm_cooldown}s）")
    else:
        print("[INFO] 声音报警已禁用")

    print("-" * 60)
    print("[INFO] 开始检测（按 q 退出）...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 本帧违规标记
        frame_has_ppe = False
        frame_has_area = False

        # 画禁区
        if roi is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [roi], (0, 0, 120))
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
            cv2.polylines(frame, [roi], True, COLOR_ROI, 3)

        # PPE 模型推理
        res_ppe = ppe_model.predict(frame, conf=args.conf, verbose=False, device=device)[0]

        head_boxes, helmet_boxes = [], []
        for box in res_ppe.boxes:
            cls_id = int(box.cls)
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf)
            if cls_id == CLS_HELMET:
                helmet_boxes.append(xyxy)
                draw_box(frame, xyxy, COLOR_HELMET_OK, f"helmet {conf:.2f}")
            elif cls_id == CLS_HEAD:
                head_boxes.append(xyxy)

        # PPE 合规判定
        ok_heads, bad_heads = judge_ppe(head_boxes, helmet_boxes, args.iou_ppe)

        for head_box, _ in ok_heads:
            draw_box(frame, head_box, COLOR_HELMET_OK, "PPE OK")

        for head_box in bad_heads:
            draw_box(frame, head_box, COLOR_VIOLATION, "PPE VIOLATION!", 3)
            total_ppe += 1
            frame_has_ppe = True

        # person 推理 + 禁区判定
        if roi is not None:
            res_person = person_model.predict(
                frame, conf=args.conf, classes=[COCO_PERSON_IDX],
                verbose=False, device=device
            )[0]

            for box in res_person.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf)
                foot = (int((xyxy[0] + xyxy[2]) / 2), int(xyxy[3]))
                if point_in_polygon(foot, roi):
                    total_area += 1
                    frame_has_area = True
                    draw_box(frame, xyxy, COLOR_VIOLATION, f"INTRUSION {conf:.2f}", 3)
                    cv2.circle(frame, foot, 6, COLOR_VIOLATION, -1)
                    cv2.putText(
                        frame, "DANGER ZONE INTRUSION!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_VIOLATION, 3
                    )
                else:
                    draw_box(frame, xyxy, COLOR_PERSON, f"person {conf:.2f}")
                    cv2.circle(frame, foot, 4, COLOR_PERSON, -1)

        # ===== 声音报警（每帧最多触发一次，由 cooldown 控制频率）=====
        if frame_has_ppe and frame_has_area:
            alarm.play("combined")
        elif frame_has_ppe:
            alarm.play("ppe")
        elif frame_has_area:
            alarm.play("area")

        # 保存违规截图
        if (bad_heads or total_area) and (frame_idx % args.snapshot_interval == 0):
            if bad_heads and roi is not None:
                save_violation(frame, "combined", frame_idx)
            elif bad_heads:
                save_violation(frame, "ppe", frame_idx)
            else:
                save_violation(frame, "area", frame_idx)

        # 统计信息
        for i, line in enumerate([
            f"Frame: {frame_idx}",
            f"PPE violation: {total_ppe}",
            f"Area intrusion: {total_area}",
        ]):
            cv2.putText(
                frame, line, (max(width - 320, 10), 30 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        # ===== 按键控制 =====
        #   空格 : 暂停 / 继续
        #   s    : 手动截图
        #   q    : 退出
        if args.show:
            cv2.putText(
                frame, "SPACE:pause  S:snapshot  Q:quit",
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("EHS Monitor", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] 用户退出 (q)")
                break

            elif key == ord(" "):
                # 暂停：显示 PAUSED 遮罩，等待再次按空格
                print("[INFO] 已暂停，按空格继续...")
                paused_frame = frame.copy()
                cv2.rectangle(paused_frame, (0, 0), (width, height), (0, 0, 0), -1)
                cv2.addWeighted(paused_frame, 0.45, frame, 0.55, 0, paused_frame)
                cv2.putText(
                    paused_frame, "PAUSED", (width // 2 - 100, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4
                )
                cv2.putText(
                    paused_frame, "Press SPACE to resume, Q to quit",
                    (width // 2 - 210, height // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                cv2.imshow("EHS Monitor", paused_frame)

                while True:
                    k2 = cv2.waitKey(100) & 0xFF
                    if k2 == ord(" "):
                        print("[INFO] 继续检测")
                        break
                    elif k2 == ord("q"):
                        print("[INFO] 用户退出 (q)")
                        cap.release()
                        if writer is not None:
                            writer.release()
                        cv2.destroyAllWindows()
                        return

            elif key == ord("s"):
                # 手动截图
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                snap_path = OUTPUT_DIR / f"snapshot_{ts}.jpg"
                cv2.imwrite(str(snap_path), frame)
                print(f"[INFO] 已截图: {snap_path}")

        if frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            print(
                f"  帧 {frame_idx} | PPE违规 {total_ppe} | "
                f"禁区闯入 {total_area} | {frame_idx / max(elapsed, 1e-9):.1f} FPS"
            )

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] 结果视频已保存: {out_path}")
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print("-" * 60)
    print("[完成] 检测结束")
    print(f"  总帧数       : {frame_idx}")
    print(f"  耗时         : {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-9):.1f} FPS)")
    print(f"  PPE 违规次数 : {total_ppe}")
    print(f"  禁区闯入次数 : {total_area}")
    print(f"  截图目录     : {VIOLATION_DIR}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="EHS 智慧工地安全监测")
    parser.add_argument("--source", type=str, required=True,
                        help="图片/视频路径，或 0 表示摄像头")
    parser.add_argument("--ppe-weights", type=str, default=str(DEFAULT_PPE_WEIGHTS))
    parser.add_argument("--person-weights", type=str, default=str(DEFAULT_PERSON_WEIGHTS))
    parser.add_argument("--roi", type=str, default="",
                        help="禁区多边形 'x1,y1,x2,y2,...'，为空则尝试从 json 加载")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    parser.add_argument("--iou-ppe", type=float, default=0.25,
                        help="head 与 helmet 匹配 IoU 阈值")
    parser.add_argument("--snapshot-interval", type=int, default=30,
                        help="每隔多少帧存一次违规截图")
    parser.add_argument("--save-video", action="store_true", help="保存结果视频")
    parser.add_argument("--show", action="store_true", help="实时显示窗口")
    parser.add_argument("--no-alarm", action="store_true", help="禁用声音报警")
    parser.add_argument("--alarm-cooldown", type=float, default=1.5,
                        help="声音报警最小间隔（秒），默认 1.5")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
