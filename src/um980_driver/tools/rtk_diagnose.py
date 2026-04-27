#!/usr/bin/env python3
"""RTK 固定解稳定性诊断工具。

连接 UM980 串口，持续监测 GGA 和 RTKSTATUSA，分析固定解丢失的根本原因。
运行一段时间后按 Ctrl+C 退出，脚本会打印完整的诊断报告。

用法:
    python3 rtk_diagnose.py                          # 默认 /dev/ttyrtk 115200
    python3 rtk_diagnose.py --config ../config/rtk_client.ini
    python3 rtk_diagnose.py --serial-device /dev/ttyUSB0 --baud 115200
    python3 rtk_diagnose.py --duration 300            # 只运行 5 分钟
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import selectors
import signal
import sys
import termios
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

BAUD_MAP = {
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
    230400: termios.B230400,
    460800: termios.B460800,
    921600: termios.B921600,
}

FIX_LABELS = {
    "0": "无定位",
    "1": "单点定位",
    "2": "DGPS",
    "4": "RTK固定解",
    "5": "RTK浮动解",
    "6": "航位推算",
}

# GGA fix=4 表示固定解
FIXED_QUALITY = "4"

# 差分龄期阈值(秒) — 超过此值认为差分数据过期
AGE_WARN_THRESHOLD = 5.0
AGE_CRITICAL_THRESHOLD = 10.0

# 卫星数阈值 — 低于此值 RTK 很难固定
SATS_WARN_THRESHOLD = 10
SATS_CRITICAL_THRESHOLD = 6

# HDOP 阈值
HDOP_WARN_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

_log_file = None


def log(message: str, level: str = "INFO") -> None:
    now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{now_str}] [{level}] {message}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


# ---------------------------------------------------------------------------
# GGA 解析
# ---------------------------------------------------------------------------

@dataclass
class GGASnapshot:
    """一条 GGA 语句的关键字段。"""
    timestamp: float           # monotonic 时间
    wall_time: str             # 可读时间
    raw: str                   # 原始语句
    utc_time: str = ""         # GGA UTC 时间
    fix_quality: str = "0"     # 0/1/2/4/5/6
    sats: int = 0
    hdop: float = 99.0
    altitude: float = 0.0
    diff_age: float = -1.0     # 差分龄期，-1 表示无数据
    diff_station: str = ""

    @property
    def fix_label(self) -> str:
        return FIX_LABELS.get(self.fix_quality, f"未知({self.fix_quality})")

    @property
    def is_fixed(self) -> bool:
        return self.fix_quality == FIXED_QUALITY

    @property
    def has_diff(self) -> bool:
        return self.fix_quality in ("2", "4", "5")


def parse_gga(line: bytes) -> Optional[GGASnapshot]:
    text = line.strip().decode("ascii", errors="ignore")
    if not text.startswith("$") or "GGA" not in text:
        return None
    parts = text.split(",")
    if len(parts) < 10:
        return None

    def safe_float(s: str, default: float = 0.0) -> float:
        try:
            return float(s) if s else default
        except ValueError:
            return default

    def safe_int(s: str, default: int = 0) -> int:
        try:
            return int(s) if s else default
        except ValueError:
            return default

    age = -1.0
    if len(parts) > 13 and parts[13]:
        age = safe_float(parts[13], -1.0)

    station = ""
    if len(parts) > 14:
        station = parts[14].split("*", 1)[0]

    return GGASnapshot(
        timestamp=time.monotonic(),
        wall_time=datetime.now().strftime("%H:%M:%S.%f")[:-3],
        raw=text,
        utc_time=parts[1] if len(parts) > 1 else "",
        fix_quality=parts[6] if parts[6] else "0",
        sats=safe_int(parts[7]),
        hdop=safe_float(parts[8], 99.0),
        altitude=safe_float(parts[9]),
        diff_age=age,
        diff_station=station,
    )


# ---------------------------------------------------------------------------
# 状态变化事件
# ---------------------------------------------------------------------------

@dataclass
class FixTransition:
    """记录一次 fix 状态变化。"""
    wall_time: str
    mono_time: float
    from_fix: str
    to_fix: str
    duration_in_prev: float   # 在前一个状态持续了多久(秒)
    sats_at_change: int
    hdop_at_change: float
    age_at_change: float      # 变化瞬间的差分龄期
    probable_cause: str = ""  # 诊断推测的原因


# ---------------------------------------------------------------------------
# 诊断统计
# ---------------------------------------------------------------------------

@dataclass
class DiagStats:
    """运行期间的诊断统计。"""
    start_time: float = 0.0
    start_wall: str = ""
    total_gga: int = 0
    total_rtkstatus: int = 0

    # Fix 状态分布(秒)
    time_in_fix: dict = field(default_factory=lambda: {
        "0": 0.0, "1": 0.0, "2": 0.0, "4": 0.0, "5": 0.0, "6": 0.0,
    })

    # 状态转换记录
    transitions: list = field(default_factory=list)

    # 差分龄期记录
    age_samples: list = field(default_factory=list)  # (mono_time, age)
    age_missing_count: int = 0
    age_warn_count: int = 0       # > AGE_WARN_THRESHOLD
    age_critical_count: int = 0   # > AGE_CRITICAL_THRESHOLD

    # 卫星数记录
    sats_samples: list = field(default_factory=list)  # (mono_time, sats)
    sats_warn_count: int = 0
    sats_critical_count: int = 0

    # HDOP 记录
    hdop_warn_count: int = 0

    # 固定解段
    fix_sessions: list = field(default_factory=list)  # (start, end, duration)
    current_fix_start: float = 0.0

    # 丢失原因分类
    loss_reasons: dict = field(default_factory=lambda: {
        "差分数据中断/过期": 0,
        "卫星数不足": 0,
        "HDOP过大": 0,
        "原因不明": 0,
    })


# ---------------------------------------------------------------------------
# 串口
# ---------------------------------------------------------------------------

def configure_serial(fd: int, baud: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = BAUD_MAP[baud]
    attrs[5] = BAUD_MAP[baud]
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)


def open_serial(path: str, baud: int) -> int:
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
    configure_serial(fd, baud)
    return fd


def send_serial(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


# ---------------------------------------------------------------------------
# 配置 & 参数
# ---------------------------------------------------------------------------

def read_ini_config(path: str) -> dict:
    parser = configparser.ConfigParser()
    if not parser.read(path, encoding="utf-8"):
        raise RuntimeError(f"Config file not found: {path}")
    values = {}
    if parser.has_option("SYSTEM", "gps_serial"):
        values["serial_device"] = parser.get("SYSTEM", "gps_serial").strip()
    if parser.has_option("SYSTEM", "gps_rate"):
        values["baud"] = parser.getint("SYSTEM", "gps_rate")
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RTK 固定解稳定性诊断工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", help="ini 配置文件路径")
    p.add_argument("--serial-device", help="串口设备 (默认 /dev/ttyrtk)")
    p.add_argument("--baud", type=int, choices=sorted(BAUD_MAP), help="波特率")
    p.add_argument("--duration", type=float, default=0,
                   help="运行时长(秒), 0=一直运行直到 Ctrl+C")
    p.add_argument("--log-file", default="",
                   help="日志输出文件 (默认: rtk_diag_<时间>.log)")
    p.add_argument("--rtkstatus-interval", type=float, default=5.0,
                   help="RTKSTATUSA 查询间隔(秒), 0=不查询")
    p.add_argument("--no-log-file", action="store_true",
                   help="不输出日志文件")
    return p


def resolve_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = {}
    if args.config:
        args.config = os.path.abspath(args.config)
        cfg = read_ini_config(args.config)
    defaults = {"serial_device": "/dev/ttyrtk", "baud": 115200}
    for f in ("serial_device", "baud"):
        if getattr(args, f) is None and f in cfg:
            setattr(args, f, cfg[f])
        if getattr(args, f) is None and f in defaults:
            setattr(args, f, defaults[f])
    return args


# ---------------------------------------------------------------------------
# 诊断引擎
# ---------------------------------------------------------------------------

def diagnose_fix_loss(prev: GGASnapshot, curr: GGASnapshot) -> str:
    """当从固定解变为非固定解时，推测原因。"""
    reasons = []

    # 差分龄期检查
    if curr.diff_age < 0 or curr.diff_age > AGE_WARN_THRESHOLD:
        reasons.append("差分数据中断/过期")
    elif prev.diff_age >= 0 and curr.diff_age > prev.diff_age + 2:
        reasons.append("差分数据中断/过期")

    # 卫星数检查
    if curr.sats < SATS_CRITICAL_THRESHOLD:
        reasons.append("卫星数不足")
    elif prev.sats - curr.sats >= 3:
        reasons.append("卫星数不足")

    # HDOP 检查
    if curr.hdop > HDOP_WARN_THRESHOLD:
        reasons.append("HDOP过大")

    if not reasons:
        # 如果 age 都正常, 卫星也正常, 但就是丢了
        if curr.diff_age >= 0 and curr.diff_age < AGE_WARN_THRESHOLD and curr.sats >= SATS_WARN_THRESHOLD:
            reasons.append("原因不明")
        elif curr.diff_age < 0:
            reasons.append("差分数据中断/过期")
        else:
            reasons.append("原因不明")

    return reasons[0] if len(reasons) == 1 else " + ".join(reasons)


def print_report(stats: DiagStats) -> None:
    """打印最终诊断报告。"""
    elapsed = time.monotonic() - stats.start_time
    if elapsed < 1:
        log("运行时间太短，无法生成有效诊断报告", "WARN")
        return

    sep = "=" * 70
    log(sep)
    log("                    RTK 固定解稳定性诊断报告")
    log(sep)
    log(f"开始时间:  {stats.start_wall}")
    log(f"运行时长:  {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    log(f"GGA 语句:  {stats.total_gga} 条")
    log(f"RTKSTATUS: {stats.total_rtkstatus} 条")

    # --- Fix 状态分布 ---
    log("")
    log("-" * 70)
    log("  [1] Fix 状态时间分布")
    log("-" * 70)
    for code in ("4", "5", "1", "2", "0", "6"):
        t = stats.time_in_fix.get(code, 0.0)
        pct = (t / elapsed * 100) if elapsed > 0 else 0
        bar = "█" * int(pct / 2)
        label = FIX_LABELS.get(code, code)
        if t > 0:
            log(f"  {label:10s} (fix={code}): {t:7.1f}s  ({pct:5.1f}%)  {bar}")

    fix_time = stats.time_in_fix.get("4", 0.0)
    fix_rate = (fix_time / elapsed * 100) if elapsed > 0 else 0
    log(f"\n  ★ 固定解率: {fix_rate:.1f}%")

    if fix_rate > 95:
        log("    → 固定解非常稳定 ✅")
    elif fix_rate > 80:
        log("    → 固定解基本可用，偶尔丢失 ⚠️")
    elif fix_rate > 50:
        log("    → 固定解不稳定，需要排查 ❌")
    else:
        log("    → 固定解严重不稳定，存在根本性问题 🚨")

    # --- 固定解段统计 ---
    if stats.fix_sessions:
        durations = [s[2] for s in stats.fix_sessions]
        log("")
        log("-" * 70)
        log("  [2] 固定解持续时长统计")
        log("-" * 70)
        log(f"  获得固定解次数:  {len(durations)}")
        log(f"  最短持续:        {min(durations):.1f} 秒")
        log(f"  最长持续:        {max(durations):.1f} 秒")
        log(f"  平均持续:        {sum(durations)/len(durations):.1f} 秒")
        log(f"  中位持续:        {sorted(durations)[len(durations)//2]:.1f} 秒")

    # --- 状态转换记录 ---
    fix_losses = [t for t in stats.transitions if t.from_fix == FIXED_QUALITY]
    fix_gains = [t for t in stats.transitions if t.to_fix == FIXED_QUALITY]

    log("")
    log("-" * 70)
    log("  [3] Fix 状态转换")
    log("-" * 70)
    log(f"  总状态变化次数:  {len(stats.transitions)}")
    log(f"  丢失固定解次数:  {len(fix_losses)}")
    log(f"  获得固定解次数:  {len(fix_gains)}")

    if fix_losses:
        log("\n  每次丢失固定解的详情:")
        log(f"  {'时间':12s} {'持续':>8s} {'→状态':10s} {'卫星':>4s} {'HDOP':>6s} {'龄期':>6s}  {'推测原因'}")
        log(f"  {'----':12s} {'----':>8s} {'-----':10s} {'---':>4s} {'----':>6s} {'----':>6s}  {'--------'}")
        for t in fix_losses:
            to_label = FIX_LABELS.get(t.to_fix, t.to_fix)
            age_str = f"{t.age_at_change:.1f}" if t.age_at_change >= 0 else "无"
            log(f"  {t.wall_time:12s} {t.duration_in_prev:7.1f}s  {to_label:10s} {t.sats_at_change:4d} {t.hdop_at_change:6.1f} {age_str:>6s}  {t.probable_cause}")

    # --- 丢失原因分类 ---
    if any(v > 0 for v in stats.loss_reasons.values()):
        log("")
        log("-" * 70)
        log("  [4] 固定解丢失原因分类")
        log("-" * 70)
        total_losses = sum(stats.loss_reasons.values())
        for reason, count in sorted(stats.loss_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / total_losses * 100
                bar = "█" * int(pct / 2)
                log(f"  {reason:20s}: {count:3d} 次 ({pct:5.1f}%)  {bar}")

    # --- 差分龄期 ---
    log("")
    log("-" * 70)
    log("  [5] 差分龄期 (Differential Age) 分析")
    log("-" * 70)
    if stats.age_samples:
        ages = [a for _, a in stats.age_samples if a >= 0]
        if ages:
            log(f"  有效样本数:    {len(ages)}")
            log(f"  最小龄期:      {min(ages):.1f} 秒")
            log(f"  最大龄期:      {max(ages):.1f} 秒")
            log(f"  平均龄期:      {sum(ages)/len(ages):.1f} 秒")
            log(f"  龄期>5秒次数:  {stats.age_warn_count}  (差分数据较旧)")
            log(f"  龄期>10秒次数: {stats.age_critical_count}  (差分数据严重过期)")
        else:
            log("  ⚠️  所有 GGA 语句中均无差分龄期字段!")
        log(f"  龄期缺失次数:  {stats.age_missing_count}  (GGA 无 age 字段)")

        if stats.age_missing_count > stats.total_gga * 0.3:
            log("\n  🚨 大量 GGA 缺少差分龄期 → 差分数据链路可能频繁中断!")
        if stats.age_critical_count > 5:
            log("\n  🚨 差分龄期频繁超过 10 秒 → 单片机转发或基站链路存在严重延迟/丢包!")
    else:
        log("  无 GGA 数据")

    # --- 卫星数 ---
    log("")
    log("-" * 70)
    log("  [6] 卫星数分析")
    log("-" * 70)
    if stats.sats_samples:
        sats_vals = [s for _, s in stats.sats_samples]
        log(f"  最小卫星数:   {min(sats_vals)}")
        log(f"  最大卫星数:   {max(sats_vals)}")
        log(f"  平均卫星数:   {sum(sats_vals)/len(sats_vals):.1f}")
        log(f"  卫星<10次数:  {stats.sats_warn_count}  (RTK 性能可能降低)")
        log(f"  卫星<6次数:   {stats.sats_critical_count}  (RTK 难以固定)")
        if stats.sats_critical_count > 3:
            log("\n  🚨 卫星频繁不足 → 可能是天线安装位置或果园遮挡问题!")
    else:
        log("  无卫星数据")

    # --- 综合诊断结论 ---
    log("")
    log(sep)
    log("  [★] 综合诊断结论")
    log(sep)

    if stats.total_gga == 0:
        log("  🚨 未收到任何 GGA 数据，请检查串口连接和 UM980 配置!")
        log(sep)
        return

    # 判断主要原因
    diff_issues = stats.loss_reasons.get("差分数据中断/过期", 0)
    sat_issues = stats.loss_reasons.get("卫星数不足", 0)
    hdop_issues = stats.loss_reasons.get("HDOP过大", 0)
    unknown_issues = stats.loss_reasons.get("原因不明", 0)
    total_issues = diff_issues + sat_issues + hdop_issues + unknown_issues

    if total_issues == 0 and fix_rate > 95:
        log("  ✅ RTK 固定解稳定，未发现明显问题")
    else:
        if diff_issues > 0 and diff_issues >= sat_issues:
            log("  🔴 主要问题: 差分数据链路不稳定")
            log("")
            log("     这说明 UM980 没有持续收到新鲜的 RTCM 差分数据。")
            log("     在你的架构中 (果园基站 → 单片机转发 → UM980)，")
            log("     最可能的原因是:")
            log("")
            log("     1. 单片机串口转发丢包或延迟")
            log("        - 检查单片机的串口缓冲区大小是否够大 (RTCM33 数据量较大)")
            log("        - 检查转发逻辑是否有阻塞 (比如其他中断抢占串口)")
            log("        - 确认单片机的波特率设置正确")
            log("")
            log("     2. 基站到单片机的无线链路不稳定")
            log("        - 果园环境中树木遮挡和多径效应会影响无线信号")
            log("        - 检查通信距离是否在有效范围内")
            log("        - 尝试提高天线位置或增加信号增益")
            log("")
            log("     3. 基站自身问题")
            log("        - 基站是否持续输出 RTCM 数据")
            log("        - 基站电源是否稳定")
            log("")
            log("     ▶ 建议验证方法: 绕过单片机，直接将差分数据接入 UM980，")
            log("       如果稳定了，则确认是单片机转发环节的问题。")

        if sat_issues > 0 and sat_issues >= diff_issues:
            log("  🔴 主要问题: 卫星可见性不足")
            log("")
            log("     果园中果树遮挡天空导致可用卫星数下降。")
            log("")
            log("     建议:")
            log("     1. 将 GNSS 天线安装在尽可能高的位置 (高于树冠)")
            log("     2. 使用高增益天线")
            log("     3. 确保天线下方有良好的接地板")
            log("     4. 检查 UM980 是否启用了全系统 (GPS+BDS+GLO+GAL)")

        if unknown_issues > 0 and unknown_issues > diff_issues and unknown_issues > sat_issues:
            log("  🟡 多次丢失固定解但原因不明确")
            log("")
            log("     差分数据和卫星数看起来都正常，但 RTK 仍然丢失固定解。")
            log("     可能的原因:")
            log("     1. 多径干扰严重 (信号被周围金属/建筑物反射)")
            log("     2. 基站坐标精度不够")
            log("     3. 基站和移动站距离过远 (>15km)")
            log("     4. UM980 模糊度固定策略需要调整")

    log("")
    log(sep)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    global _log_file
    args = resolve_args(argv)
    stop = False

    def handle_signal(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # 日志文件
    if not args.no_log_file:
        log_path = args.log_file or f"rtk_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _log_file = open(log_path, "w", encoding="utf-8")
        log(f"日志将保存到: {os.path.abspath(log_path)}")

    stats = DiagStats()
    stats.start_time = time.monotonic()
    stats.start_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fd = -1
    selector = selectors.DefaultSelector()
    prev_gga: Optional[GGASnapshot] = None
    prev_fix: str = ""
    prev_fix_start: float = stats.start_time

    try:
        log(f"连接串口 {args.serial_device} @ {args.baud} ...")
        fd = open_serial(args.serial_device, args.baud)
        selector.register(fd, selectors.EVENT_READ)
        log("串口已打开，开始监测... (按 Ctrl+C 结束并查看诊断报告)")
        log("")

        rx_buffer = bytearray()
        next_rtkstatus_at = time.monotonic()
        last_print_at = time.monotonic()
        deadline = (stats.start_time + args.duration) if args.duration > 0 else 0

        while not stop:
            if deadline and time.monotonic() >= deadline:
                log(f"已达到设定运行时长 {args.duration:.0f} 秒，停止")
                break

            timeout = 0.2
            if args.rtkstatus_interval > 0:
                timeout = max(0.0, min(timeout, next_rtkstatus_at - time.monotonic()))

            events = selector.select(timeout)
            if events:
                incoming = os.read(fd, 4096)
                if incoming:
                    rx_buffer.extend(incoming.replace(b"\r", b"\n"))
                    while b"\n" in rx_buffer:
                        raw_line, _, rx_buffer = rx_buffer.partition(b"\n")
                        if not raw_line.strip():
                            continue

                        # --- 解析 GGA ---
                        gga = parse_gga(raw_line)
                        if gga:
                            stats.total_gga += 1

                            # 更新 fix 状态时间
                            now = time.monotonic()
                            if prev_fix and prev_fix in stats.time_in_fix:
                                stats.time_in_fix[prev_fix] += now - prev_fix_start

                            # 检测状态变化
                            if gga.fix_quality != prev_fix and prev_fix != "":
                                duration_in_prev = now - prev_fix_start

                                # 诊断原因
                                cause = ""
                                if prev_fix == FIXED_QUALITY and gga.fix_quality != FIXED_QUALITY:
                                    # 丢失固定解
                                    cause = diagnose_fix_loss(prev_gga, gga) if prev_gga else "原因不明"
                                    if cause in stats.loss_reasons:
                                        stats.loss_reasons[cause] += 1
                                    else:
                                        # 复合原因
                                        for r in cause.split(" + "):
                                            if r in stats.loss_reasons:
                                                stats.loss_reasons[r] += 1

                                    # 记录固定解段
                                    stats.fix_sessions.append((
                                        prev_fix_start, now, duration_in_prev
                                    ))

                                    log(f"❌ 固定解丢失! {gga.fix_label} | "
                                        f"持续了 {duration_in_prev:.1f}s | "
                                        f"卫星={gga.sats} HDOP={gga.hdop:.1f} "
                                        f"龄期={gga.diff_age:.1f if gga.diff_age >= 0 else '无'} | "
                                        f"推测: {cause}",
                                        "ALERT")

                                elif gga.fix_quality == FIXED_QUALITY:
                                    # 获得固定解
                                    from_label = FIX_LABELS.get(prev_fix, prev_fix)
                                    log(f"✅ 获得固定解! 从 {from_label} → RTK固定解 | "
                                        f"卫星={gga.sats} HDOP={gga.hdop:.1f} "
                                        f"龄期={gga.diff_age:.1f if gga.diff_age >= 0 else '无'}",
                                        "ALERT")
                                else:
                                    from_label = FIX_LABELS.get(prev_fix, prev_fix)
                                    log(f"⚡ 状态变化: {from_label} → {gga.fix_label} | "
                                        f"卫星={gga.sats} HDOP={gga.hdop:.1f}",
                                        "INFO")

                                trans = FixTransition(
                                    wall_time=gga.wall_time,
                                    mono_time=now,
                                    from_fix=prev_fix,
                                    to_fix=gga.fix_quality,
                                    duration_in_prev=duration_in_prev,
                                    sats_at_change=gga.sats,
                                    hdop_at_change=gga.hdop,
                                    age_at_change=gga.diff_age,
                                    probable_cause=cause,
                                )
                                stats.transitions.append(trans)
                                prev_fix_start = now

                            prev_fix = gga.fix_quality
                            if prev_fix == "" :
                                prev_fix_start = now

                            # 差分龄期监控
                            if gga.diff_age < 0:
                                stats.age_missing_count += 1
                            else:
                                stats.age_samples.append((now, gga.diff_age))
                                if gga.diff_age > AGE_CRITICAL_THRESHOLD:
                                    stats.age_critical_count += 1
                                    if stats.age_critical_count <= 5 or stats.age_critical_count % 10 == 0:
                                        log(f"⚠️  差分龄期过大: {gga.diff_age:.1f}s (>10s)", "WARN")
                                elif gga.diff_age > AGE_WARN_THRESHOLD:
                                    stats.age_warn_count += 1

                            # 卫星数监控
                            stats.sats_samples.append((now, gga.sats))
                            if gga.sats < SATS_CRITICAL_THRESHOLD:
                                stats.sats_critical_count += 1
                            elif gga.sats < SATS_WARN_THRESHOLD:
                                stats.sats_warn_count += 1

                            # HDOP 监控
                            if gga.hdop > HDOP_WARN_THRESHOLD:
                                stats.hdop_warn_count += 1

                            prev_gga = gga

                        # --- 解析 RTKSTATUSA ---
                        text = raw_line.strip().decode("ascii", errors="ignore")
                        if text.startswith("#RTKSTATUSA"):
                            stats.total_rtkstatus += 1
                            log(f"RTKSTATUS: {text[:120]}...")

            # 定时查询 RTKSTATUSA
            now = time.monotonic()
            if args.rtkstatus_interval > 0 and now >= next_rtkstatus_at:
                send_serial(fd, b"RTKSTATUSA\r\n")
                next_rtkstatus_at = now + args.rtkstatus_interval

            # 定时心跳
            if now - last_print_at >= 30.0:
                elapsed = now - stats.start_time
                fix_time = stats.time_in_fix.get("4", 0.0)
                current_label = FIX_LABELS.get(prev_fix, prev_fix) if prev_fix else "等待数据"
                fix_pct = (fix_time / elapsed * 100) if elapsed > 0 else 0
                age_str = f"{prev_gga.diff_age:.1f}s" if (prev_gga and prev_gga.diff_age >= 0) else "无"
                sats_str = str(prev_gga.sats) if prev_gga else "?"
                log(f"── 运行 {elapsed:.0f}s | 当前: {current_label} | "
                    f"固定率: {fix_pct:.1f}% | 卫星: {sats_str} | 龄期: {age_str} | "
                    f"丢解: {len([t for t in stats.transitions if t.from_fix == FIXED_QUALITY])} 次 ──")
                last_print_at = now

    except Exception as exc:
        log(f"运行异常: {exc}", "ERROR")
        import traceback
        traceback.print_exc()
    finally:
        # 记录最后一段 fix 状态的时间
        now = time.monotonic()
        if prev_fix and prev_fix in stats.time_in_fix:
            stats.time_in_fix[prev_fix] += now - prev_fix_start
        if prev_fix == FIXED_QUALITY:
            stats.fix_sessions.append((prev_fix_start, now, now - prev_fix_start))

        # 打印诊断报告
        log("")
        print_report(stats)

        try:
            selector.close()
        except Exception:
            pass
        if fd >= 0:
            try:
                os.close(fd)
            except Exception:
                pass
        if _log_file:
            log(f"完整日志已保存到文件")
            _log_file.close()
            _log_file = None

    return 0


if __name__ == "__main__":
    sys.exit(main())
