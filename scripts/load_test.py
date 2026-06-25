#!/usr/bin/env python3
"""
Load test SBD API — đo throughput và latency từ 100 đến 1000 CCU.

Usage:
    python scripts/load_test.py
    python scripts/load_test.py --url http://192.168.3.92:8000 --max-ccu 1000 --step 100
    python scripts/load_test.py --endpoint /predict/batch --batch-size 8
"""

import asyncio
import argparse
import json
import random
import statistics
import time
from collections import defaultdict

import aiohttp

# ── Realistic Vietnamese callbot samples ──────────────────────────────────────
SAMPLE_TEXTS = [
    "Tôi muốn đặt lịch khám bệnh",
    "Cho tôi hỏi phí dịch vụ là bao nhiêu",
    "Vâng cảm ơn anh",
    "Tôi muốn",
    "Ờ thì là",
    "Số điện thoại của tôi là 0912 345 678",
    "Dạ anh ơi em gọi để hỏi về gói cước",
    "ờ anh thấy cũng ok đó em à",
    "tôi không hài lòng với dịch vụ này",
    "Mạng nhà tôi bị mất kết nối từ sáng đến giờ",
    "Để anh nghĩ thêm đã",
    "Gửi thông tin qua zalo cho anh nhé",
    "Tôi muốn hủy đơn hàng đó",
    "Thì ra là vậy",
    "à không cần đâu",
    "Hóa đơn tháng này sai so với thực tế sử dụng",
    "Cho tôi nói chuyện với quản lý",
    "ừ thì tôi đồng ý với mức giá đó",
    "Tôi cần",
    "dạ em thấy dịch vụ cũng ổn lắm",
    "không cần đâu em ơi",
    "Tôi đã gọi 3 lần rồi mà vẫn chưa được giải quyết",
    "ờ thôi được rồi",
    "em hỏi chút bên em có hỗ trợ tích hợp CRM không",
    "nói chung là bên mình có hỗ trợ template ZNS",
    "Anh không có nhu cầu",
    "giả sử như bên mình có hỗ trợ báo cáo realtime",
    "tôi cho 8 điểm",
    "Thôi không cần đâu em ơi",
    "dạ để em kiểm tra lại đã nghen",
]


async def single_request(session: aiohttp.ClientSession, url: str, text: str) -> dict:
    """Send one request, return timing and status."""
    payload = {"text": text}
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            body = await resp.json(content_type=None)
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "ok": resp.status == 200,
                "status": resp.status,
                "latency_ms": elapsed,
                "inference_ms": body.get("latency", {}).get("inference_ms") if isinstance(body.get("latency"), dict) else None,
            }
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"ok": False, "status": 0, "latency_ms": elapsed, "inference_ms": None, "error": str(e)}


async def run_ccu_level(url: str, n: int) -> dict:
    """Fire n concurrent requests simultaneously, return stats."""
    texts = [random.choice(SAMPLE_TEXTS) for _ in range(n)]

    connector = aiohttp.TCPConnector(limit=n + 10, limit_per_host=n + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup connection pool
        t_start = time.perf_counter()
        tasks = [single_request(session, url, t) for t in texts]
        results = await asyncio.gather(*tasks)
        wall_ms = (time.perf_counter() - t_start) * 1000

    ok      = [r for r in results if r["ok"]]
    failed  = [r for r in results if not r["ok"]]
    latencies = [r["latency_ms"] for r in ok]
    infer_ms  = [r["inference_ms"] for r in ok if r["inference_ms"] is not None]

    def pct(arr, p):
        return round(sorted(arr)[int(len(arr) * p / 100)], 1) if arr else 0

    return {
        "ccu":          n,
        "success":      len(ok),
        "failed":       len(failed),
        "success_rate": round(100 * len(ok) / n, 1),
        "wall_ms":      round(wall_ms, 1),
        "throughput":   round(n / (wall_ms / 1000), 1),
        "latency": {
            "min":  round(min(latencies), 1) if latencies else 0,
            "avg":  round(statistics.mean(latencies), 1) if latencies else 0,
            "p50":  pct(latencies, 50),
            "p90":  pct(latencies, 90),
            "p95":  pct(latencies, 95),
            "p99":  pct(latencies, 99),
            "max":  round(max(latencies), 1) if latencies else 0,
        },
        "inference_avg_ms": round(statistics.mean(infer_ms), 1) if infer_ms else None,
    }


def print_header():
    print(f"\n{'═'*100}")
    print(f" SBD Load Test — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*100}")
    print(f"{'CCU':>6}  {'OK':>5}  {'Fail':>5}  {'Rate%':>6}  {'Wall(ms)':>9}  "
          f"{'RPS':>7}  {'Avg':>7}  {'P50':>7}  {'P90':>7}  {'P95':>7}  {'P99':>7}  {'Max':>7}  {'Infer':>7}")
    print(f"{'─'*100}")


def print_row(r: dict):
    L = r["latency"]
    infer = f"{r['inference_avg_ms']:.1f}" if r["inference_avg_ms"] else "  —  "
    status = "✓" if r["success_rate"] >= 99 else ("~" if r["success_rate"] >= 90 else "✗")
    print(
        f"{status}{r['ccu']:>5}  {r['success']:>5}  {r['failed']:>5}  "
        f"{r['success_rate']:>5.1f}%  {r['wall_ms']:>9.1f}  "
        f"{r['throughput']:>7.1f}  {L['avg']:>7.1f}  {L['p50']:>7.1f}  "
        f"{L['p90']:>7.1f}  {L['p95']:>7.1f}  {L['p99']:>7.1f}  {L['max']:>7.1f}  "
        f"{infer:>7}"
    )


async def main(args):
    base_url  = args.url.rstrip("/")
    endpoint  = args.endpoint
    url       = base_url + endpoint
    ccu_steps = list(range(args.step, args.max_ccu + 1, args.step))

    # Health check
    async with aiohttp.ClientSession() as s:
        try:
            async with s.get(base_url + "/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                info = await r.json(content_type=None)
                print(f"\n Server: {base_url}")
                print(f" Device: {info.get('device','?')}  |  threshold={info.get('threshold','?')}")
                print(f" Endpoint: {endpoint}  |  CCU steps: {ccu_steps}")
        except Exception as e:
            print(f"\n[ERROR] Cannot reach server: {e}")
            return

    print_header()
    all_results = []

    for ccu in ccu_steps:
        result = await run_ccu_level(url, ccu)
        all_results.append(result)
        print_row(result)
        await asyncio.sleep(1)  # brief pause between levels

    # Summary
    print(f"{'─'*100}")
    print(f"\n LEGEND: ✓ = ≥99% success  ~ = 90-99%  ✗ = <90%")
    print(f" Wall(ms) = total wall-clock for all {ccu_steps[-1]} concurrent requests")
    print(f" RPS = requests/second throughput at this CCU level")
    print(f" Infer = avg model inference_ms reported by server (GPU time only)\n")

    # Find degradation point
    degraded = [r for r in all_results if r["success_rate"] < 99 or r["latency"]["p99"] > 5000]
    if degraded:
        first = degraded[0]
        print(f" ⚠️  Bắt đầu suy giảm tại CCU={first['ccu']} "
              f"(success={first['success_rate']}%, p99={first['latency']['p99']}ms)")
    else:
        print(f" ✅ Stable ở tất cả CCU levels (max {max(ccu_steps)} CCU)")

    # JSON output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f" Results saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",       default="http://192.168.3.92:8000")
    parser.add_argument("--endpoint",  default="/predict")
    parser.add_argument("--max-ccu",   type=int, default=1000)
    parser.add_argument("--step",      type=int, default=100)
    parser.add_argument("--output",    default=None, help="Save JSON results to file")
    args = parser.parse_args()
    asyncio.run(main(args))
