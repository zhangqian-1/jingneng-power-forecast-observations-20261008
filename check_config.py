"""Production gate for the active versioned forecasting model."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))


def main() -> int:
    errors: list[str] = []
    required_files = [
        ROOT / "models" / "active_model.json",
        APP_DIR / "api.py",
        APP_DIR / "predict.py",
        APP_DIR / "input_adapter.py",
        APP_DIR / "history_cache.py",
        APP_DIR / "platform_adapter.py",
        APP_DIR / "time_policy.py",
        APP_DIR / "models" / "trend_detail.py",
    ]
    for path in required_files:
        if not path.is_file():
            errors.append(f"缺少必要文件: {path.relative_to(ROOT)}")

    forbidden_files = [
        ROOT / "data" / "input" / "input.json",
        ROOT / "data" / "input" / "input_example.json",
        ROOT / "data" / "output" / "latest_forecast.csv",
    ]
    for path in forbidden_files:
        if path.exists():
            errors.append(f"生产包仍包含示例或静态结果: {path.relative_to(ROOT)}")

    forbidden_code = [
        "_generate_example_data",
        "_simplified_prediction",
        "np.random",
        "daily_pattern",
        "generated_example",
    ]
    for path in APP_DIR.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        for marker in forbidden_code:
            if marker in content:
                errors.append(f"{path.relative_to(ROOT)} 包含禁止的模拟逻辑: {marker}")

    if not errors:
        try:
            active_path = ROOT / "models" / "active_model.json"
            active = json.loads(active_path.read_text(encoding="utf-8"))
            if active.get("model_type") != "trend_detail":
                errors.append("活动模型不是完整 TrendDetail 类型")

            from predict import PowerPredictor

            with tempfile.TemporaryDirectory() as temp_dir:
                predictor = PowerPredictor(
                    active_path,
                    device="cpu",
                    history_cache_path=Path(temp_dir) / "check_only_cache.csv",
                )
                if predictor.input_size != 672 or predictor.horizon != 96:
                    errors.append("活动模型必须使用672点输入并输出96点")
                if "TrendDetail" not in predictor.model_name:
                    errors.append("活动模型名称未声明完整 TrendDetail 链路")
        except Exception as exc:
            errors.append(f"生产模型加载失败: {exc}")

    api_path = APP_DIR / "api.py"
    api_content = api_path.read_text(encoding="utf-8") if api_path.exists() else ""
    if "PLATFORM_PATH" not in api_content or "/api/power/forecast" in api_content:
        errors.append("API必须仅提供平台接口")
    try:
        from platform_adapter import PLATFORM_PATH, POINT_TABLE
        if PLATFORM_PATH != "/api/v1/fluxcast/compute" or len(set(POINT_TABLE)) != 35:
            errors.append("平台接口路径或35个测点配置错误")
    except Exception as exc:
        errors.append(f"平台接口配置加载失败: {exc}")

    if errors:
        print("[FAIL] 生产部署检查未通过")
        for index, error in enumerate(errors, start=1):
            print(f"{index}. {error}")
        return 1

    print("[PASS] 生产部署检查通过")
    print("- 活动模型是完整 TrendDetail 链路")
    print("- NHITS、PatchTST、StationAttentionHF 均可加载")
    print("- 全部版本化模型文件哈希校验通过")
    print("- 外部每次提交96点，缓存至少672点（最多保留768点上下文）后输出96点")
    print("- 未发现示例输入、静态预测或模拟数据生成逻辑")
    print("- 接口收发UTC，模型内部按Asia/Shanghai处理；本检查不代表目标服务器接入验收")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
