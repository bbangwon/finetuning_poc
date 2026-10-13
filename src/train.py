"""LoRA 파인튜닝 실행 스크립트."""

import os
import subprocess
import sys
from pathlib import Path
from rich.console import Console

console = Console()

def run_training():
    project_root = Path(__file__).resolve().parent.parent
    config_file = project_root / "configs" / "lora_config.yaml"
    adapters_dir = project_root / "adapters"
    adapters_dir.mkdir(exist_ok=True)

    console.print("[bold green]🚀 Gemma 4 LoRA 파인튜닝 학습을 시작합니다...[/bold green]")
    console.print(f"[cyan]설정 파일:[/cyan] {config_file}")
    console.print(f"[cyan]어댑터 저장 위치:[/cyan] {adapters_dir}\n")

    cmd = [
        sys.executable,
        "-m", "mlx_lm.lora",
        "--config", str(config_file)
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="")

    process.stdout.close()
    return_code = process.wait()

    if return_code == 0:
        console.print("\n[bold green]✅ LoRA 파인튜닝이 성공적으로 완료되었습니다![/bold green]")
        # 어댑터 파일 크기 확인
        adapter_file = adapters_dir / "adapters.safetensors"
        if adapter_file.exists():
            size_mb = adapter_file.stat().st_size / (1024 * 1024)
            console.print(f"[bold yellow]📦 생성된 어댑터 가중치:[/bold yellow] {adapter_file} ([cyan]{size_mb:.2f} MB[/cyan])")
    else:
        console.print(f"\n[bold red]❌ 학습 중 오류 발생 (종료 코드: {return_code})[/bold red]")
        sys.exit(return_code)

if __name__ == "__main__":
    run_training()
