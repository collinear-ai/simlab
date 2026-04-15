from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from simlab.autoresearch.config import load_run_config
from simlab.cli.autoresearch import autoresearch


def test_autoresearch_init_writes_absolute_tasks_dir_when_out_is_elsewhere(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        env_dir = Path("environments") / "env1"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "env.yaml").write_text("name: env1\n", encoding="utf-8")

        (Path("bundle") / "tasks").mkdir(parents=True, exist_ok=True)

        out_path = Path("configs") / "run.toml"
        result = runner.invoke(
            autoresearch,
            [
                "init",
                "--env",
                "env1",
                "--tasks-dir",
                "bundle",
                "--task",
                "t1",
                "--out",
                str(out_path),
            ],
        )

        assert result.exit_code == 0, result.output
        cfg = load_run_config(out_path)
        assert Path(cfg.run.tasks_dir).is_absolute()
        assert Path(cfg.run.tasks_dir) == Path("bundle").resolve()
