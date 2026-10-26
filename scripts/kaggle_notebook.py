"""Build the Kaggle submission notebook from kaggle/my_agent.py.

The notebook has the layout of the official starter kit's (scripts/build_notebook.py of
ARC-AGI-3-Kaggle-Starter): install the competition's offline wheels, write the agent to
/tmp/my_agent.py, run the framework against the gateway in the competition rerun, and write a
dummy submission.parquet otherwise. One cell is added: in the commit run only, the agent plays
two public games offline inside the same framework, so an error shows before a submission is
spent (the competition allows one a day). CPU only, no internet.

    python scripts/kaggle_notebook.py <owner>/<slug> "<title>" [NAME=VALUE ...] [--gpu=rtx6000]
        [--model=<owner>/<model>/<framework>/<variation>/<version>]   # writes kaggle/notebook/

NAME=VALUE pairs are set for the agent in both runs, e.g. NEO4_AGENT=model for the
model-guided agent alone (the commit check then asks only that it plays). --gpu picks the
accelerator (cpu, t4, p100, rtx6000) and --model attaches a Kaggle model; with
NEO4_ADVISOR=auto:<variation> the agent finds it under /kaggle/input and asks it about goals
(Phase 58). The commit check then also plays a set of public games without the solvers built
for them, once without the advisor and once with it, in the same process, and prints both.
"""
import json
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
kernel_id, title = sys.argv[1:3]
flags = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--"))
env = dict(a.split("=", 1) for a in sys.argv[3:] if not a.startswith("--"))
ACCELERATORS = {"cpu": ("none", False), "t4": ("nvidiaTeslaT4", True), "p100": ("nvidiaTeslaP100", True),
                "rtx6000": ("nvidiaRtx6000", True)}
accelerator, gpu = ACCELERATORS[flags.get("gpu", "cpu")]
models = [flags["model"]] if "model" in flags else []
out = ROOT / "kaggle" / "notebook"
out.mkdir(parents=True, exist_ok=True)
agent_body = (ROOT / "kaggle" / "my_agent.py").read_text()

SLIM_INIT = '''from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    'random': Random,
    'myagent': MyAgent,
}
'''


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [], "execution_count": None, "source": source}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


install_cell = code_cell(
    "!pip install --no-index --find-links \\\n"
    "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
    "    arc-agi python-dotenv"
)
write_agent_cell = code_cell("%%writefile /tmp/my_agent.py\n" + agent_body)
run_cell = code_cell(dedent(
    """\
    import os

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Wait for the gateway sidecar to be ready.
        !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
              --retry-max-time 600 http://gateway:8001/api/games

        # Copy the framework into a writable location.
        !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
               /kaggle/working/ARC-AGI-3-Agents

        # Drop our agent in as a framework template.
        !cp /tmp/my_agent.py \\
            /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

        # Register MyAgent in the framework's agent registry. We rewrite
        # __init__.py because the upstream version eagerly imports
        # templates with deps we don't ship (langgraph, smolagents, etc.).
        with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
            f.write(\"\"\"from typing import Type
    from dotenv import load_dotenv
    from .agent import Agent, Playback
    from .swarm import Swarm
    from .templates.random_agent import Random
    from .templates.my_agent import MyAgent

    load_dotenv()

    AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
        'random': Random,
        'myagent': MyAgent,
    }
    \"\"\")

        # Point the framework at the gateway sidecar.
        with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
            f.write(\"\"\"SCHEME=http
    HOST=gateway
    PORT=8001
    ARC_API_KEY=test-key-123
    ARC_BASE_URL=http://gateway:8001/
    OPERATION_MODE=online
    ENVIRONMENTS_DIR=
    RECORDINGS_DIR=/kaggle/working/server_recording
    \"\"\")

        # Run it. The gateway records every action and emits submission.parquet.
        !cd /kaggle/working/ARC-AGI-3-Agents && \\
            MPLBACKEND=agg ENV_VARS\\
            python main.py --agent myagent
    """
).replace("ENV_VARS", "".join(f"{k}={v} " for k, v in env.items())))
dummy_submission_cell = code_cell(dedent(
    """\
    import os
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Save-and-run-all (commit) mode: emit a dummy submission so the
        # commit succeeds. The real submission.parquet is produced by the
        # gateway during competition rerun.
        import pandas as pd
        submission = pd.DataFrame(
            data=[['1_0', '1', True, 1]],
            columns=['row_id', 'game_id', 'end_of_game', 'score'])
        submission.to_parquet('/kaggle/working/submission.parquet', index=False)
        submission.head()
    """
))
check_cell = code_cell(dedent(
    """\
    import os
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Commit run only: the agent plays two public games offline inside the framework the
        # rerun uses, so an error fails the commit here, before a submission is spent on it.
        import shutil, sys, time
        os.environ.update(ENV_DICT)
        import arc_agi
        COMP = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3'
        fw = '/tmp/check/ARC-AGI-3-Agents'
        shutil.copytree(COMP + '/ARC-AGI-3-Agents', fw, dirs_exist_ok=True)
        shutil.copy('/tmp/my_agent.py', fw + '/agents/templates/my_agent.py')
        with open(fw + '/agents/__init__.py', 'w') as f:
            f.write(SLIM_INIT)
        shutil.copytree(COMP + '/environment_files', '/tmp/check/environment_files', dirs_exist_ok=True)
        sys.path.insert(0, fw)
        from agents.templates.my_agent import MyAgent
        arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                             environments_dir='/tmp/check/environment_files',
                             recordings_dir='/tmp/check/recordings')
        MyAgent.MAX_ACTIONS = 300
        for game in ('ls20', 'ar25'):
            env = arc.make(game)
            agent = MyAgent(card_id='commit-check', game_id=game, agent_name='MyAgent.check.' + game,
                            ROOT_URL='http://localhost', record=False, arc_env=env, tags=['check'])
            t = time.time()
            agent.main()
            last = agent.frames[-1]
            print(f'{game}: {last.levels_completed} levels in {agent.action_counter} actions, '
                  f'{last.state.name}, {time.time() - t:.0f}s', flush=True)
            assert last.levels_completed >= 1 or os.environ.get('NEO4_AGENT') == 'model' and agent.action_counter > 0, game
        print('commit check passed')
    """
).replace("f.write(SLIM_INIT)", "f.write(" + repr(SLIM_INIT) + ")").replace("ENV_DICT", repr(env)))
advisor_cell = code_cell(dedent(
    """\
    import os
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN') and os.environ.get('NEO4_ADVISOR'):
        # Commit run only: the advisor on public games, each without the solvers built for it,
        # first without the advisor, then with it, in one process (the same hash order).
        import subprocess, time
        print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],
                             capture_output=True, text=True).stdout)
        for root, dirs, files in os.walk('/kaggle/input'):
            depth = root.count('/') - 2
            if depth <= 5 and 'config.json' in files:
                print('model dir:', root)
        import torch, transformers
        print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())
        os.environ['NEO4_HOLDOUT'] = '1'
        from agents.templates.my_agent import MyAgent
        import neo4.advisor as nadv
        adv = nadv.shared()
        print('advisor:', adv.name, adv.path, nadv.resolve(adv.path))
        MyAgent.MAX_ACTIONS = AB_ACTIONS
        results = {}
        for mode in ('off', 'on'):
            for game in AB_GAMES:
                env = arc.make(game)
                agent = MyAgent(card_id='ab-check', game_id=game, agent_name='MyAgent.ab.' + game,
                                ROOT_URL='http://localhost', record=False, arc_env=env, tags=['ab'])
                fb = getattr(agent._neo4, 'fallback', agent._neo4)
                if mode == 'off':
                    fb.advisor = None
                t = time.time()
                agent.main()
                ups, prev = [], 0
                for i, f in enumerate(agent.frames):
                    if f.levels_completed > prev:
                        ups.append(i)
                        prev = f.levels_completed
                results[(mode, game)] = ups
                print(f'{mode} {game}: levels at actions {ups}, {time.time() - t:.0f}s, '
                      f'advice {fb.stats.get("advice_queries", 0)}q/{fb.stats.get("advice_answers", 0)}a '
                      f'{fb.stats.get("advice_plans", 0)}p/{fb.stats.get("advice_clicks", 0)}c', flush=True)
                for line in getattr(fb, 'advice_log', [])[:6]:
                    print('    ', line[:240])
        print('advisor stats:', adv.stats)
        for game in AB_GAMES:
            print(f'{game}: off {results[("off", game)]} | on {results[("on", game)]}')
    """
).replace("AB_ACTIONS", flags.get("ab_actions", "1000")).replace(
    "AB_GAMES", repr(flags.get("ab_games", "vc33,lp85,r11l,sp80,tu93,ls20,ft09,cd82,su15,s5i5,m0r0,bp35,lf52,dc22,ar25").split(","))))

notebook = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "mimetype": "text/x-python", "file_extension": ".py",
                          "pygments_lexer": "ipython3"},
        "kaggle": {"accelerator": accelerator, "isInternetEnabled": False, "isGpuEnabled": gpu,
                   "language": "python", "sourceType": "notebook"},
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": [
        markdown_cell(
            "# Neo4 — ARC-AGI-3 submission\n\n"
            "The agent is `kaggle/my_agent.py` of https://github.com/ahmedmah987/neo4 (MIT-0), written to "
            "`/tmp/my_agent.py` below; the rest follows the official ARC-AGI-3-Kaggle-Starter notebook, "
            "with one more cell that checks the agent in the commit run."
        ),
        install_cell,
        write_agent_cell,
        run_cell,
        dummy_submission_cell,
        check_cell,
        advisor_cell,
    ],
}
(out / "submission.ipynb").write_text(json.dumps(notebook, indent=1))
meta = {
    "id": kernel_id,
    "title": title,
    "code_file": "submission.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": gpu,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": models,
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", out / "submission.ipynb", (out / "submission.ipynb").stat().st_size, "bytes")
