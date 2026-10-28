"""Push the submission notebook to Kaggle, follow its runs and submit it, through the official
`kaggle` library (pip install "kaggle>=2.2").

    python scripts/kaggle_submit.py push                 # kaggle/notebook/ -> a new version
    python scripts/kaggle_submit.py status <owner/slug>  # the latest run's state
    python scripts/kaggle_submit.py output <owner/slug> <dir>
    python scripts/kaggle_submit.py submit <owner/slug> <version> "<message>"
    python scripts/kaggle_submit.py submissions          # this account's submissions and scores

Credentials: an API token in KAGGLE_API_TOKEN (or ~/.kaggle/access_token) is used as the
library uses it. Where the token is instead added to every request by the network, as an
`Authorization: Bearer` header for www.kaggle.com and api.kaggle.com, set KAGGLE_USERNAME:
the library is then given a placeholder token, and its own token check, which sends the token
in a request body, is skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPETITION = "arc-prize-2026-arc-agi-3"


def api():
    from kaggle.api.kaggle_api_extended import AuthMethod, KaggleApi

    a = KaggleApi()
    token_file = Path.home() / ".kaggle" / "access_token"
    if os.environ.get("KAGGLE_API_TOKEN") or token_file.exists() or not os.environ.get("KAGGLE_USERNAME"):
        a.authenticate()
        return a
    a._load_config()
    a.config_values[a.CONFIG_NAME_TOKEN] = "placeholder"
    a.config_values[a.CONFIG_NAME_USER] = os.environ["KAGGLE_USERNAME"]
    a.config_values[a.CONFIG_NAME_AUTH_METHOD] = str(AuthMethod.ACCESS_TOKEN)
    a._authenticated = True
    return a


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, args, a = sys.argv[1], sys.argv[2:], api()
    if cmd == "push":
        r = a.kernels_push(str(ROOT / args[0]) if args else str(ROOT / "kaggle" / "notebook"))
        print("ref:", r.ref, "| version:", r.version_number, "| url:", r.url, "| error:", r.error)
    elif cmd == "status":
        r = a.kernels_status(args[0])
        print("status:", getattr(r, "status", r), "| failure:", getattr(r, "failure_message", ""))
    elif cmd == "output":
        files, _ = a.kernels_output(args[0], args[1], force=True)
        print("files:", files)
    elif cmd == "submit":
        r = a.competition_submit_code(file_name="submission.parquet", message=args[2], competition=COMPETITION,
                                      kernel=args[0], kernel_version=int(args[1]))
        print("submitted:", r)
    elif cmd == "submissions":
        for s in a.competition_submissions(COMPETITION) or []:
            print(s.ref, s.date, s.status, "| public:", s.public_score, "| private:", s.private_score,
                  "|", s.description, "|", s.error_description or "")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
