# Getting started, step by step

From the sign-up to the first push. `teamNN`, `22NN`, `<server>` and the
password are on your team's slip; `<owner>/<repo>` is your team's GitHub
repository.

## Before the session

1. **The sheet (by 16 November).** Put your team - two or three students -
   into the [teams' sheet](https://docs.google.com/spreadsheets/d/1cYZvlUaCBKgaTFwQWd6Pbs244kXuAcpy26M5ovdcdc4/edit): the machine (`team02` ...
   `team15`), the team's name, the members' names, their GitHub usernames.
2. **The repository (by 20 November).** One of you creates a **private**
   repository on GitHub (*New repository*, *Private*, nothing else ticked),
   adds the other members and **`ChmelJ`** under *Settings -> Collaborators*,
   and puts the link into the sheet. The others accept the invitation from
   the e-mail.
3. **An SSH client on your laptop.** Windows 10/11: PowerShell or the
   Windows Terminal has `ssh`; macOS and Linux: the terminal. Any other
   client works too.

## In the session

4. **Log in.**

   ```bash
   ssh -p 22NN teamNN@<server>
   ```

   The first time, answer `yes` to the question about the host's
   fingerprint, then type the password (nothing shows while you type). All
   of you log in to the same account at the same time; each terminal is
   independent.

5. **Look around.**

   ```bash
   nvidia-smi                 # your GPU
   ls /data/export            # the data, read-only
   ls -A /data/template       # the skeleton of your repository
   ```

6. **GitHub on the machine** - one of you, once for the whole team:

   ```bash
   gh auth login
   ```

   Answer *GitHub.com*, *HTTPS*, *Yes* (authenticate Git), *Login with a web
   browser*. It prints a one-time code: open <https://github.com/login/device>
   on your laptop, enter the code, confirm. Everyone on the machine now
   pushes through this login.

7. **Your repository in `/work`, with the skeleton.**

   ```bash
   cd /work
   gh repo clone <owner>/<repo>
   cd <repo>
   cp -r /data/template/. .
   git config user.name "Your Name"           # who you are in this repository
   git config user.email "you@example.com"
   git add -A
   git commit -m "Start from the project template"
   git push -u origin HEAD
   ```

   To sign a commit with your own name on the shared account:
   `git commit --author="Your Name <you@example.com>" -m "..."`.

8. **The Python environment** - with `uv` (installed). Inside the
   repository:

   ```bash
   uv init --bare             # once: a pyproject.toml
   uv add <the packages you need>
   uv run python your_script.py
   ```

   Commit `pyproject.toml` and `uv.lock`: we rebuild the environment from
   them with `uv sync`.

9. **Long runs** survive a lost connection inside `tmux`:

   ```bash
   tmux new -s train          # start; run your training inside
   # Ctrl-b, then d           # detach - it keeps running
   tmux attach -t train       # back to it, also after logging in again
   ```

## Rules of the machine

- The export in `/data/export` is read-only. Do not commit images or other
  data: `.gitignore` keeps out every `*.jpg` / `*.png` (except `figures/*.png`)
  and any folder named `data/` - put tables you want to commit elsewhere
  (e.g. `splits/`, `results/`). Large outputs go into `/work`, outside the
  repository.
- No `sudo`; everything installs into your account with `uv`.
- **Commit and push as you work.** A hand-in whose work arrives in one commit
  at the end gets **0 points**: the history is part of the hand-in.
- **Push often.** At **17:00** we clone your repository: what is on its
  default branch then is your hand-in. The machine is deleted the next day.
- **Before the last push, check the hand-in** in the root of your
  repository:

  ```bash
  python3 check_handin.py          # runs your Setup and Predict on the smoke file, then checks the rest
  ```

  It lists what is required and still missing or malformed (the README
  commands, predict on the twelve smoke images, `data_table.csv`,
  `expected.csv`, the search, the report, what is not committed or not
  pushed). We run the same checks at 17:00.

## When something goes wrong

| what | what to do |
|---|---|
| `Permission denied` when logging in | the password is case-sensitive; check `-p 22NN` and `teamNN` on the slip |
| `Connection refused` or a timeout | check the server and the port on the slip; then ask the teacher |
| the connection dropped during training | log in again, `tmux attach -t train` |
| `git push` is rejected | `git pull --rebase`, then `git push` again (someone else pushed first) |
| `CUDA out of memory` | a smaller batch or image size; `nvidia-smi` shows who is using the GPU |
| the machine is broken | tell the teacher: there is a spare |

You may delete this file from your repository.
