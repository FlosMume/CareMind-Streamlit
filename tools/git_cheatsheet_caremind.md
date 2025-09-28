# CareMind‑Streamlit Git Cheat Sheet (1‑pager)

> Branches you likely use: `demo-data` (working), `main` (stable).  
> Keep large/generated stuff out of Git: `chroma_store*/`, `*.sqlite3`, `.env`, `__pycache__/`.

---

## Daily loop
```bash
git status                     # What changed? Any conflicts?
git fetch origin               # Update remote refs
git switch demo-data           # Work on the feature branch
git pull --rebase origin demo-data   # Reapply your local commits on top
```

## Save work-in-progress (incl. untracked)
```bash
git stash push -u -m "wip: <note>"
git stash list
git stash pop                  # Reapply latest stash
```

## Merge vs Rebase (quick rules)
- **Prefer rebase** to keep linear history when syncing with `origin/demo-data`:
  ```bash
  git fetch origin
  git rebase origin/demo-data
  ```
- **Use merge** when combining a completed branch into another (team work, many commits):
  ```bash
  git switch demo-data
  git merge main
  ```

## Handling “push rejected (fetch first)”
```bash
git fetch origin
git rebase origin/demo-data     # or: git pull --rebase
git push origin demo-data
# If you intentionally rewrote history:
git push --force-with-lease origin demo-data
```

## In-progress merge/rebase conflicts
1) See conflicts
```bash
git status
git diff --name-only --diff-filter=U
```
2) Resolve each file: open and fix markers
```
<<<<<<< ours (your branch)
=======
>>>>>>> theirs (incoming)
```
3) Mark resolved + continue
```bash
git add requirements.txt app.py rag/retriever.py readme.md chroma_store_clean/.keep
git rebase --continue          # if rebasing
# or
git commit                     # if merging
```

## Abort a bad merge/rebase
```bash
git merge --abort
git rebase --abort
```

## Choose Ours/Theirs quickly (per file)
```bash
git checkout --ours requirements.txt
git checkout --theirs requirements.txt
git add requirements.txt
```

## Review history & differences
```bash
git log --oneline --graph --decorate --all --max-count=30
git diff                        # staged vs working tree
git diff HEAD~1..HEAD           # last commit vs previous
```

## Make a clean commit
```bash
git add -A
git commit -m "feat: <what changed>; rationale"
```

## Undo safely
```bash
git restore --staged <file>     # unstage
git restore <file>              # discard working changes
git revert <commit>             # make a new commit that reverts
```

## Bring one commit from another branch (cherry-pick)
```bash
git fetch origin
git cherry-pick <commit-sha>
```

## Repo hygiene for CareMind
Add/ensure `.gitignore` contains:
```
.env
chroma_store*/
chroma_store_clean/
chroma_store_quarantined_*/
chroma_store_sqlite_legacy/
*.sqlite3
__pycache__/
*.pyc
tools/
demo.py
fullRequirements.txt
CompareRequirements.py
```

## Typical requirements conflict pattern
- Pin Python for Streamlit Cloud; coordinate sqlite builds:
```txt
# example lines (adjust as needed)
python==3.10.18
pysqlite3-binary==0.5.3
```
When conflict appears in `requirements.txt`: decide the final version lines, delete markers, `git add`, then `git rebase --continue`.

## Quick sanity before push
```bash
git status && git log --oneline -5
git push origin demo-data
```