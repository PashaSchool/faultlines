# Benchmark harness

Ground-truth test set for the Faultlines detection engine.
See `SPRINT_6_PLAN.md` for design and metric definitions.

## Layout

```
benchmarks/
  _template/                   ← copy this for each new repo
    expected_features.yaml
    expected_flows.yaml
    expected_attribution_sample.yaml
  formbricks/
    expected_features.yaml
    expected_flows.yaml
    expected_attribution_sample.yaml
  documenso/
    ...
```

## Adding a new repo

1. `cp -r benchmarks/_template benchmarks/<repo>`
2. Read the repo's README + product docs. List the user-facing
   features in `expected_features.yaml`. Aim for the buyer-visible
   surface (~10-25 entries on most apps).
3. For apps, list the top user flows in `expected_flows.yaml`.
   For libraries, leave the file empty or delete it.
4. Generate a 50-file attribution sample:

   ```sh
   cd /path/to/repo
   git ls-files \
     | grep -Ev '^(node_modules|dist|build|vendor)/|\.lock$|/__tests__/|\.test\.|\.spec\.' \
     | shuf -n 50 > /tmp/sample.txt
   ```

   Hand-label each in `expected_attribution_sample.yaml`.

5. Run a scan with the full Sprint 1-5 stack to a JSON file, then
   score:

   ```sh
   ANTHROPIC_API_KEY=sk-ant-... faultline analyze /path/to/repo \
       --llm --tool-use --dedup --sub-decompose --tool-flows --critique \
       -o /tmp/<repo>-scan.json

   python scripts/run_benchmark.py \
       --repo <repo> --scan /tmp/<repo>-scan.json \
       --report benchmarks/results/<repo>.md
   ```

## Metrics

| metric | range | direction |
|---|---|---|
| feature recall | 0-1 | higher better |
| feature precision | 0-1 | higher better |
| flow recall | 0-1 | higher better; library skips |
| attribution accuracy | 0-1 | higher better; n/a if no sample |
| generic-name rate | 0-1 | **lower** better |

## CI integration (deferred)

Once the harness is stable across 5+ repos, wire it into CI so PRs
that regress any metric by more than 2pp on any repo block merge.
