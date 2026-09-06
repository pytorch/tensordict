# Flaky Test Report - 2026-09-06

## Summary

- **Confirmed flaky test families**: 1
- **Affected parameterized cases**: 1
- **Newly confirmed**: 1
- **Resolved since previous report**: 0
- **Total tests analyzed**: 45666
- **CI runs analyzed**: 29

---

## Flaky Tests

| Test | Environments | Confirmed revisions | Failures | Last failed |
|------|--------------|---------------------|----------|-------------|
| `test.tensordict.test_mp.TestMap::test_map_seed_single` **NEW** | test-linux.yml / test-results-cpu-3.11 | [`0ded728`](https://github.com/pytorch/tensordict/actions/runs/33958973896) | 3/27 | 2026-09-05 |


### Newly Flaky

- `test.tensordict.test_mp.TestMap::test_map_seed_single`

---

## Configuration

- Required evidence: fail and pass on the same commit and CI environment
- Active failure window: 14 days

---

*Generated at 2026-09-06T06:24:21.446630+00:00*