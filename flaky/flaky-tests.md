# Flaky Test Report - 2026-09-21

## Summary

- **Confirmed flaky test families**: 1
- **Affected parameterized cases**: 1
- **Newly confirmed**: 0
- **Resolved since previous report**: 0
- **Total tests analyzed**: 45939
- **CI runs analyzed**: 28

---

## Flaky Tests

| Test | Environments | Confirmed revisions | Failures | Last failed |
|------|--------------|---------------------|----------|-------------|
| `test.tensordict.test_mp.TestMap::test_map_seed_single` | test-linux.yml / test-results-cpu-3.11 | [`5847b5f`](https://github.com/pytorch/tensordict/actions/runs/34741908072) | 3/28 | 2026-09-13 |


---

## Configuration

- Required evidence: fail and pass on the same commit and CI environment
- Active failure window: 14 days

---

*Generated at 2026-09-21T06:34:39.336485+00:00*