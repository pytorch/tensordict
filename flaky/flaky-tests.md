# Flaky Test Report - 2026-10-07

## Summary

- **Confirmed flaky test families**: 1
- **Affected parameterized cases**: 1
- **Newly confirmed**: 0
- **Resolved since previous report**: 0
- **Total tests analyzed**: 52927
- **CI runs analyzed**: 18

---

## Flaky Tests

| Test | Environments | Confirmed revisions | Failures | Last failed |
|------|--------------|---------------------|----------|-------------|
| `test.tensordict.test_mp.TestMap::test_map_seed_single` | test-linux.yml / test-results-stable-cpu-3.10 | [`16c48b3`](https://github.com/pytorch/tensordict/actions/runs/36738062670) | 1/19 | 2026-09-30 |


---

## Configuration

- Required evidence: fail and pass on the same commit and CI environment
- Active failure window: 14 days

---

*Generated at 2026-10-07T06:27:25.601264+00:00*