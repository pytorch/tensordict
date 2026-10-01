# Flaky Test Report - 2026-10-01

## Summary

- **Confirmed flaky test families**: 1
- **Affected parameterized cases**: 1
- **Newly confirmed**: 1
- **Resolved since previous report**: 0
- **Total tests analyzed**: 46528
- **CI runs analyzed**: 24

---

## Flaky Tests

| Test | Environments | Confirmed revisions | Failures | Last failed |
|------|--------------|---------------------|----------|-------------|
| `test.tensordict.test_mp.TestMap::test_map_seed_single` **NEW** | test-linux.yml / test-results-stable-cpu-3.10 | [`16c48b3`](https://github.com/pytorch/tensordict/actions/runs/36738062670) | 1/25 | 2026-09-30 |


### Newly Flaky

- `test.tensordict.test_mp.TestMap::test_map_seed_single`

---

## Configuration

- Required evidence: fail and pass on the same commit and CI environment
- Active failure window: 14 days

---

*Generated at 2026-10-01T06:29:07.289560+00:00*