# Flaky Test Report - 2026-11-12

## Summary

- **Confirmed flaky test families**: 1
- **Affected parameterized cases**: 1
- **Newly confirmed**: 0
- **Resolved since previous report**: 1
- **Total tests analyzed**: 79426
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Environments | Confirmed revisions | Failures | Last failed |
|------|--------------|---------------------|----------|-------------|
| `test.memmap.test_memmap::test_anonymous_file_descriptors[shape0]` | test-linux.yml / test-results-cpu-3.14t | [`767d51c`](https://github.com/pytorch/tensordict/actions/runs/42970208772) | 1/31 | 2026-11-11 |


## Resolved Since Previous Report

- `test.compile.test_compile.TestNN::test_prob_module_thread_local_interaction_type`

---

## Configuration

- Required evidence: fail and pass on the same commit and CI environment
- Active failure window: 14 days

---

*Generated at 2026-11-12T06:23:49.312816+00:00*