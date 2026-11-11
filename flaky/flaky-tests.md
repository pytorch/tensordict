# Flaky Test Report - 2026-11-11

## Summary

- **Confirmed flaky test families**: 2
- **Affected parameterized cases**: 2
- **Newly confirmed**: 2
- **Resolved since previous report**: 0
- **Total tests analyzed**: 78632
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Environments | Confirmed revisions | Failures | Last failed |
|------|--------------|---------------------|----------|-------------|
| `...mpile.TestNN::test_prob_module_thread_local_interaction_type[reduce-overhead]` **NEW** | test-linux.yml / test-results-cpu-3.14t | [`f78b35c`](https://github.com/pytorch/tensordict/actions/runs/42953197253) | 1/31 | 2026-11-11 |
| `test.memmap.test_memmap::test_anonymous_file_descriptors[shape0]` **NEW** | test-linux.yml / test-results-cpu-3.14t | [`3b94861`](https://github.com/pytorch/tensordict/actions/runs/42967640435) | 1/3 | 2026-11-11 |


### Newly Flaky

- `test.compile.test_compile.TestNN::test_prob_module_thread_local_interaction_type[reduce-overhead]`
- `test.memmap.test_memmap::test_anonymous_file_descriptors[shape0]`

---

## Configuration

- Required evidence: fail and pass on the same commit and CI environment
- Active failure window: 14 days

---

*Generated at 2026-11-11T06:24:35.942536+00:00*