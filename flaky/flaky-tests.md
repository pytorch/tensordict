# Flaky Test Report - 2026-10-15

## Summary

- **Confirmed flaky test families**: 1
- **Affected parameterized cases**: 8
- **Newly confirmed**: 1
- **Resolved since previous report**: 0
- **Total tests analyzed**: 66116
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Environments | Confirmed revisions | Failures | Last failed |
|------|--------------|---------------------|----------|-------------|
| `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[add]` **NEW** | test-linux.yml / test-results-gpu-3.12 | [`4b178db`](https://github.com/pytorch/tensordict/actions/runs/38594439249), [`c0a887e`](https://github.com/pytorch/tensordict/actions/runs/38594483505) | 2/32 | 2026-10-14 |
| `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[add_]` **NEW** | test-linux.yml / test-results-gpu-3.12 | [`4b178db`](https://github.com/pytorch/tensordict/actions/runs/38594439249), [`c0a887e`](https://github.com/pytorch/tensordict/actions/runs/38594483505) | 2/32 | 2026-10-14 |
| `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[radd]` **NEW** | test-linux.yml / test-results-gpu-3.12 | [`4b178db`](https://github.com/pytorch/tensordict/actions/runs/38594439249), [`c0a887e`](https://github.com/pytorch/tensordict/actions/runs/38594483505) | 2/32 | 2026-10-14 |
| `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[rsub]` **NEW** | test-linux.yml / test-results-gpu-3.12 | [`4b178db`](https://github.com/pytorch/tensordict/actions/runs/38594439249), [`c0a887e`](https://github.com/pytorch/tensordict/actions/runs/38594483505) | 2/32 | 2026-10-14 |
| `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[sub-alpha]` **NEW** | test-linux.yml / test-results-gpu-3.12 | [`4b178db`](https://github.com/pytorch/tensordict/actions/runs/38594439249), [`c0a887e`](https://github.com/pytorch/tensordict/actions/runs/38594483505) | 2/32 | 2026-10-14 |
| `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[sub]` **NEW** | test-linux.yml / test-results-gpu-3.12 | [`4b178db`](https://github.com/pytorch/tensordict/actions/runs/38594439249), [`c0a887e`](https://github.com/pytorch/tensordict/actions/runs/38594483505) | 2/32 | 2026-10-14 |
| `....tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[sub_-alpha]` **NEW** | test-linux.yml / test-results-gpu-3.12 | [`4b178db`](https://github.com/pytorch/tensordict/actions/runs/38594439249), [`c0a887e`](https://github.com/pytorch/tensordict/actions/runs/38594483505) | 2/32 | 2026-10-14 |
| `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[sub_]` **NEW** | test-linux.yml / test-results-gpu-3.12 | [`4b178db`](https://github.com/pytorch/tensordict/actions/runs/38594439249), [`c0a887e`](https://github.com/pytorch/tensordict/actions/runs/38594483505) | 2/32 | 2026-10-14 |


### Newly Flaky

- `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[add]`
- `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[add_]`
- `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[radd]`
- `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[rsub]`
- `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[sub-alpha]`
- `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[sub]`
- `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[sub_-alpha]`
- `test.tensordict.test_methods::test_add_sub_cuda_scalar_tensor_no_sync[sub_]`

---

## Configuration

- Required evidence: fail and pass on the same commit and CI environment
- Active failure window: 14 days

---

*Generated at 2026-10-15T06:29:10.278132+00:00*