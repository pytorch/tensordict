# Flaky Test Report - 2026-04-07

## Summary

- **Flaky tests**: 19
- **Newly flaky** (last 7 days): 0
- **Total tests analyzed**: 46017
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ensordict.TestTensorDicts::test_pin_memory[True-0-td_with_unbatched-device33]` | 38.9% (21/54) | 21 | 0.78 | 2026-03-04 |
| `...dict.TestTensorDicts::test_pin_memory[True-cuda:0-td_with_unbatched-device33]` | 38.9% (21/54) | 21 | 0.78 | 2026-03-04 |
| `...estTensorDicts::test_pin_memory[True-device_cast2-td_with_unbatched-device33]` | 38.9% (21/54) | 21 | 0.78 | 2026-03-04 |
| `...estTensorDictMP::test_chunksize_num_chunks[None-2-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `...estTensorDictMP::test_chunksize_num_chunks[4-None-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `...TensorDictMP::test_chunksize_num_chunks[None-None-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[-2-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[-1-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[0-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[1-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[2-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[3-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `...ensordict.TestTensorDictMP::test_map_exception[-2-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `...ensordict.TestTensorDictMP::test_map_exception[-1-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[0-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[1-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[2-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[3-td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |
| `...nsordict.TestTensorDictMP::test_sharing_locked_td[td_with_unbatched-device34]` | 14.8% (8/54) | 8 | 0.30 | 2026-03-12 |


---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 80%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-04-07T06:40:31.283212+00:00*