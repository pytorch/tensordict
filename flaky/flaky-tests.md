# Flaky Test Report - 2026-03-19

## Summary

- **Flaky tests**: 36
- **Newly flaky** (last 7 days): 0
- **Total tests analyzed**: 46011
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ensordict.TestTensorDicts::test_pin_memory[True-0-td_with_unbatched-device33]` | 55.4% (31/56) | 31 | 0.89 | 2026-03-04 |
| `...dict.TestTensorDicts::test_pin_memory[True-cuda:0-td_with_unbatched-device33]` | 55.4% (31/56) | 31 | 0.89 | 2026-03-04 |
| `...estTensorDicts::test_pin_memory[True-device_cast2-td_with_unbatched-device33]` | 55.4% (31/56) | 31 | 0.89 | 2026-03-04 |
| `...estTensorDictMP::test_chunksize_num_chunks[None-2-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...estTensorDictMP::test_chunksize_num_chunks[4-None-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...TensorDictMP::test_chunksize_num_chunks[None-None-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[-2-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[-1-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[0-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[1-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[2-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[3-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...ensordict.TestTensorDictMP::test_map_exception[-2-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...ensordict.TestTensorDictMP::test_map_exception[-1-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[0-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[1-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[2-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[3-td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...nsordict.TestTensorDictMP::test_sharing_locked_td[td_with_unbatched-device34]` | 14.3% (8/56) | 8 | 0.29 | 2026-03-12 |
| `...ensorDicts::test_cast_device[0-False-device_cast0-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...ensorDicts::test_cast_device[0-False-device_cast1-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...TensorDicts::test_cast_device[0-True-device_cast1-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...ensorDicts::test_cast_device[1-False-device_cast0-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...ensorDicts::test_cast_device[1-False-device_cast1-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...TensorDicts::test_cast_device[1-True-device_cast1-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...ensorDicts::test_cast_device[4-False-device_cast0-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...ensorDicts::test_cast_device[4-False-device_cast1-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...TensorDicts::test_cast_device[4-True-device_cast1-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...orDicts::test_cast_device[None-False-device_cast0-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |
| `...orDicts::test_cast_device[None-False-device_cast1-td_with_unbatched-device34]` | 7.1% (4/56) | 4 | 0.11 | 2026-02-27 |


---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 80%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-03-19T06:29:29.188213+00:00*