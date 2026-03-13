# Flaky Test Report - 2026-03-13

## Summary

- **Flaky tests**: 33
- **Newly flaky** (last 7 days): 0
- **Total tests analyzed**: 43518
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ensorDicts::test_cast_device[0-False-device_cast0-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...ensorDicts::test_cast_device[0-False-device_cast1-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...TensorDicts::test_cast_device[0-True-device_cast1-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...ensorDicts::test_cast_device[1-False-device_cast0-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...ensorDicts::test_cast_device[1-False-device_cast1-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...TensorDicts::test_cast_device[1-True-device_cast1-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...ensorDicts::test_cast_device[4-False-device_cast0-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...ensorDicts::test_cast_device[4-False-device_cast1-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...TensorDicts::test_cast_device[4-True-device_cast1-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...orDicts::test_cast_device[None-False-device_cast0-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...orDicts::test_cast_device[None-False-device_cast1-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...sorDicts::test_cast_device[None-True-device_cast1-td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...est_tensordict.TestTensorDicts::test_equal_tensor[td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...nsordict.TestTensorDicts::test_pin_memory[False-0-td_with_unbatched-device33]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...ict.TestTensorDicts::test_pin_memory[False-cuda:0-td_with_unbatched-device33]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...stTensorDicts::test_pin_memory[False-device_cast2-td_with_unbatched-device33]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...ict.TestTensorDicts::test_to_device_dtype_inplace[td_with_unbatched-device34]` | 31.4% (16/51) | 16 | 0.63 | 2026-02-27 |
| `...estTensorDictMP::test_chunksize_num_chunks[None-2-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `...estTensorDictMP::test_chunksize_num_chunks[4-None-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `...TensorDictMP::test_chunksize_num_chunks[None-None-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[-2-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[-1-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[0-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[1-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[2-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `test.test_tensordict.TestTensorDictMP::test_map[3-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `...ensordict.TestTensorDictMP::test_map_exception[-2-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `...ensordict.TestTensorDictMP::test_map_exception[-1-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[0-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |
| `...tensordict.TestTensorDictMP::test_map_exception[1-td_with_unbatched-device34]` | 15.7% (8/51) | 8 | 0.31 | 2026-03-12 |


---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 80%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-03-13T06:24:40.329485+00:00*