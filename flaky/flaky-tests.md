# Flaky Test Report - 2026-02-28

## Summary

- **Flaky tests**: 17
- **Newly flaky** (last 7 days): 17
- **Total tests analyzed**: 43428
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...ensorDicts::test_cast_device[0-False-device_cast0-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...ensorDicts::test_cast_device[0-False-device_cast1-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...TensorDicts::test_cast_device[0-True-device_cast1-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...ensorDicts::test_cast_device[1-False-device_cast0-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...ensorDicts::test_cast_device[1-False-device_cast1-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...TensorDicts::test_cast_device[1-True-device_cast1-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...ensorDicts::test_cast_device[4-False-device_cast0-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...ensorDicts::test_cast_device[4-False-device_cast1-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...TensorDicts::test_cast_device[4-True-device_cast1-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...orDicts::test_cast_device[None-False-device_cast0-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...orDicts::test_cast_device[None-False-device_cast1-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...sorDicts::test_cast_device[None-True-device_cast1-td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...est_tensordict.TestTensorDicts::test_equal_tensor[td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...nsordict.TestTensorDicts::test_pin_memory[False-0-td_with_unbatched-device33]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...ict.TestTensorDicts::test_pin_memory[False-cuda:0-td_with_unbatched-device33]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...stTensorDicts::test_pin_memory[False-device_cast2-td_with_unbatched-device33]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |
| `...ict.TestTensorDicts::test_to_device_dtype_inplace[td_with_unbatched-device34]` **NEW** | 72.7% (16/22) | 16 | 0.55 | 2026-02-27 |


### Newly Flaky

- `test.test_tensordict.TestTensorDicts::test_cast_device[0-False-device_cast0-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[0-False-device_cast1-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[0-True-device_cast1-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[1-False-device_cast0-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[1-False-device_cast1-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[1-True-device_cast1-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[4-False-device_cast0-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[4-False-device_cast1-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[4-True-device_cast1-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[None-False-device_cast0-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[None-False-device_cast1-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_cast_device[None-True-device_cast1-td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_equal_tensor[td_with_unbatched-device34]`
- `test.test_tensordict.TestTensorDicts::test_pin_memory[False-0-td_with_unbatched-device33]`
- `test.test_tensordict.TestTensorDicts::test_pin_memory[False-cuda:0-td_with_unbatched-device33]`
- `test.test_tensordict.TestTensorDicts::test_pin_memory[False-device_cast2-td_with_unbatched-device33]`
- `test.test_tensordict.TestTensorDicts::test_to_device_dtype_inplace[td_with_unbatched-device34]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 80%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-02-28T06:14:05.611023+00:00*