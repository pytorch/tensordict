# Flaky Test Report - 2026-08-24

## Summary

- **Flaky tests**: 132
- **Newly flaky** (last 7 days): 0
- **Total tests analyzed**: 45654
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_td-device1]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...ds.TestTensorDicts::test_squeeze_with_none_legacy[nested_tensorclass-device2]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...ods.TestTensorDicts::test_squeeze_with_none_legacy[nested_stacked_td-device3]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[stacked_td-device4]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[idx_td-device5]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td-device6]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `....test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td2-device7]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[memmap_td-device8]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[permute_td-device9]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...ethods.TestTensorDicts::test_squeeze_with_none_legacy[unsqueezed_td-device10]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `..._methods.TestTensorDicts::test_squeeze_with_none_legacy[squeezed_td-device11]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `..._methods.TestTensorDicts::test_squeeze_with_none_legacy[td_reset_bs-device12]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_h5-device13]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_params-device14]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...s.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor-device15]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...icts::test_squeeze_with_none_legacy[td_with_non_tensor_and_metadata-device16]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...ds.TestTensorDicts::test_squeeze_with_none_legacy[td_with_unbatched-device17]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[typed_td-device18]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...hods.TestTensorDicts::test_squeeze_with_none_legacy[nested_typed_td-device19]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...rdict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_td-device1]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...t_methods.TestTensorDicts::test_squeeze_with_none[nested_tensorclass-device2]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none[nested_stacked_td-device3]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...dict.test_methods.TestTensorDicts::test_squeeze_with_none[stacked_td-device4]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...nsordict.test_methods.TestTensorDicts::test_squeeze_with_none[idx_td-device5]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...nsordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td-device6]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...sordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td2-device7]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...rdict.test_methods.TestTensorDicts::test_squeeze_with_none[memmap_td-device8]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...dict.test_methods.TestTensorDicts::test_squeeze_with_none[permute_td-device9]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `....test_methods.TestTensorDicts::test_squeeze_with_none[unsqueezed_td-device10]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |
| `...ct.test_methods.TestTensorDicts::test_squeeze_with_none[squeezed_td-device11]` | 7.5% (16/214) | 16 | 0.15 | 2026-07-09 |


---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 80%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-24T06:19:53.447316+00:00*