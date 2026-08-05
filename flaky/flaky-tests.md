# Flaky Test Report - 2026-08-05

## Summary

- **Flaky tests**: 132
- **Newly flaky** (last 7 days): 0
- **Total tests analyzed**: 90467
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_td-device1]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...ds.TestTensorDicts::test_squeeze_with_none_legacy[nested_tensorclass-device2]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...ods.TestTensorDicts::test_squeeze_with_none_legacy[nested_stacked_td-device3]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[stacked_td-device4]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[idx_td-device5]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td-device6]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `....test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td2-device7]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[memmap_td-device8]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[permute_td-device9]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...ethods.TestTensorDicts::test_squeeze_with_none_legacy[unsqueezed_td-device10]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `..._methods.TestTensorDicts::test_squeeze_with_none_legacy[squeezed_td-device11]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `..._methods.TestTensorDicts::test_squeeze_with_none_legacy[td_reset_bs-device12]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_h5-device13]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_params-device14]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...s.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor-device15]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...icts::test_squeeze_with_none_legacy[td_with_non_tensor_and_metadata-device16]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...ds.TestTensorDicts::test_squeeze_with_none_legacy[td_with_unbatched-device17]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[typed_td-device18]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...hods.TestTensorDicts::test_squeeze_with_none_legacy[nested_typed_td-device19]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...rdict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_td-device1]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...t_methods.TestTensorDicts::test_squeeze_with_none[nested_tensorclass-device2]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none[nested_stacked_td-device3]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...dict.test_methods.TestTensorDicts::test_squeeze_with_none[stacked_td-device4]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...nsordict.test_methods.TestTensorDicts::test_squeeze_with_none[idx_td-device5]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...nsordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td-device6]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...sordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td2-device7]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...rdict.test_methods.TestTensorDicts::test_squeeze_with_none[memmap_td-device8]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...dict.test_methods.TestTensorDicts::test_squeeze_with_none[permute_td-device9]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `....test_methods.TestTensorDicts::test_squeeze_with_none[unsqueezed_td-device10]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |
| `...ct.test_methods.TestTensorDicts::test_squeeze_with_none[squeezed_td-device11]` | 9.2% (16/174) | 16 | 0.18 | 2026-07-09 |


---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 80%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-05T07:01:11.291376+00:00*