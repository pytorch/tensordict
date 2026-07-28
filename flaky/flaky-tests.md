# Flaky Test Report - 2026-07-28

## Summary

- **Flaky tests**: 132
- **Newly flaky** (last 7 days): 0
- **Total tests analyzed**: 90463
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...nsordict.test_generic.TestGeneric::test_consolidate_non_contiguous[0-device1]` | 33.3% (2/6) | 2 | 0.27 | 2026-07-16 |
| `...nsordict.test_generic.TestGeneric::test_consolidate_non_contiguous[2-device1]` | 33.3% (2/6) | 2 | 0.27 | 2026-07-16 |
| `...t_generic.TestGeneric::test_consolidate_non_contiguous_requires_grad[device1]` | 33.3% (2/6) | 2 | 0.27 | 2026-07-16 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_td-device1]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...ds.TestTensorDicts::test_squeeze_with_none_legacy[nested_tensorclass-device2]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...ods.TestTensorDicts::test_squeeze_with_none_legacy[nested_stacked_td-device3]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[stacked_td-device4]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[idx_td-device5]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td-device6]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `....test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td2-device7]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[memmap_td-device8]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[permute_td-device9]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...ethods.TestTensorDicts::test_squeeze_with_none_legacy[unsqueezed_td-device10]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `..._methods.TestTensorDicts::test_squeeze_with_none_legacy[squeezed_td-device11]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `..._methods.TestTensorDicts::test_squeeze_with_none_legacy[td_reset_bs-device12]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_h5-device13]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_params-device14]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...s.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor-device15]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...icts::test_squeeze_with_none_legacy[td_with_non_tensor_and_metadata-device16]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...ds.TestTensorDicts::test_squeeze_with_none_legacy[td_with_unbatched-device17]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[typed_td-device18]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...hods.TestTensorDicts::test_squeeze_with_none_legacy[nested_typed_td-device19]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...rdict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_td-device1]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...t_methods.TestTensorDicts::test_squeeze_with_none[nested_tensorclass-device2]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none[nested_stacked_td-device3]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...dict.test_methods.TestTensorDicts::test_squeeze_with_none[stacked_td-device4]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...nsordict.test_methods.TestTensorDicts::test_squeeze_with_none[idx_td-device5]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...nsordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td-device6]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...sordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td2-device7]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |
| `...rdict.test_methods.TestTensorDicts::test_squeeze_with_none[memmap_td-device8]` | 10.1% (16/158) | 16 | 0.20 | 2026-07-09 |


---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 80%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-07-28T07:00:18.690087+00:00*