# Flaky Test Report - 2026-07-11

## Summary

- **Flaky tests**: 129
- **Newly flaky** (last 7 days): 129
- **Total tests analyzed**: 90304
- **CI runs analyzed**: 30

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `test.compile.test_compile.TestNN::test_dispatch_nontensor[None]` **NEW** | 14.3% (14/98) | 14 | 0.29 | 2026-07-09 |
| `test.compile.test_compile.TestNN::test_dispatch_nontensor[reduce-overhead]` **NEW** | 14.3% (14/98) | 14 | 0.29 | 2026-07-09 |
| `test.compile.test_compile.TestTCNonTensorInit::test_tc_nontensor_init_fullgraph` **NEW** | 14.3% (14/98) | 14 | 0.29 | 2026-07-09 |
| `test.compile.test_compile.TestTCNonTensorInit::test_tc_nontensor_init_roundtrip` **NEW** | 14.3% (14/98) | 14 | 0.29 | 2026-07-09 |
| `....compile.test_compile.TestTCNonTensorInit::test_tc_nontensor_init_with_device` **NEW** | 14.3% (14/98) | 14 | 0.29 | 2026-07-09 |
| `...compile.test_compile.TestTCPostInitCompile::test_post_init_runs_under_compile` **NEW** | 14.3% (14/98) | 14 | 0.29 | 2026-07-09 |
| `...st_compile.TestTCDefaultsCompile::test_concrete_default_applied_under_compile` **NEW** | 14.3% (14/98) | 14 | 0.29 | 2026-07-09 |
| `...est_compile.TestTCDefaultsCompile::test_default_factory_applied_under_compile` **NEW** | 14.3% (14/98) | 14 | 0.29 | 2026-07-09 |
| `...e.test_compile.TestTCDefaultsCompile::test_omitted_none_default_under_compile` **NEW** | 14.3% (14/98) | 14 | 0.29 | 2026-07-09 |
| `...rdict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td-device0]` **NEW** | 14.3% (20/140) | 20 | 0.29 | 2026-07-09 |
| `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td-device0]` **NEW** | 14.3% (20/140) | 20 | 0.29 | 2026-07-09 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_td-device1]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...ds.TestTensorDicts::test_squeeze_with_none_legacy[nested_tensorclass-device2]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...ods.TestTensorDicts::test_squeeze_with_none_legacy[nested_stacked_td-device3]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[stacked_td-device4]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[idx_td-device5]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td-device6]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `....test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td2-device7]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[memmap_td-device8]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[permute_td-device9]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...ethods.TestTensorDicts::test_squeeze_with_none_legacy[unsqueezed_td-device10]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `..._methods.TestTensorDicts::test_squeeze_with_none_legacy[squeezed_td-device11]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `..._methods.TestTensorDicts::test_squeeze_with_none_legacy[td_reset_bs-device12]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...t.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_h5-device13]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...st_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_params-device14]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...s.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor-device15]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...icts::test_squeeze_with_none_legacy[td_with_non_tensor_and_metadata-device16]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...ds.TestTensorDicts::test_squeeze_with_none_legacy[td_with_unbatched-device17]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...est_methods.TestTensorDicts::test_squeeze_with_none_legacy[typed_td-device18]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |
| `...hods.TestTensorDicts::test_squeeze_with_none_legacy[nested_typed_td-device19]` **NEW** | 14.3% (16/112) | 16 | 0.29 | 2026-07-09 |


### Newly Flaky

- `test.compile.test_compile.TestNN::test_dispatch_nontensor[None]`
- `test.compile.test_compile.TestNN::test_dispatch_nontensor[reduce-overhead]`
- `test.compile.test_compile.TestTCNonTensorInit::test_tc_nontensor_init_fullgraph`
- `test.compile.test_compile.TestTCNonTensorInit::test_tc_nontensor_init_roundtrip`
- `test.compile.test_compile.TestTCNonTensorInit::test_tc_nontensor_init_with_device`
- `test.compile.test_compile.TestTCPostInitCompile::test_post_init_runs_under_compile`
- `test.compile.test_compile.TestTCDefaultsCompile::test_concrete_default_applied_under_compile`
- `test.compile.test_compile.TestTCDefaultsCompile::test_default_factory_applied_under_compile`
- `test.compile.test_compile.TestTCDefaultsCompile::test_omitted_none_default_under_compile`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td-device0]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td-device0]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_td-device1]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_tensorclass-device2]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_stacked_td-device3]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[stacked_td-device4]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[idx_td-device5]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td-device6]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td2-device7]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[memmap_td-device8]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[permute_td-device9]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[unsqueezed_td-device10]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[squeezed_td-device11]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_reset_bs-device12]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_h5-device13]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_params-device14]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor-device15]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor_and_metadata-device16]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_with_unbatched-device17]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[typed_td-device18]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_typed_td-device19]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_td-device1]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_tensorclass-device2]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_stacked_td-device3]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[stacked_td-device4]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[idx_td-device5]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td-device6]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td2-device7]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[memmap_td-device8]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[permute_td-device9]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[unsqueezed_td-device10]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[squeezed_td-device11]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_reset_bs-device12]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_h5-device13]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_params-device14]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_with_non_tensor-device15]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_with_non_tensor_and_metadata-device16]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_with_unbatched-device17]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[typed_td-device18]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_typed_td-device19]`
- `test.compile.test_compile.TestCompileNontensor::test_nontensor_no_device_no_batch_size`
- `test.compile.test_compile.TestCompileNontensor::test_nontensor_no_device`
- `test.compile.test_compile.TestCompileNontensor::test_nontensor_with_device`
- `test.compile.test_compile.TestCompileNontensor::test_nontensor_with_device_without_batch_size`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td-device1]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_td-device2]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_td-device3]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_tensorclass-device4]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_tensorclass-device5]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_stacked_td-device6]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_stacked_td-device7]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[stacked_td-device8]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[stacked_td-device9]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[idx_td-device10]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[idx_td-device11]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td-device12]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td-device13]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td2-device14]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[sub_td2-device15]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[memmap_td-device16]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[permute_td-device17]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[permute_td-device18]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[unsqueezed_td-device19]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[unsqueezed_td-device20]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[squeezed_td-device21]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[squeezed_td-device22]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_reset_bs-device23]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_reset_bs-device24]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_h5-device25]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_h5-device26]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_params-device27]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_params-device28]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor-device29]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor-device30]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor_and_metadata-device31]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_with_non_tensor_and_metadata-device32]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_with_unbatched-device33]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[td_with_unbatched-device34]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[typed_td-device35]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[typed_td-device36]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_typed_td-device37]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy[nested_typed_td-device38]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td-device1]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_td-device2]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_td-device3]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_tensorclass-device4]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_tensorclass-device5]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_stacked_td-device6]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_stacked_td-device7]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[stacked_td-device8]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[stacked_td-device9]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[idx_td-device10]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[idx_td-device11]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td-device12]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td-device13]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td2-device14]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[sub_td2-device15]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[memmap_td-device16]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[permute_td-device17]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[permute_td-device18]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[unsqueezed_td-device19]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[unsqueezed_td-device20]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[squeezed_td-device21]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[squeezed_td-device22]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_reset_bs-device23]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_reset_bs-device24]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_h5-device25]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_h5-device26]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_params-device27]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_params-device28]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_with_non_tensor-device29]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_with_non_tensor-device30]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_with_non_tensor_and_metadata-device31]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_with_non_tensor_and_metadata-device32]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_with_unbatched-device33]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[td_with_unbatched-device34]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[typed_td-device35]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[typed_td-device36]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_typed_td-device37]`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none[nested_typed_td-device38]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 80%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-07-11T06:52:58.184226+00:00*