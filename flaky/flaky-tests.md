# Flaky Test Report - 2026-08-28

## Summary

- **Confirmed flaky test families**: 0
- **Affected parameterized cases**: 0
- **Newly confirmed**: 0
- **Resolved since previous report**: 16
- **Total tests analyzed**: 45663
- **CI runs analyzed**: 29

---

## No Flaky Tests Detected!

No test has recent fail/pass evidence on the same commit and CI environment.

## Resolved Since Previous Report

- `test.compile.test_compile.TestCompileNontensor::test_nontensor_no_device`
- `test.compile.test_compile.TestCompileNontensor::test_nontensor_no_device_no_batch_size`
- `test.compile.test_compile.TestCompileNontensor::test_nontensor_with_device`
- `test.compile.test_compile.TestCompileNontensor::test_nontensor_with_device_without_batch_size`
- `test.compile.test_compile.TestNN::test_dispatch_nontensor`
- `test.compile.test_compile.TestTCDefaultsCompile::test_concrete_default_applied_under_compile`
- `test.compile.test_compile.TestTCDefaultsCompile::test_default_factory_applied_under_compile`
- `test.compile.test_compile.TestTCDefaultsCompile::test_omitted_none_default_under_compile`
- `test.compile.test_compile.TestTCNonTensorInit::test_tc_nontensor_init_fullgraph`
- `test.compile.test_compile.TestTCNonTensorInit::test_tc_nontensor_init_roundtrip`
- `test.compile.test_compile.TestTCNonTensorInit::test_tc_nontensor_init_with_device`
- `test.compile.test_compile.TestTCPostInitCompile::test_post_init_runs_under_compile`
- `test.tensordict.test_generic.TestGeneric::test_consolidate_non_contiguous`
- `test.tensordict.test_generic.TestGeneric::test_consolidate_non_contiguous_requires_grad`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none`
- `test.tensordict.test_methods.TestTensorDicts::test_squeeze_with_none_legacy`

---

## Configuration

- Required evidence: fail and pass on the same commit and CI environment
- Active failure window: 14 days

---

*Generated at 2026-08-28T09:18:21.322262+00:00*