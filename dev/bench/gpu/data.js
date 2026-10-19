window.BENCHMARK_DATA = {
  "lastUpdate": 1792447666804,
  "repoUrl": "https://github.com/pytorch/tensordict",
  "entries": {
    "GPU Benchmark Results": [
      {
        "commit": {
          "author": {
            "email": "peterdsharpe@gmail.com",
            "name": "Peter Sharpe",
            "username": "peterdsharpe"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "687b27a6113d08ba5fb141b85b573c1593abab65",
          "message": "[CI] Keep 8 days of benchmark charts instead of 3, and stop timing CPU-only benchmarks in the GPU job (#3058)",
          "timestamp": "2026-10-19T10:57:24-04:00",
          "tree_id": "3e44f3cdb322968149c2db6f09ca8de53f253a07",
          "url": "https://github.com/pytorch/tensordict/commit/687b27a6113d08ba5fb141b85b573c1593abab65"
        },
        "date": 1792423588368,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4470.43228366,
            "unit": "iter/sec",
            "range": "stddev: 0.00000630188884159",
            "extra": "mean: 223.69201378 usec\nrounds: 146"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3564.18282094,
            "unit": "iter/sec",
            "range": "stddev: 0.00000916992105703",
            "extra": "mean: 280.569221681 usec\nrounds: 3142"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2020.24037155,
            "unit": "iter/sec",
            "range": "stddev: 0.00000873192192768",
            "extra": "mean: 494.990603139 usec\nrounds: 1823"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4465.83799519,
            "unit": "iter/sec",
            "range": "stddev: 0.00000542168939997",
            "extra": "mean: 223.922139826 usec\nrounds: 3943"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3568.3023867,
            "unit": "iter/sec",
            "range": "stddev: 0.00000627384804462",
            "extra": "mean: 280.245307608 usec\nrounds: 3200"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1797.77620155,
            "unit": "iter/sec",
            "range": "stddev: 0.00000780672422989",
            "extra": "mean: 556.242762106 usec\nrounds: 1647"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2253.21371023,
            "unit": "iter/sec",
            "range": "stddev: 0.000011085359006",
            "extra": "mean: 443.810542897 usec\nrounds: 72"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 601.55145162,
            "unit": "iter/sec",
            "range": "stddev: 0.0000259461187788",
            "extra": "mean: 1.66236819362 msec\nrounds: 588"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 152.669904877,
            "unit": "iter/sec",
            "range": "stddev: 0.000055923411715",
            "extra": "mean: 6.55007940698 msec\nrounds: 150"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2276.84699854,
            "unit": "iter/sec",
            "range": "stddev: 0.00000870178047869",
            "extra": "mean: 439.203864222 usec\nrounds: 2113"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 601.288929795,
            "unit": "iter/sec",
            "range": "stddev: 0.0000170665221818",
            "extra": "mean: 1.66309398103 msec\nrounds: 590"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 151.73737279,
            "unit": "iter/sec",
            "range": "stddev: 0.0000444975479048",
            "extra": "mean: 6.59033421767 msec\nrounds: 152"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3078.62154481,
            "unit": "iter/sec",
            "range": "stddev: 0.00000978597064302",
            "extra": "mean: 324.820698304 usec\nrounds: 2352"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 831.190503008,
            "unit": "iter/sec",
            "range": "stddev: 0.0000143547710107",
            "extra": "mean: 1.20309363062 msec\nrounds: 806"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 212.963118417,
            "unit": "iter/sec",
            "range": "stddev: 0.0000317829521213",
            "extra": "mean: 4.69564874628 msec\nrounds: 205"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3063.05729635,
            "unit": "iter/sec",
            "range": "stddev: 0.00000721592268756",
            "extra": "mean: 326.47120287 usec\nrounds: 2888"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 823.992310233,
            "unit": "iter/sec",
            "range": "stddev: 0.0000195231765378",
            "extra": "mean: 1.21360355865 msec\nrounds: 809"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 211.311961024,
            "unit": "iter/sec",
            "range": "stddev: 0.0000265628839384",
            "extra": "mean: 4.73233978405 msec\nrounds: 208"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3557.89955232,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116431332024",
            "extra": "mean: 281.064708346 usec\nrounds: 3021"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 955.01510043,
            "unit": "iter/sec",
            "range": "stddev: 0.0000131434693438",
            "extra": "mean: 1.04710386207 msec\nrounds: 900"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 243.714303207,
            "unit": "iter/sec",
            "range": "stddev: 0.0000417080095182",
            "extra": "mean: 4.10316500443 msec\nrounds: 242"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3575.62543008,
            "unit": "iter/sec",
            "range": "stddev: 0.00000741497271142",
            "extra": "mean: 279.671352482 usec\nrounds: 3277"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 954.938665037,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155372265955",
            "extra": "mean: 1.04718767457 msec\nrounds: 925"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 243.291656544,
            "unit": "iter/sec",
            "range": "stddev: 0.0000444751796431",
            "extra": "mean: 4.11029302938 msec\nrounds: 242"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 189.365375095,
            "unit": "iter/sec",
            "range": "stddev: 0.0000511081212292",
            "extra": "mean: 5.28079644706 msec\nrounds: 186"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 100.135330412,
            "unit": "iter/sec",
            "range": "stddev: 0.0000242495706722",
            "extra": "mean: 9.98648524839 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 182.363221615,
            "unit": "iter/sec",
            "range": "stddev: 0.0000432894394538",
            "extra": "mean: 5.48356182318 msec\nrounds: 181"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 95.5113041644,
            "unit": "iter/sec",
            "range": "stddev: 0.0000373063575336",
            "extra": "mean: 10.4699648774 msec\nrounds: 90"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.0583492752,
            "unit": "iter/sec",
            "range": "stddev: 0.000423173305916",
            "extra": "mean: 76.579357691 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.49073973449,
            "unit": "iter/sec",
            "range": "stddev: 0.00296825403276",
            "extra": "mean: 154.065644427 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.5581526635,
            "unit": "iter/sec",
            "range": "stddev: 0.00187699411351",
            "extra": "mean: 79.6295463828 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.23863344932,
            "unit": "iter/sec",
            "range": "stddev: 0.00220676137504",
            "extra": "mean: 160.291513858 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.3108500131,
            "unit": "iter/sec",
            "range": "stddev: 0.000085361257135",
            "extra": "mean: 42.8984785813 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.7012340526,
            "unit": "iter/sec",
            "range": "stddev: 0.000270788660455",
            "extra": "mean: 85.4610714996 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.1954191722,
            "unit": "iter/sec",
            "range": "stddev: 0.000155323757307",
            "extra": "mean: 43.1119607098 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.6535498791,
            "unit": "iter/sec",
            "range": "stddev: 0.000104012071662",
            "extra": "mean: 85.8107624178 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.0494872259,
            "unit": "iter/sec",
            "range": "stddev: 0.000151162190909",
            "extra": "mean: 36.9692775189 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.4929228703,
            "unit": "iter/sec",
            "range": "stddev: 0.00036377007836",
            "extra": "mean: 74.112926429 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.1745480183,
            "unit": "iter/sec",
            "range": "stddev: 0.000192820047311",
            "extra": "mean: 36.7991401118 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.4993729467,
            "unit": "iter/sec",
            "range": "stddev: 0.000120407039313",
            "extra": "mean: 74.0775148557 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6511.45237473,
            "unit": "iter/sec",
            "range": "stddev: 0.0000175845493006",
            "extra": "mean: 153.575568468 usec\nrounds: 44"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6811.45349784,
            "unit": "iter/sec",
            "range": "stddev: 0.00000605573166107",
            "extra": "mean: 146.81154328 usec\nrounds: 1441"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6925.33608021,
            "unit": "iter/sec",
            "range": "stddev: 0.00000540948948428",
            "extra": "mean: 144.39732432 usec\nrounds: 5111"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 23147.3163773,
            "unit": "iter/sec",
            "range": "stddev: 0.00000259183241634",
            "extra": "mean: 43.2015523397 usec\nrounds: 11836"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 24391.1977355,
            "unit": "iter/sec",
            "range": "stddev: 0.00000248625651676",
            "extra": "mean: 40.9983966694 usec\nrounds: 12763"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10952.8459605,
            "unit": "iter/sec",
            "range": "stddev: 0.00000768147195917",
            "extra": "mean: 91.3004714579 usec\nrounds: 189"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 11245.1681921,
            "unit": "iter/sec",
            "range": "stddev: 0.00000383804475682",
            "extra": "mean: 88.9270825405 usec\nrounds: 8216"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 55.2853086261,
            "unit": "iter/sec",
            "range": "stddev: 0.0000682883505563",
            "extra": "mean: 18.087988018 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 92.4955451975,
            "unit": "iter/sec",
            "range": "stddev: 0.0122296192668",
            "extra": "mean: 10.8113314848 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 94.4640935876,
            "unit": "iter/sec",
            "range": "stddev: 0.000104071564002",
            "extra": "mean: 10.5860328726 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 92.4710143816,
            "unit": "iter/sec",
            "range": "stddev: 0.000100165676463",
            "extra": "mean: 10.8141995271 msec\nrounds: 91"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3861.7842919,
            "unit": "iter/sec",
            "range": "stddev: 0.00000553180534222",
            "extra": "mean: 258.947658495 usec\nrounds: 3956"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 293.819959746,
            "unit": "iter/sec",
            "range": "stddev: 0.00003005888441",
            "extra": "mean: 3.4034447519 msec\nrounds: 291"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 445.97863484,
            "unit": "iter/sec",
            "range": "stddev: 0.000204067076911",
            "extra": "mean: 2.24225987946 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 443.780322456,
            "unit": "iter/sec",
            "range": "stddev: 0.000263395222897",
            "extra": "mean: 2.25336714901 msec\nrounds: 409"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 474.65927547,
            "unit": "iter/sec",
            "range": "stddev: 0.0000554499816264",
            "extra": "mean: 2.10677437834 msec\nrounds: 442"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 636.566342277,
            "unit": "iter/sec",
            "range": "stddev: 0.000043705584611",
            "extra": "mean: 1.57092817133 msec\nrounds: 626"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 192.780085311,
            "unit": "iter/sec",
            "range": "stddev: 0.000047461631285",
            "extra": "mean: 5.18725779371 msec\nrounds: 194"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 117.435303328,
            "unit": "iter/sec",
            "range": "stddev: 0.000125824351352",
            "extra": "mean: 8.51532692183 msec\nrounds: 115"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 141.026364109,
            "unit": "iter/sec",
            "range": "stddev: 0.0000684843046503",
            "extra": "mean: 7.09087273373 msec\nrounds: 139"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 266.043261649,
            "unit": "iter/sec",
            "range": "stddev: 0.0000590036602198",
            "extra": "mean: 3.75878717545 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 112.132456511,
            "unit": "iter/sec",
            "range": "stddev: 0.0000981503337973",
            "extra": "mean: 8.91802454987 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.7739374808,
            "unit": "iter/sec",
            "range": "stddev: 0.00229355053232",
            "extra": "mean: 92.8165772057 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.54411710021,
            "unit": "iter/sec",
            "range": "stddev: 0.00219098965967",
            "extra": "mean: 220.064751402 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.495996592569,
            "unit": "iter/sec",
            "range": "stddev: 0.00842578930156",
            "extra": "mean: 2.0161428828 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 40.995967508,
            "unit": "iter/sec",
            "range": "stddev: 0.00106488683612",
            "extra": "mean: 24.3926430034 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.6711591012,
            "unit": "iter/sec",
            "range": "stddev: 0.0202376353984",
            "extra": "mean: 103.4002222 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.0856292719,
            "unit": "iter/sec",
            "range": "stddev: 0.00129728199797",
            "extra": "mean: 99.1509774001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 43.6910690482,
            "unit": "iter/sec",
            "range": "stddev: 0.000223067724545",
            "extra": "mean: 22.8879728005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4769.71584126,
            "unit": "iter/sec",
            "range": "stddev: 0.00000710306391662",
            "extra": "mean: 209.656095516 usec\nrounds: 1693"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8213.56686874,
            "unit": "iter/sec",
            "range": "stddev: 0.00000466210686092",
            "extra": "mean: 121.749784957 usec\nrounds: 6977"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6492.61937735,
            "unit": "iter/sec",
            "range": "stddev: 0.00000652699388096",
            "extra": "mean: 154.021041721 usec\nrounds: 2918"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5756.11411937,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157165375306",
            "extra": "mean: 173.7283138 usec\nrounds: 5374"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 21171.3685299,
            "unit": "iter/sec",
            "range": "stddev: 0.00000242042695447",
            "extra": "mean: 47.2336022392 usec\nrounds: 16161"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30337.4353344,
            "unit": "iter/sec",
            "range": "stddev: 0.00000130053754391",
            "extra": "mean: 32.9625754115 usec\nrounds: 26164"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 47876.8264617,
            "unit": "iter/sec",
            "range": "stddev: 0.00000433527273015",
            "extra": "mean: 20.8869316098 usec\nrounds: 38080"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15724.7236807,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113468963449",
            "extra": "mean: 63.5941222438 usec\nrounds: 14925"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3739.77836563,
            "unit": "iter/sec",
            "range": "stddev: 0.00000643521474941",
            "extra": "mean: 267.395525144 usec\nrounds: 1310"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5338.48686307,
            "unit": "iter/sec",
            "range": "stddev: 0.0000048406216839",
            "extra": "mean: 187.318996122 usec\nrounds: 4601"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5870.9212691,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102574957334",
            "extra": "mean: 170.331018619 usec\nrounds: 3309"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16167.4165127,
            "unit": "iter/sec",
            "range": "stddev: 0.00000599189482388",
            "extra": "mean: 61.8528012322 usec\nrounds: 13300"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4063.89075858,
            "unit": "iter/sec",
            "range": "stddev: 0.0000102128847526",
            "extra": "mean: 246.069606543 usec\nrounds: 1378"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1904.96465346,
            "unit": "iter/sec",
            "range": "stddev: 0.0000188304115192",
            "extra": "mean: 524.944123339 usec\nrounds: 1807"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4338.49208614,
            "unit": "iter/sec",
            "range": "stddev: 0.00000841396568961",
            "extra": "mean: 230.494830956 usec\nrounds: 4031"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3701.42352018,
            "unit": "iter/sec",
            "range": "stddev: 0.00000824520276944",
            "extra": "mean: 270.166327778 usec\nrounds: 1342"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 13418.4312103,
            "unit": "iter/sec",
            "range": "stddev: 0.00000342046125852",
            "extra": "mean: 74.5243601378 usec\nrounds: 11968"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5910.16711937,
            "unit": "iter/sec",
            "range": "stddev: 0.00000769071657803",
            "extra": "mean: 169.1999532 usec\nrounds: 3287"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2199.798846,
            "unit": "iter/sec",
            "range": "stddev: 0.0000223950808068",
            "extra": "mean: 454.58701909 usec\nrounds: 2160"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4106.43650907,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128390658691",
            "extra": "mean: 243.520141561 usec\nrounds: 1386"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 31417.1129004,
            "unit": "iter/sec",
            "range": "stddev: 0.00000392311365504",
            "extra": "mean: 31.8297866254 usec\nrounds: 22163"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53813.5548878,
            "unit": "iter/sec",
            "range": "stddev: 0.00000194362797797",
            "extra": "mean: 18.5826786966 usec\nrounds: 48031"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 43005.2670896,
            "unit": "iter/sec",
            "range": "stddev: 0.00000523883640371",
            "extra": "mean: 23.2529656871 usec\nrounds: 28556"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19637.2226623,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117115384006",
            "extra": "mean: 50.9236981825 usec\nrounds: 15870"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4132.45481901,
            "unit": "iter/sec",
            "range": "stddev: 0.000210615726165",
            "extra": "mean: 241.986916687 usec\nrounds: 96"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 307.01919958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000632937683131",
            "extra": "mean: 3.25712529174 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 4969.7082293,
            "unit": "iter/sec",
            "range": "stddev: 0.000228334035072",
            "extra": "mean: 201.219056303 usec\nrounds: 89"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 356.484471609,
            "unit": "iter/sec",
            "range": "stddev: 0.0000515317822846",
            "extra": "mean: 2.80517127572 msec\nrounds: 355"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6277.56698178,
            "unit": "iter/sec",
            "range": "stddev: 0.00000729176349321",
            "extra": "mean: 159.297384305 usec\nrounds: 3872"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 19053.5144972,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120385626311",
            "extra": "mean: 52.4837556948 usec\nrounds: 14771"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7463.78255527,
            "unit": "iter/sec",
            "range": "stddev: 0.00000854293331984",
            "extra": "mean: 133.980323327 usec\nrounds: 4503"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 20955.7840576,
            "unit": "iter/sec",
            "range": "stddev: 0.0000122638737662",
            "extra": "mean: 47.7195220782 usec\nrounds: 17376"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7378.31120173,
            "unit": "iter/sec",
            "range": "stddev: 0.00000518607764941",
            "extra": "mean: 135.532369489 usec\nrounds: 6655"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21342.7765963,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112865199937",
            "extra": "mean: 46.8542598235 usec\nrounds: 17841"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11686.5221206,
            "unit": "iter/sec",
            "range": "stddev: 0.00000487710647041",
            "extra": "mean: 85.568656755 usec\nrounds: 9861"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 44337.9478119,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112127434056",
            "extra": "mean: 22.5540434177 usec\nrounds: 34806"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15207.3257028,
            "unit": "iter/sec",
            "range": "stddev: 0.0000139484488012",
            "extra": "mean: 65.7577814496 usec\nrounds: 13024"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 47016.3673527,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108916562272",
            "extra": "mean: 21.2691889294 usec\nrounds: 36403"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15178.9056655,
            "unit": "iter/sec",
            "range": "stddev: 0.0000046512071293",
            "extra": "mean: 65.8809022231 usec\nrounds: 13465"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 45938.3212571,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115227407269",
            "extra": "mean: 21.7683183154 usec\nrounds: 36376"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11653.4107769,
            "unit": "iter/sec",
            "range": "stddev: 0.00000424656353222",
            "extra": "mean: 85.8117867076 usec\nrounds: 10059"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 45065.7451918,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108596208825",
            "extra": "mean: 22.1898028257 usec\nrounds: 36089"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15549.2668742,
            "unit": "iter/sec",
            "range": "stddev: 0.00000344138030167",
            "extra": "mean: 64.3117137349 usec\nrounds: 13275"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 45918.6389985,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108172001508",
            "extra": "mean: 21.7776489419 usec\nrounds: 36166"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15384.388927,
            "unit": "iter/sec",
            "range": "stddev: 0.00000319240936968",
            "extra": "mean: 65.0009567976 usec\nrounds: 13727"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 46967.676685,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105649668243",
            "extra": "mean: 21.2912383703 usec\nrounds: 37538"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 34349.8783489,
            "unit": "iter/sec",
            "range": "stddev: 0.00000199293003121",
            "extra": "mean: 29.1121846152 usec\nrounds: 25119"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 7011.46015004,
            "unit": "iter/sec",
            "range": "stddev: 0.00000838597993526",
            "extra": "mean: 142.623644519 usec\nrounds: 4261"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2718.93249524,
            "unit": "iter/sec",
            "range": "stddev: 0.0000136948947525",
            "extra": "mean: 367.791404071 usec\nrounds: 2594"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6524.40892867,
            "unit": "iter/sec",
            "range": "stddev: 0.00000803504357561",
            "extra": "mean: 153.270589096 usec\nrounds: 3415"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6052.19394198,
            "unit": "iter/sec",
            "range": "stddev: 0.00000555926497807",
            "extra": "mean: 165.229338251 usec\nrounds: 5330"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5814.96808944,
            "unit": "iter/sec",
            "range": "stddev: 0.00000652677716712",
            "extra": "mean: 171.969989279 usec\nrounds: 3827"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5710.27443255,
            "unit": "iter/sec",
            "range": "stddev: 0.00000700873079844",
            "extra": "mean: 175.122931798 usec\nrounds: 5584"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5375.48954928,
            "unit": "iter/sec",
            "range": "stddev: 0.00000613847515856",
            "extra": "mean: 186.029568253 usec\nrounds: 3612"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 109568.302119,
            "unit": "iter/sec",
            "range": "stddev: 6.66588349849e-7",
            "extra": "mean: 9.12672717076 usec\nrounds: 84460"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 44384.6657735,
            "unit": "iter/sec",
            "range": "stddev: 0.00000196830922514",
            "extra": "mean: 22.5303037112 usec\nrounds: 24414"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65455.9870381,
            "unit": "iter/sec",
            "range": "stddev: 0.00000126067327276",
            "extra": "mean: 15.2774413046 usec\nrounds: 55553"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 35958.3978669,
            "unit": "iter/sec",
            "range": "stddev: 0.00000262387972023",
            "extra": "mean: 27.8099153277 usec\nrounds: 23223"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 37160.0463339,
            "unit": "iter/sec",
            "range": "stddev: 0.00000176224491696",
            "extra": "mean: 26.9106230658 usec\nrounds: 33635"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 26335.6520709,
            "unit": "iter/sec",
            "range": "stddev: 0.00000278854643129",
            "extra": "mean: 37.9713400416 usec\nrounds: 18087"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 38391.0937516,
            "unit": "iter/sec",
            "range": "stddev: 0.00000159708269678",
            "extra": "mean: 26.0477080041 usec\nrounds: 29797"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8071.62702526,
            "unit": "iter/sec",
            "range": "stddev: 0.00000706224300495",
            "extra": "mean: 123.890759183 usec\nrounds: 7005"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 32509.3862425,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024826054243",
            "extra": "mean: 30.7603469514 usec\nrounds: 21944"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8969.55926055,
            "unit": "iter/sec",
            "range": "stddev: 0.000016233231098",
            "extra": "mean: 111.488198132 usec\nrounds: 5700"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5113.69107634,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117834383755",
            "extra": "mean: 195.553463256 usec\nrounds: 302"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3873.28689196,
            "unit": "iter/sec",
            "range": "stddev: 0.00000778331918971",
            "extra": "mean: 258.17865495 usec\nrounds: 3336"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3171.56363505,
            "unit": "iter/sec",
            "range": "stddev: 0.00000688475560097",
            "extra": "mean: 315.301887356 usec\nrounds: 2505"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 250.807416792,
            "unit": "iter/sec",
            "range": "stddev: 0.00179827565719",
            "extra": "mean: 3.98712292 msec\nrounds: 1037"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6364.17463898,
            "unit": "iter/sec",
            "range": "stddev: 0.00000515945116403",
            "extra": "mean: 157.129566162 usec\nrounds: 5212"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6256.16136142,
            "unit": "iter/sec",
            "range": "stddev: 0.00000702525149704",
            "extra": "mean: 159.842424488 usec\nrounds: 5486"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 670.397256847,
            "unit": "iter/sec",
            "range": "stddev: 0.0000256548699966",
            "extra": "mean: 1.49165288161 msec\nrounds: 643"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 736.731563503,
            "unit": "iter/sec",
            "range": "stddev: 0.000103147491283",
            "extra": "mean: 1.35734648757 msec\nrounds: 664"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1066.00607833,
            "unit": "iter/sec",
            "range": "stddev: 0.0000769125908464",
            "extra": "mean: 938.080955 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10335.7101484,
            "unit": "iter/sec",
            "range": "stddev: 0.00000399149099193",
            "extra": "mean: 96.7519392132 usec\nrounds: 8685"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 8081.26006057,
            "unit": "iter/sec",
            "range": "stddev: 0.00000461595493075",
            "extra": "mean: 123.74307874 usec\nrounds: 6951"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4938.36771583,
            "unit": "iter/sec",
            "range": "stddev: 0.00000776973859861",
            "extra": "mean: 202.496058929 usec\nrounds: 3279"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2529.97809529,
            "unit": "iter/sec",
            "range": "stddev: 0.00000963738147899",
            "extra": "mean: 395.260339155 usec\nrounds: 2425"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2939.97339065,
            "unit": "iter/sec",
            "range": "stddev: 0.00000911944136799",
            "extra": "mean: 340.139132953 usec\nrounds: 2761"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3271.69741838,
            "unit": "iter/sec",
            "range": "stddev: 0.00000860417003474",
            "extra": "mean: 305.651737347 usec\nrounds: 2311"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1253.07163163,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152017108085",
            "extra": "mean: 798.038974593 usec\nrounds: 1231"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1419.13045864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000185116293843",
            "extra": "mean: 704.6568509 usec\nrounds: 1369"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2124.4527287,
            "unit": "iter/sec",
            "range": "stddev: 0.0000166815684328",
            "extra": "mean: 470.709461544 usec\nrounds: 1259"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 785.523459345,
            "unit": "iter/sec",
            "range": "stddev: 0.000021820889399",
            "extra": "mean: 1.273036455 msec\nrounds: 753"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1380.7258407,
            "unit": "iter/sec",
            "range": "stddev: 0.0000157157179567",
            "extra": "mean: 724.256742737 usec\nrounds: 1291"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2024.14871683,
            "unit": "iter/sec",
            "range": "stddev: 0.0000140522231986",
            "extra": "mean: 494.0348462 usec\nrounds: 1274"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1251.96161949,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179955210959",
            "extra": "mean: 798.746530593 usec\nrounds: 1204"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1434.93580064,
            "unit": "iter/sec",
            "range": "stddev: 0.0000165518135794",
            "extra": "mean: 696.895289362 usec\nrounds: 1383"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2212.17747967,
            "unit": "iter/sec",
            "range": "stddev: 0.0000103335062107",
            "extra": "mean: 452.04329634 usec\nrounds: 1096"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 759.970666983,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176680254752",
            "extra": "mean: 1.31584025995 msec\nrounds: 749"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1239.11645947,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125405716449",
            "extra": "mean: 807.026645764 usec\nrounds: 1146"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1345.87429537,
            "unit": "iter/sec",
            "range": "stddev: 0.0000471360001823",
            "extra": "mean: 743.011441293 usec\nrounds: 1056"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 346.419208808,
            "unit": "iter/sec",
            "range": "stddev: 0.00016016200103",
            "extra": "mean: 2.8866759538 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1301.46149081,
            "unit": "iter/sec",
            "range": "stddev: 0.0000226180973076",
            "extra": "mean: 768.366952891 usec\nrounds: 1069"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1980.80093646,
            "unit": "iter/sec",
            "range": "stddev: 0.0000320349737358",
            "extra": "mean: 504.846287982 usec\nrounds: 128"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1061.53418824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000363912810456",
            "extra": "mean: 942.032777727 usec\nrounds: 990"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1063.45408895,
            "unit": "iter/sec",
            "range": "stddev: 0.000033772574289",
            "extra": "mean: 940.332084278 usec\nrounds: 832"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1158.53335981,
            "unit": "iter/sec",
            "range": "stddev: 0.0000357880141471",
            "extra": "mean: 863.16029792 usec\nrounds: 1132"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1162.73365734,
            "unit": "iter/sec",
            "range": "stddev: 0.0000348727176259",
            "extra": "mean: 860.042189098 usec\nrounds: 1152"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.7723583844,
            "unit": "iter/sec",
            "range": "stddev: 0.000450973473499",
            "extra": "mean: 26.4743861059 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 38.3736775381,
            "unit": "iter/sec",
            "range": "stddev: 0.000173216276034",
            "extra": "mean: 26.0595299736 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 38.4240250222,
            "unit": "iter/sec",
            "range": "stddev: 0.000292512626587",
            "extra": "mean: 26.0253838431 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 38.3180039065,
            "unit": "iter/sec",
            "range": "stddev: 0.000307812213075",
            "extra": "mean: 26.0973928193 msec\nrounds: 39"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peterdsharpe@gmail.com",
            "name": "Peter Sharpe",
            "username": "peterdsharpe"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "69270737b4b74fd280dc89c4eca9aa45c3d5ac55",
          "message": "[CI] Fail the docs build on Sphinx warnings, and build it on an 8-core runner instead of a 48-core one (#3057)",
          "timestamp": "2026-10-19T11:14:53-04:00",
          "tree_id": "81fd6f5a68d9b4e636a9a425ec795a459628f242",
          "url": "https://github.com/pytorch/tensordict/commit/69270737b4b74fd280dc89c4eca9aa45c3d5ac55"
        },
        "date": 1792423882033,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4430.54721953,
            "unit": "iter/sec",
            "range": "stddev: 0.00000922168110923",
            "extra": "mean: 225.705753816 usec\nrounds: 134"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3536.01435359,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128529952294",
            "extra": "mean: 282.804281885 usec\nrounds: 3090"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 1993.96571653,
            "unit": "iter/sec",
            "range": "stddev: 0.0000152645233716",
            "extra": "mean: 501.513136215 usec\nrounds: 1828"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4445.0188108,
            "unit": "iter/sec",
            "range": "stddev: 0.00000861429981025",
            "extra": "mean: 224.97092646 usec\nrounds: 3931"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3506.22446562,
            "unit": "iter/sec",
            "range": "stddev: 0.000010821097216",
            "extra": "mean: 285.207068117 usec\nrounds: 3187"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1773.56788523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125072785942",
            "extra": "mean: 563.83519815 usec\nrounds: 1645"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2209.09336172,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135034290331",
            "extra": "mean: 452.674394541 usec\nrounds: 71"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 591.912485344,
            "unit": "iter/sec",
            "range": "stddev: 0.0000275598667789",
            "extra": "mean: 1.68943893694 msec\nrounds: 571"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 150.717957955,
            "unit": "iter/sec",
            "range": "stddev: 0.000056728648487",
            "extra": "mean: 6.63490942662 msec\nrounds: 150"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2204.23715149,
            "unit": "iter/sec",
            "range": "stddev: 0.0000144604131924",
            "extra": "mean: 453.671692868 usec\nrounds: 2071"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 588.266450744,
            "unit": "iter/sec",
            "range": "stddev: 0.0000276774034903",
            "extra": "mean: 1.69990996212 msec\nrounds: 581"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 146.734855739,
            "unit": "iter/sec",
            "range": "stddev: 0.0000977837419676",
            "extra": "mean: 6.81501334472 msec\nrounds: 148"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 2962.52491555,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104883644959",
            "extra": "mean: 337.549903716 usec\nrounds: 2244"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 807.436634189,
            "unit": "iter/sec",
            "range": "stddev: 0.0000211960073038",
            "extra": "mean: 1.23848727895 msec\nrounds: 778"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 205.547450708,
            "unit": "iter/sec",
            "range": "stddev: 0.0000554829825853",
            "extra": "mean: 4.86505668914 msec\nrounds: 206"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 2907.16822255,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127091724079",
            "extra": "mean: 343.977342709 usec\nrounds: 2810"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 788.491238272,
            "unit": "iter/sec",
            "range": "stddev: 0.0000231285903085",
            "extra": "mean: 1.26824491061 msec\nrounds: 783"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 204.138719574,
            "unit": "iter/sec",
            "range": "stddev: 0.0000526199624675",
            "extra": "mean: 4.89862972632 msec\nrounds: 201"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3297.34308402,
            "unit": "iter/sec",
            "range": "stddev: 0.0000124230700581",
            "extra": "mean: 303.274477214 usec\nrounds: 2808"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 895.986120732,
            "unit": "iter/sec",
            "range": "stddev: 0.0000227867433174",
            "extra": "mean: 1.11608871707 msec\nrounds: 887"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 229.762603899,
            "unit": "iter/sec",
            "range": "stddev: 0.0000486615979972",
            "extra": "mean: 4.35231836265 msec\nrounds: 226"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3306.05095022,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112264883809",
            "extra": "mean: 302.475677192 usec\nrounds: 3023"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 900.73940844,
            "unit": "iter/sec",
            "range": "stddev: 0.0000216510530965",
            "extra": "mean: 1.11019901053 msec\nrounds: 850"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 228.941789942,
            "unit": "iter/sec",
            "range": "stddev: 0.0000443280376767",
            "extra": "mean: 4.36792251976 msec\nrounds: 227"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 187.215148333,
            "unit": "iter/sec",
            "range": "stddev: 0.0000331267447694",
            "extra": "mean: 5.3414481088 msec\nrounds: 184"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 98.5456194048,
            "unit": "iter/sec",
            "range": "stddev: 0.0000786774074957",
            "extra": "mean: 10.1475844998 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 180.031760324,
            "unit": "iter/sec",
            "range": "stddev: 0.0000561658684984",
            "extra": "mean: 5.55457547156 msec\nrounds: 176"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 93.6164564289,
            "unit": "iter/sec",
            "range": "stddev: 0.0000770821016018",
            "extra": "mean: 10.681882632 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.7989300506,
            "unit": "iter/sec",
            "range": "stddev: 0.000261717647963",
            "extra": "mean: 78.1315309986 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.42602401955,
            "unit": "iter/sec",
            "range": "stddev: 0.000427332799204",
            "extra": "mean: 155.617221 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.5439246509,
            "unit": "iter/sec",
            "range": "stddev: 0.000343670008603",
            "extra": "mean: 79.7198666153 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.16561418927,
            "unit": "iter/sec",
            "range": "stddev: 0.000273894688828",
            "extra": "mean: 162.189843429 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 22.9953539496,
            "unit": "iter/sec",
            "range": "stddev: 0.0002022026754",
            "extra": "mean: 43.487045348 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.4844229555,
            "unit": "iter/sec",
            "range": "stddev: 0.000314319892829",
            "extra": "mean: 87.0744663337 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 22.6495821095,
            "unit": "iter/sec",
            "range": "stddev: 0.000284956595823",
            "extra": "mean: 44.1509249559 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.3592368,
            "unit": "iter/sec",
            "range": "stddev: 0.000224800799727",
            "extra": "mean: 88.0340834165 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 25.6494227662,
            "unit": "iter/sec",
            "range": "stddev: 0.000181612381274",
            "extra": "mean: 38.9872321539 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 12.6964621994,
            "unit": "iter/sec",
            "range": "stddev: 0.000752644154039",
            "extra": "mean: 78.762098 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 25.3818039557,
            "unit": "iter/sec",
            "range": "stddev: 0.000424994687011",
            "extra": "mean: 39.3983028845 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 12.7499232923,
            "unit": "iter/sec",
            "range": "stddev: 0.00295641075473",
            "extra": "mean: 78.4318444175 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6089.59886996,
            "unit": "iter/sec",
            "range": "stddev: 0.0000185429043563",
            "extra": "mean: 164.214428791 usec\nrounds: 42"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6463.33051048,
            "unit": "iter/sec",
            "range": "stddev: 0.00000711456406434",
            "extra": "mean: 154.718994855 usec\nrounds: 1382"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6510.97599431,
            "unit": "iter/sec",
            "range": "stddev: 0.00000701271083575",
            "extra": "mean: 153.586804939 usec\nrounds: 5147"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22360.9995703,
            "unit": "iter/sec",
            "range": "stddev: 0.00000357460475845",
            "extra": "mean: 44.7207199686 usec\nrounds: 11367"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 22483.0173965,
            "unit": "iter/sec",
            "range": "stddev: 0.00000340428729141",
            "extra": "mean: 44.4780156669 usec\nrounds: 12191"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10320.6305751,
            "unit": "iter/sec",
            "range": "stddev: 0.00000834028603115",
            "extra": "mean: 96.8933044083 usec\nrounds: 174"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10483.9012688,
            "unit": "iter/sec",
            "range": "stddev: 0.00000555617557131",
            "extra": "mean: 95.3843396991 usec\nrounds: 7922"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 34.1563317198,
            "unit": "iter/sec",
            "range": "stddev: 0.000392748319355",
            "extra": "mean: 29.2771486178 msec\nrounds: 34"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 93.8635326338,
            "unit": "iter/sec",
            "range": "stddev: 0.0125092980992",
            "extra": "mean: 10.6537648002 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 95.9878902482,
            "unit": "iter/sec",
            "range": "stddev: 0.000126386460701",
            "extra": "mean: 10.4179808246 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 93.8547320229,
            "unit": "iter/sec",
            "range": "stddev: 0.000112782911554",
            "extra": "mean: 10.654763787 msec\nrounds: 94"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3876.59715014,
            "unit": "iter/sec",
            "range": "stddev: 0.00000992090861018",
            "extra": "mean: 257.958194073 usec\nrounds: 3818"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 288.389959791,
            "unit": "iter/sec",
            "range": "stddev: 0.000104857154574",
            "extra": "mean: 3.46752709673 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 413.472442707,
            "unit": "iter/sec",
            "range": "stddev: 0.000225628629607",
            "extra": "mean: 2.41854086684 msec\nrounds: 420"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 437.462256139,
            "unit": "iter/sec",
            "range": "stddev: 0.000279344219802",
            "extra": "mean: 2.28591149514 msec\nrounds: 414"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 461.096169743,
            "unit": "iter/sec",
            "range": "stddev: 0.0000322534775814",
            "extra": "mean: 2.16874497256 msec\nrounds: 439"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 610.003170791,
            "unit": "iter/sec",
            "range": "stddev: 0.0000639677363914",
            "extra": "mean: 1.639335741 msec\nrounds: 610"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 187.296573804,
            "unit": "iter/sec",
            "range": "stddev: 0.0000599750598735",
            "extra": "mean: 5.33912596312 msec\nrounds: 190"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 107.527371845,
            "unit": "iter/sec",
            "range": "stddev: 0.0000678344375463",
            "extra": "mean: 9.29995760931 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 133.329925659,
            "unit": "iter/sec",
            "range": "stddev: 0.0000588857632417",
            "extra": "mean: 7.5001916866 msec\nrounds: 134"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 248.758549126,
            "unit": "iter/sec",
            "range": "stddev: 0.0000478357556556",
            "extra": "mean: 4.01996234306 msec\nrounds: 239"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 104.202146185,
            "unit": "iter/sec",
            "range": "stddev: 0.000070310368694",
            "extra": "mean: 9.59673132093 msec\nrounds: 106"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 14.903956327,
            "unit": "iter/sec",
            "range": "stddev: 0.000575620925764",
            "extra": "mean: 67.0962782002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.40010814903,
            "unit": "iter/sec",
            "range": "stddev: 0.00225197682775",
            "extra": "mean: 227.2671412 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.635331186095,
            "unit": "iter/sec",
            "range": "stddev: 0.102391875432",
            "extra": "mean: 1.5739822346 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 37.6016585273,
            "unit": "iter/sec",
            "range": "stddev: 0.000466884523392",
            "extra": "mean: 26.5945716004 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 15.1512711704,
            "unit": "iter/sec",
            "range": "stddev: 0.000129342511966",
            "extra": "mean: 66.0010627988 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 15.4534778411,
            "unit": "iter/sec",
            "range": "stddev: 0.000822853696371",
            "extra": "mean: 64.7103526004 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 39.8384912518,
            "unit": "iter/sec",
            "range": "stddev: 0.000336319119979",
            "extra": "mean: 25.1013521993 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4582.35831551,
            "unit": "iter/sec",
            "range": "stddev: 0.000011662013655",
            "extra": "mean: 218.228242129 usec\nrounds: 1623"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 7839.271351,
            "unit": "iter/sec",
            "range": "stddev: 0.00000801243280926",
            "extra": "mean: 127.562876093 usec\nrounds: 6699"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6167.33831699,
            "unit": "iter/sec",
            "range": "stddev: 0.0000142000797745",
            "extra": "mean: 162.144501988 usec\nrounds: 2751"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5548.99511857,
            "unit": "iter/sec",
            "range": "stddev: 0.0000208992780176",
            "extra": "mean: 180.21280946 usec\nrounds: 5175"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 19545.0694653,
            "unit": "iter/sec",
            "range": "stddev: 0.00000358648480648",
            "extra": "mean: 51.1637987154 usec\nrounds: 15593"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30686.1483666,
            "unit": "iter/sec",
            "range": "stddev: 0.00000207082563454",
            "extra": "mean: 32.5879933856 usec\nrounds: 24194"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 48446.5852477,
            "unit": "iter/sec",
            "range": "stddev: 0.00000397074057429",
            "extra": "mean: 20.6412896778 usec\nrounds: 33978"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15696.1896946,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116307491291",
            "extra": "mean: 63.7097295239 usec\nrounds: 14793"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3587.20426172,
            "unit": "iter/sec",
            "range": "stddev: 0.00000903356296931",
            "extra": "mean: 278.768625102 usec\nrounds: 1059"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5120.99446195,
            "unit": "iter/sec",
            "range": "stddev: 0.00000906093685548",
            "extra": "mean: 195.274571654 usec\nrounds: 4891"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5467.09970673,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115090334219",
            "extra": "mean: 182.912339932 usec\nrounds: 3192"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15661.0091353,
            "unit": "iter/sec",
            "range": "stddev: 0.00000464293288973",
            "extra": "mean: 63.8528457115 usec\nrounds: 13349"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 3953.95878221,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111972168783",
            "extra": "mean: 252.911083569 usec\nrounds: 1364"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1921.48231467,
            "unit": "iter/sec",
            "range": "stddev: 0.0000216838526231",
            "extra": "mean: 520.431539945 usec\nrounds: 1906"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4271.21705336,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100037786852",
            "extra": "mean: 234.125306091 usec\nrounds: 4048"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3566.66933145,
            "unit": "iter/sec",
            "range": "stddev: 0.0000092356699113",
            "extra": "mean: 280.373622299 usec\nrounds: 1316"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12730.7757359,
            "unit": "iter/sec",
            "range": "stddev: 0.00000487002947369",
            "extra": "mean: 78.5498088052 usec\nrounds: 11014"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5592.45636016,
            "unit": "iter/sec",
            "range": "stddev: 0.00000808741664489",
            "extra": "mean: 178.812302788 usec\nrounds: 3161"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2193.35091531,
            "unit": "iter/sec",
            "range": "stddev: 0.000019452313793",
            "extra": "mean: 455.923396945 usec\nrounds: 2174"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4007.38220817,
            "unit": "iter/sec",
            "range": "stddev: 0.000010059992781",
            "extra": "mean: 249.539461937 usec\nrounds: 1366"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 30556.3416532,
            "unit": "iter/sec",
            "range": "stddev: 0.00000296891253371",
            "extra": "mean: 32.7264307799 usec\nrounds: 21087"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 54562.8110805,
            "unit": "iter/sec",
            "range": "stddev: 0.00000126668205255",
            "extra": "mean: 18.3275014648 usec\nrounds: 46793"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 42458.4820456,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038445487305",
            "extra": "mean: 23.5524199599 usec\nrounds: 26637"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19486.1818,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116568182567",
            "extra": "mean: 51.318416828 usec\nrounds: 16995"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3795.51717108,
            "unit": "iter/sec",
            "range": "stddev: 0.000227564189538",
            "extra": "mean: 263.468706615 usec\nrounds: 92"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 292.020787341,
            "unit": "iter/sec",
            "range": "stddev: 0.0000419370998389",
            "extra": "mean: 3.42441375186 msec\nrounds: 290"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 4856.8315151,
            "unit": "iter/sec",
            "range": "stddev: 0.000223826176269",
            "extra": "mean: 205.895550812 usec\nrounds: 89"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 343.886246218,
            "unit": "iter/sec",
            "range": "stddev: 0.000045850843714",
            "extra": "mean: 2.90793834007 msec\nrounds: 347"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 5926.25209927,
            "unit": "iter/sec",
            "range": "stddev: 0.00000969640098595",
            "extra": "mean: 168.740712215 usec\nrounds: 3791"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 17967.716832,
            "unit": "iter/sec",
            "range": "stddev: 0.00001291943044",
            "extra": "mean: 55.6553739882 usec\nrounds: 14607"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7312.21011077,
            "unit": "iter/sec",
            "range": "stddev: 0.00000764015660999",
            "extra": "mean: 136.757558228 usec\nrounds: 4373"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 20819.5529305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125239592649",
            "extra": "mean: 48.0317710634 usec\nrounds: 17730"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7242.20312972,
            "unit": "iter/sec",
            "range": "stddev: 0.00000794566663661",
            "extra": "mean: 138.079529404 usec\nrounds: 6517"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 20995.7023625,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120802421898",
            "extra": "mean: 47.6287948236 usec\nrounds: 17771"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11217.5578065,
            "unit": "iter/sec",
            "range": "stddev: 0.0000058502746777",
            "extra": "mean: 89.1459636091 usec\nrounds: 9338"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 43731.2181156,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113326252801",
            "extra": "mean: 22.8669596479 usec\nrounds: 33522"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15230.4003264,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128125332902",
            "extra": "mean: 65.6581559625 usec\nrounds: 12189"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 44172.6219958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113711895291",
            "extra": "mean: 22.6384569179 usec\nrounds: 34829"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 14843.452409,
            "unit": "iter/sec",
            "range": "stddev: 0.00000446337712117",
            "extra": "mean: 67.3697716978 usec\nrounds: 13009"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 43850.0701018,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113093404574",
            "extra": "mean: 22.8049806461 usec\nrounds: 35675"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11117.5908759,
            "unit": "iter/sec",
            "range": "stddev: 0.00000583082509217",
            "extra": "mean: 89.9475444961 usec\nrounds: 10002"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 43100.4443703,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108774592821",
            "extra": "mean: 23.2016169348 usec\nrounds: 33817"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 14762.3584994,
            "unit": "iter/sec",
            "range": "stddev: 0.00000434605873322",
            "extra": "mean: 67.7398533601 usec\nrounds: 12731"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 44805.7413183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110441930525",
            "extra": "mean: 22.3185683481 usec\nrounds: 34129"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 14855.8993641,
            "unit": "iter/sec",
            "range": "stddev: 0.00000415004784666",
            "extra": "mean: 67.3133262075 usec\nrounds: 13121"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 44329.978233,
            "unit": "iter/sec",
            "range": "stddev: 0.000010993235824",
            "extra": "mean: 22.5580981508 usec\nrounds: 35854"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 31707.1979794,
            "unit": "iter/sec",
            "range": "stddev: 0.00000284332482179",
            "extra": "mean: 31.5385799985 usec\nrounds: 22924"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6679.96820701,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119598120101",
            "extra": "mean: 149.701311295 usec\nrounds: 4247"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2495.21100719,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176256694381",
            "extra": "mean: 400.767709471 usec\nrounds: 2350"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6205.90525251,
            "unit": "iter/sec",
            "range": "stddev: 0.00000900289571538",
            "extra": "mean: 161.136846167 usec\nrounds: 3387"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 5884.21463588,
            "unit": "iter/sec",
            "range": "stddev: 0.00000928878622744",
            "extra": "mean: 169.946214046 usec\nrounds: 5658"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5239.22978301,
            "unit": "iter/sec",
            "range": "stddev: 0.00001009833027",
            "extra": "mean: 190.867749921 usec\nrounds: 3451"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5745.38001007,
            "unit": "iter/sec",
            "range": "stddev: 0.00000850753631001",
            "extra": "mean: 174.052890888 usec\nrounds: 5517"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 4880.37577962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100680623386",
            "extra": "mean: 204.90225449 usec\nrounds: 3517"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 107257.810774,
            "unit": "iter/sec",
            "range": "stddev: 8.78022919224e-7",
            "extra": "mean: 9.323330327 usec\nrounds: 81894"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 36883.0070528,
            "unit": "iter/sec",
            "range": "stddev: 0.00000484486064824",
            "extra": "mean: 27.1127567925 usec\nrounds: 23923"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 64801.5460309,
            "unit": "iter/sec",
            "range": "stddev: 0.00000117015578861",
            "extra": "mean: 15.4317305875 usec\nrounds: 46684"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 30336.4609568,
            "unit": "iter/sec",
            "range": "stddev: 0.00000266520562855",
            "extra": "mean: 32.9636341373 usec\nrounds: 20669"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36394.2607007,
            "unit": "iter/sec",
            "range": "stddev: 0.00000174396518289",
            "extra": "mean: 27.476859833 usec\nrounds: 33167"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 22576.5297908,
            "unit": "iter/sec",
            "range": "stddev: 0.00000397886766057",
            "extra": "mean: 44.2937869223 usec\nrounds: 16961"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 35859.2792503,
            "unit": "iter/sec",
            "range": "stddev: 0.00000253560940034",
            "extra": "mean: 27.8867847014 usec\nrounds: 27502"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8010.021742,
            "unit": "iter/sec",
            "range": "stddev: 0.00000672391086165",
            "extra": "mean: 124.843606199 usec\nrounds: 6968"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 29517.5536922,
            "unit": "iter/sec",
            "range": "stddev: 0.00000266227854422",
            "extra": "mean: 33.8781462186 usec\nrounds: 21303"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8690.31667031,
            "unit": "iter/sec",
            "range": "stddev: 0.0000325431840531",
            "extra": "mean: 115.070605357 usec\nrounds: 5666"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 4964.92693842,
            "unit": "iter/sec",
            "range": "stddev: 0.0000158485895005",
            "extra": "mean: 201.412832939 usec\nrounds: 299"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3590.10333558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134424037829",
            "extra": "mean: 278.543514358 usec\nrounds: 3276"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3007.19114808,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120998216582",
            "extra": "mean: 332.536227582 usec\nrounds: 2491"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 255.400413469,
            "unit": "iter/sec",
            "range": "stddev: 0.00173499000512",
            "extra": "mean: 3.91542044281 msec\nrounds: 987"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6006.24527939,
            "unit": "iter/sec",
            "range": "stddev: 0.00000760366018725",
            "extra": "mean: 166.493367068 usec\nrounds: 4980"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 5939.44196847,
            "unit": "iter/sec",
            "range": "stddev: 0.00000826334736049",
            "extra": "mean: 168.365985443 usec\nrounds: 5162"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 633.152686175,
            "unit": "iter/sec",
            "range": "stddev: 0.000121969091914",
            "extra": "mean: 1.57939786379 msec\nrounds: 631"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 748.701503741,
            "unit": "iter/sec",
            "range": "stddev: 0.0000296784088344",
            "extra": "mean: 1.33564577472 msec\nrounds: 657"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1039.66568795,
            "unit": "iter/sec",
            "range": "stddev: 0.000251734780825",
            "extra": "mean: 961.847651207 usec\nrounds: 86"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 9832.73257217,
            "unit": "iter/sec",
            "range": "stddev: 0.00000695583765142",
            "extra": "mean: 101.701128619 usec\nrounds: 8490"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7446.67594974,
            "unit": "iter/sec",
            "range": "stddev: 0.00000740569562519",
            "extra": "mean: 134.288104753 usec\nrounds: 6502"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4645.46467366,
            "unit": "iter/sec",
            "range": "stddev: 0.00000783199900641",
            "extra": "mean: 215.263718541 usec\nrounds: 3162"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2371.05552665,
            "unit": "iter/sec",
            "range": "stddev: 0.000013932038822",
            "extra": "mean: 421.75309214 usec\nrounds: 2256"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2747.06406665,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119973080879",
            "extra": "mean: 364.025001142 usec\nrounds: 2616"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3118.69400002,
            "unit": "iter/sec",
            "range": "stddev: 0.0000121797562331",
            "extra": "mean: 320.647040073 usec\nrounds: 2198"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1134.81410475,
            "unit": "iter/sec",
            "range": "stddev: 0.0000191080256752",
            "extra": "mean: 881.201595767 usec\nrounds: 1133"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1327.73655263,
            "unit": "iter/sec",
            "range": "stddev: 0.0000182599939492",
            "extra": "mean: 753.161459645 usec\nrounds: 1275"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2047.28188557,
            "unit": "iter/sec",
            "range": "stddev: 0.0000151276811361",
            "extra": "mean: 488.452521878 usec\nrounds: 1278"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 741.517609009,
            "unit": "iter/sec",
            "range": "stddev: 0.0000254307187345",
            "extra": "mean: 1.34858564092 msec\nrounds: 710"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1305.80950222,
            "unit": "iter/sec",
            "range": "stddev: 0.0000212614222149",
            "extra": "mean: 765.808487608 usec\nrounds: 1296"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 1937.68379552,
            "unit": "iter/sec",
            "range": "stddev: 0.000017208863292",
            "extra": "mean: 516.080075763 usec\nrounds: 1161"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1157.16459423,
            "unit": "iter/sec",
            "range": "stddev: 0.000020524844355",
            "extra": "mean: 864.1812971 usec\nrounds: 1155"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1335.76330352,
            "unit": "iter/sec",
            "range": "stddev: 0.0000214193025845",
            "extra": "mean: 748.635628305 usec\nrounds: 1232"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2124.79905908,
            "unit": "iter/sec",
            "range": "stddev: 0.0000128679803046",
            "extra": "mean: 470.63273853 usec\nrounds: 1098"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 711.492224732,
            "unit": "iter/sec",
            "range": "stddev: 0.0000301581035009",
            "extra": "mean: 1.40549673663 msec\nrounds: 714"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1188.7588721,
            "unit": "iter/sec",
            "range": "stddev: 0.0000181851820509",
            "extra": "mean: 841.213490362 usec\nrounds: 1138"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1295.64525948,
            "unit": "iter/sec",
            "range": "stddev: 0.000083821104955",
            "extra": "mean: 771.816199447 usec\nrounds: 948"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 334.454971249,
            "unit": "iter/sec",
            "range": "stddev: 0.0000364751466786",
            "extra": "mean: 2.98993911278 msec\nrounds: 328"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1247.79843864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000153537365153",
            "extra": "mean: 801.411485246 usec\nrounds: 983"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1884.35363551,
            "unit": "iter/sec",
            "range": "stddev: 0.0000340643501248",
            "extra": "mean: 530.685950425 usec\nrounds: 121"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1003.87620283,
            "unit": "iter/sec",
            "range": "stddev: 0.0000374400353959",
            "extra": "mean: 996.1387641 usec\nrounds: 941"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1017.00725717,
            "unit": "iter/sec",
            "range": "stddev: 0.0000358432148858",
            "extra": "mean: 983.277152594 usec\nrounds: 1009"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1105.54822439,
            "unit": "iter/sec",
            "range": "stddev: 0.0000353402206796",
            "extra": "mean: 904.528611182 usec\nrounds: 1088"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1107.66633074,
            "unit": "iter/sec",
            "range": "stddev: 0.0000358860193689",
            "extra": "mean: 902.798949694 usec\nrounds: 1114"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 36.5605105624,
            "unit": "iter/sec",
            "range": "stddev: 0.0000915043515832",
            "extra": "mean: 27.351915622 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 36.6483765651,
            "unit": "iter/sec",
            "range": "stddev: 0.0000715451850914",
            "extra": "mean: 27.28633827 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 36.633285644,
            "unit": "iter/sec",
            "range": "stddev: 0.000182315111617",
            "extra": "mean: 27.2975787572 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 36.8007987484,
            "unit": "iter/sec",
            "range": "stddev: 0.0000937479432291",
            "extra": "mean: 27.1733232432 msec\nrounds: 37"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peterdsharpe@gmail.com",
            "name": "Peter Sharpe",
            "username": "peterdsharpe"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "977215952c51209e92bc7449ea41cb5e5e6b16ba",
          "message": "[CI] Run the core unit tests nightly on Windows and Linux ARM, without blocking the upload (#3059)",
          "timestamp": "2026-10-19T11:15:11-04:00",
          "tree_id": "4385c4802410bc665eff3c7ce0d94add2431963c",
          "url": "https://github.com/pytorch/tensordict/commit/977215952c51209e92bc7449ea41cb5e5e6b16ba"
        },
        "date": 1792424027313,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4556.49972302,
            "unit": "iter/sec",
            "range": "stddev: 0.00000868341464326",
            "extra": "mean: 219.46670927 usec\nrounds: 141"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3607.46824999,
            "unit": "iter/sec",
            "range": "stddev: 0.00000842872911689",
            "extra": "mean: 277.202716893 usec\nrounds: 3136"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2031.85162747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000112999379036",
            "extra": "mean: 492.161920921 usec\nrounds: 1822"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4388.57821064,
            "unit": "iter/sec",
            "range": "stddev: 0.00000761195733804",
            "extra": "mean: 227.864231194 usec\nrounds: 3802"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3487.3568611,
            "unit": "iter/sec",
            "range": "stddev: 0.00000944620318678",
            "extra": "mean: 286.750120458 usec\nrounds: 3129"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1795.67003103,
            "unit": "iter/sec",
            "range": "stddev: 0.0000101598497067",
            "extra": "mean: 556.89518827 usec\nrounds: 1678"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2278.9257453,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110513508754",
            "extra": "mean: 438.803239667 usec\nrounds: 75"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 606.121646324,
            "unit": "iter/sec",
            "range": "stddev: 0.0000418686119706",
            "extra": "mean: 1.64983383462 msec\nrounds: 569"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 153.772757768,
            "unit": "iter/sec",
            "range": "stddev: 0.000120602835097",
            "extra": "mean: 6.50310246443 msec\nrounds: 155"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2267.05216702,
            "unit": "iter/sec",
            "range": "stddev: 0.0000136722808685",
            "extra": "mean: 441.101450839 usec\nrounds: 2160"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 604.450061432,
            "unit": "iter/sec",
            "range": "stddev: 0.0000349891116201",
            "extra": "mean: 1.65439639071 msec\nrounds: 599"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 151.710451607,
            "unit": "iter/sec",
            "range": "stddev: 0.000112692909537",
            "extra": "mean: 6.59150367959 msec\nrounds: 153"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3042.74202221,
            "unit": "iter/sec",
            "range": "stddev: 0.0000256140480267",
            "extra": "mean: 328.650931529 usec\nrounds: 2393"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 821.277629748,
            "unit": "iter/sec",
            "range": "stddev: 0.0000563344619083",
            "extra": "mean: 1.21761504731 msec\nrounds: 783"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 211.619126659,
            "unit": "iter/sec",
            "range": "stddev: 0.0000740136013388",
            "extra": "mean: 4.72547078228 msec\nrounds: 193"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3060.12867766,
            "unit": "iter/sec",
            "range": "stddev: 0.0000182222330149",
            "extra": "mean: 326.783643871 usec\nrounds: 2869"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 833.324973644,
            "unit": "iter/sec",
            "range": "stddev: 0.000027248358578",
            "extra": "mean: 1.20001203807 msec\nrounds: 816"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 213.324353206,
            "unit": "iter/sec",
            "range": "stddev: 0.0000319512857296",
            "extra": "mean: 4.68769732555 msec\nrounds: 209"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3637.8497336,
            "unit": "iter/sec",
            "range": "stddev: 0.00000832554035742",
            "extra": "mean: 274.887659807 usec\nrounds: 3060"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 963.229925674,
            "unit": "iter/sec",
            "range": "stddev: 0.00003847376811",
            "extra": "mean: 1.03817372503 msec\nrounds: 953"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 245.520497638,
            "unit": "iter/sec",
            "range": "stddev: 0.0000508179388217",
            "extra": "mean: 4.07297968855 msec\nrounds: 247"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3568.39506962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000176278102659",
            "extra": "mean: 280.238028719 usec\nrounds: 3342"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 962.751318282,
            "unit": "iter/sec",
            "range": "stddev: 0.0000438084569925",
            "extra": "mean: 1.03868982676 msec\nrounds: 952"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 246.267182409,
            "unit": "iter/sec",
            "range": "stddev: 0.0000502327997803",
            "extra": "mean: 4.0606303699 msec\nrounds: 246"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 184.326055483,
            "unit": "iter/sec",
            "range": "stddev: 0.000275027885633",
            "extra": "mean: 5.4251689886 msec\nrounds: 178"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 99.0264571794,
            "unit": "iter/sec",
            "range": "stddev: 0.000061788348032",
            "extra": "mean: 10.0983113855 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 181.2132313,
            "unit": "iter/sec",
            "range": "stddev: 0.0000930579038782",
            "extra": "mean: 5.51836084387 msec\nrounds: 179"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.1543423827,
            "unit": "iter/sec",
            "range": "stddev: 0.0000238530035992",
            "extra": "mean: 10.6208590565 msec\nrounds: 88"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.9365025612,
            "unit": "iter/sec",
            "range": "stddev: 0.000530387722648",
            "extra": "mean: 77.3006456164 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.50778004131,
            "unit": "iter/sec",
            "range": "stddev: 0.00113531365507",
            "extra": "mean: 153.662230999 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.8259509729,
            "unit": "iter/sec",
            "range": "stddev: 0.000398287094903",
            "extra": "mean: 77.9669283091 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.4548693343,
            "unit": "iter/sec",
            "range": "stddev: 0.00112946740804",
            "extra": "mean: 154.921803713 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.52234171,
            "unit": "iter/sec",
            "range": "stddev: 0.000166964641731",
            "extra": "mean: 42.5127741246 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.8425637611,
            "unit": "iter/sec",
            "range": "stddev: 0.000164021523917",
            "extra": "mean: 84.4411750847 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.4253093687,
            "unit": "iter/sec",
            "range": "stddev: 0.000098202692875",
            "extra": "mean: 42.6888705827 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.6987126189,
            "unit": "iter/sec",
            "range": "stddev: 0.000268519181718",
            "extra": "mean: 85.4794909986 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.4679120979,
            "unit": "iter/sec",
            "range": "stddev: 0.000167685704432",
            "extra": "mean: 36.4061162142 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.7281800846,
            "unit": "iter/sec",
            "range": "stddev: 0.000304471691882",
            "extra": "mean: 72.8428672872 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.2578926959,
            "unit": "iter/sec",
            "range": "stddev: 0.000416590836787",
            "extra": "mean: 36.6866217854 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.5611750748,
            "unit": "iter/sec",
            "range": "stddev: 0.000348118742423",
            "extra": "mean: 73.7399225717 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6441.15908167,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163232833481",
            "extra": "mean: 155.251560677 usec\nrounds: 41"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6766.82613996,
            "unit": "iter/sec",
            "range": "stddev: 0.00000573723489648",
            "extra": "mean: 147.779768435 usec\nrounds: 1429"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6907.07427104,
            "unit": "iter/sec",
            "range": "stddev: 0.00000543807325879",
            "extra": "mean: 144.779100493 usec\nrounds: 5126"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 23277.4610277,
            "unit": "iter/sec",
            "range": "stddev: 0.00000249305797297",
            "extra": "mean: 42.9600117817 usec\nrounds: 12032"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 24243.2849749,
            "unit": "iter/sec",
            "range": "stddev: 0.00000247885201398",
            "extra": "mean: 41.2485354619 usec\nrounds: 12519"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10572.5231792,
            "unit": "iter/sec",
            "range": "stddev: 0.00000825778062147",
            "extra": "mean: 94.5848009079 usec\nrounds: 176"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10916.385152,
            "unit": "iter/sec",
            "range": "stddev: 0.00000383735120207",
            "extra": "mean: 91.605415719 usec\nrounds: 8123"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 56.6077826183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000900997952979",
            "extra": "mean: 17.6654154914 msec\nrounds: 57"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 95.7291358155,
            "unit": "iter/sec",
            "range": "stddev: 0.0118588494837",
            "extra": "mean: 10.4461404721 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 97.3623759438,
            "unit": "iter/sec",
            "range": "stddev: 0.0000852953661643",
            "extra": "mean: 10.2709079386 msec\nrounds: 98"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 95.638560699,
            "unit": "iter/sec",
            "range": "stddev: 0.0000967333794907",
            "extra": "mean: 10.4560335569 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3822.12977939,
            "unit": "iter/sec",
            "range": "stddev: 0.00000758514596189",
            "extra": "mean: 261.634234764 usec\nrounds: 3676"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 291.494485638,
            "unit": "iter/sec",
            "range": "stddev: 0.0000412361033146",
            "extra": "mean: 3.43059662968 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 240.594392499,
            "unit": "iter/sec",
            "range": "stddev: 0.0391769705087",
            "extra": "mean: 4.15637284649 msec\nrounds: 417"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 453.834642601,
            "unit": "iter/sec",
            "range": "stddev: 0.000238801994929",
            "extra": "mean: 2.20344571818 msec\nrounds: 429"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 504.12125129,
            "unit": "iter/sec",
            "range": "stddev: 0.0000352743237506",
            "extra": "mean: 1.98364976171 msec\nrounds: 474"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 648.298784325,
            "unit": "iter/sec",
            "range": "stddev: 0.0000581789670883",
            "extra": "mean: 1.54249865059 msec\nrounds: 647"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 191.295698605,
            "unit": "iter/sec",
            "range": "stddev: 0.000110654114627",
            "extra": "mean: 5.22750907255 msec\nrounds: 193"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 109.839781201,
            "unit": "iter/sec",
            "range": "stddev: 0.000489888668621",
            "extra": "mean: 9.10416962842 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 138.954370593,
            "unit": "iter/sec",
            "range": "stddev: 0.000170419732902",
            "extra": "mean: 7.19660702811 msec\nrounds: 142"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 263.307760922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000734596045424",
            "extra": "mean: 3.79783716401 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 107.891129717,
            "unit": "iter/sec",
            "range": "stddev: 0.000895408461871",
            "extra": "mean: 9.26860254988 msec\nrounds: 111"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.8451157761,
            "unit": "iter/sec",
            "range": "stddev: 0.00100262249251",
            "extra": "mean: 92.2074066009 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 2.94106365313,
            "unit": "iter/sec",
            "range": "stddev: 0.0832060098033",
            "extra": "mean: 340.013042199 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.44416260945,
            "unit": "iter/sec",
            "range": "stddev: 0.077132987739",
            "extra": "mean: 2.251427695 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 43.1658109633,
            "unit": "iter/sec",
            "range": "stddev: 0.000355007958426",
            "extra": "mean: 23.1664824008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 6.57580212763,
            "unit": "iter/sec",
            "range": "stddev: 0.0985837446578",
            "extra": "mean: 152.072702401 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 9.94658308901,
            "unit": "iter/sec",
            "range": "stddev: 0.00215615210092",
            "extra": "mean: 100.5370378 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 43.1155046331,
            "unit": "iter/sec",
            "range": "stddev: 0.000326486832151",
            "extra": "mean: 23.1935126008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4798.19404841,
            "unit": "iter/sec",
            "range": "stddev: 0.00000624843368326",
            "extra": "mean: 208.411746151 usec\nrounds: 1761"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8165.99758573,
            "unit": "iter/sec",
            "range": "stddev: 0.00000560301272508",
            "extra": "mean: 122.459012447 usec\nrounds: 6589"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6558.16456488,
            "unit": "iter/sec",
            "range": "stddev: 0.00000571224747159",
            "extra": "mean: 152.481687537 usec\nrounds: 3046"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5721.87687753,
            "unit": "iter/sec",
            "range": "stddev: 0.0000138229577902",
            "extra": "mean: 174.767829054 usec\nrounds: 5341"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 20990.5973758,
            "unit": "iter/sec",
            "range": "stddev: 0.00000271082313082",
            "extra": "mean: 47.640378313 usec\nrounds: 16407"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30289.7330998,
            "unit": "iter/sec",
            "range": "stddev: 0.00000161240218499",
            "extra": "mean: 33.0144870113 usec\nrounds: 25569"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 50918.390569,
            "unit": "iter/sec",
            "range": "stddev: 0.00000475718179666",
            "extra": "mean: 19.6392696004 usec\nrounds: 38535"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15716.5324911,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117670807264",
            "extra": "mean: 63.6272664193 usec\nrounds: 14629"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3736.8627471,
            "unit": "iter/sec",
            "range": "stddev: 0.00000795242423076",
            "extra": "mean: 267.604155592 usec\nrounds: 1344"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5307.72477831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000578617049526",
            "extra": "mean: 188.404644507 usec\nrounds: 5078"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5704.70327867,
            "unit": "iter/sec",
            "range": "stddev: 0.00000751919370492",
            "extra": "mean: 175.293955032 usec\nrounds: 2954"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16212.640855,
            "unit": "iter/sec",
            "range": "stddev: 0.00000355748988131",
            "extra": "mean: 61.6802659692 usec\nrounds: 14023"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4062.04557771,
            "unit": "iter/sec",
            "range": "stddev: 0.00000786473324902",
            "extra": "mean: 246.18138346 usec\nrounds: 1229"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1906.72405185,
            "unit": "iter/sec",
            "range": "stddev: 0.0000162459383118",
            "extra": "mean: 524.459739745 usec\nrounds: 1691"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4259.36494413,
            "unit": "iter/sec",
            "range": "stddev: 0.00000903350471876",
            "extra": "mean: 234.776783186 usec\nrounds: 3833"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3663.17385628,
            "unit": "iter/sec",
            "range": "stddev: 0.00000824361237584",
            "extra": "mean: 272.987316254 usec\nrounds: 1173"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 13155.6115831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000409972673051",
            "extra": "mean: 76.0131897848 usec\nrounds: 11366"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5777.48322602,
            "unit": "iter/sec",
            "range": "stddev: 0.0000071798964751",
            "extra": "mean: 173.085747008 usec\nrounds: 3170"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2186.35896348,
            "unit": "iter/sec",
            "range": "stddev: 0.0000229330206485",
            "extra": "mean: 457.381434934 usec\nrounds: 2150"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4095.73637036,
            "unit": "iter/sec",
            "range": "stddev: 0.00000989761169221",
            "extra": "mean: 244.156339561 usec\nrounds: 1401"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 33620.487249,
            "unit": "iter/sec",
            "range": "stddev: 0.00000221573206218",
            "extra": "mean: 29.7437688096 usec\nrounds: 21552"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 55350.8204747,
            "unit": "iter/sec",
            "range": "stddev: 0.00000115250555192",
            "extra": "mean: 18.0665795272 usec\nrounds: 46491"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 34511.5245985,
            "unit": "iter/sec",
            "range": "stddev: 0.00000252123833079",
            "extra": "mean: 28.9758279773 usec\nrounds: 29044"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19810.8759743,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113280903474",
            "extra": "mean: 50.4773237336 usec\nrounds: 17337"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 4160.13749106,
            "unit": "iter/sec",
            "range": "stddev: 0.000233652413576",
            "extra": "mean: 240.376670759 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 304.856295251,
            "unit": "iter/sec",
            "range": "stddev: 0.0000390982163738",
            "extra": "mean: 3.28023404987 msec\nrounds: 302"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5084.0566832,
            "unit": "iter/sec",
            "range": "stddev: 0.000209994386451",
            "extra": "mean: 196.693322343 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 349.494739832,
            "unit": "iter/sec",
            "range": "stddev: 0.0000441854902975",
            "extra": "mean: 2.86127339279 msec\nrounds: 349"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6285.46900575,
            "unit": "iter/sec",
            "range": "stddev: 0.00000544358777113",
            "extra": "mean: 159.097117349 usec\nrounds: 3905"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 17799.2375181,
            "unit": "iter/sec",
            "range": "stddev: 0.0000125240113649",
            "extra": "mean: 56.1821819043 usec\nrounds: 15399"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7665.87898413,
            "unit": "iter/sec",
            "range": "stddev: 0.00000592121272645",
            "extra": "mean: 130.448185012 usec\nrounds: 4384"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21325.7569363,
            "unit": "iter/sec",
            "range": "stddev: 0.000011651526301",
            "extra": "mean: 46.8916532711 usec\nrounds: 18015"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7521.26249449,
            "unit": "iter/sec",
            "range": "stddev: 0.00000586793987758",
            "extra": "mean: 132.956402031 usec\nrounds: 6968"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 20218.4834488,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116460054063",
            "extra": "mean: 49.4596937764 usec\nrounds: 17668"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11529.069108,
            "unit": "iter/sec",
            "range": "stddev: 0.0000081501200066",
            "extra": "mean: 86.7372717285 usec\nrounds: 9774"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 45477.4842806,
            "unit": "iter/sec",
            "range": "stddev: 0.0000119435255644",
            "extra": "mean: 21.9889032082 usec\nrounds: 33806"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15461.5740278,
            "unit": "iter/sec",
            "range": "stddev: 0.00000635323760156",
            "extra": "mean: 64.67646814 usec\nrounds: 12869"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 39648.0060292,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127547459122",
            "extra": "mean: 25.2219493526 usec\nrounds: 35816"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15432.3068518,
            "unit": "iter/sec",
            "range": "stddev: 0.00000574560803198",
            "extra": "mean: 64.7991262488 usec\nrounds: 12663"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 46708.5009069,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113113009085",
            "extra": "mean: 21.4093790335 usec\nrounds: 36602"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11713.1511429,
            "unit": "iter/sec",
            "range": "stddev: 0.00000759155950946",
            "extra": "mean: 85.374122454 usec\nrounds: 10214"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 46225.9945067,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113127803656",
            "extra": "mean: 21.6328498861 usec\nrounds: 36656"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15877.6577885,
            "unit": "iter/sec",
            "range": "stddev: 0.00000549899954809",
            "extra": "mean: 62.9815816236 usec\nrounds: 13725"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 46893.9206114,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113123688728",
            "extra": "mean: 21.3247258272 usec\nrounds: 36696"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15784.5574358,
            "unit": "iter/sec",
            "range": "stddev: 0.00000373369958467",
            "extra": "mean: 63.3530590937 usec\nrounds: 11398"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 46510.5045988,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113273423063",
            "extra": "mean: 21.5005192617 usec\nrounds: 37271"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 32873.9572642,
            "unit": "iter/sec",
            "range": "stddev: 0.00000202904189831",
            "extra": "mean: 30.4192157933 usec\nrounds: 23127"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6748.37128119,
            "unit": "iter/sec",
            "range": "stddev: 0.0000080690539667",
            "extra": "mean: 148.183903691 usec\nrounds: 4297"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2560.65271156,
            "unit": "iter/sec",
            "range": "stddev: 0.0000145345183271",
            "extra": "mean: 390.525429507 usec\nrounds: 2547"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 6476.88315485,
            "unit": "iter/sec",
            "range": "stddev: 0.00000708649909631",
            "extra": "mean: 154.39525094 usec\nrounds: 3530"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 5999.23269831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000579825201716",
            "extra": "mean: 166.687983329 usec\nrounds: 5819"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5773.33961169,
            "unit": "iter/sec",
            "range": "stddev: 0.00000663661789031",
            "extra": "mean: 173.209973301 usec\nrounds: 3825"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5709.31729557,
            "unit": "iter/sec",
            "range": "stddev: 0.00000541200763498",
            "extra": "mean: 175.152290235 usec\nrounds: 5357"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5031.12555039,
            "unit": "iter/sec",
            "range": "stddev: 0.00000715900670195",
            "extra": "mean: 198.762680435 usec\nrounds: 3452"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 108235.23577,
            "unit": "iter/sec",
            "range": "stddev: 6.25209629968e-7",
            "extra": "mean: 9.23913541543 usec\nrounds: 82170"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 31744.8567668,
            "unit": "iter/sec",
            "range": "stddev: 0.00000518402630466",
            "extra": "mean: 31.5011659161 usec\nrounds: 25368"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65014.4042327,
            "unit": "iter/sec",
            "range": "stddev: 0.00000128282559799",
            "extra": "mean: 15.381206854 usec\nrounds: 55341"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 32657.0671594,
            "unit": "iter/sec",
            "range": "stddev: 0.00000364319537555",
            "extra": "mean: 30.621243332 usec\nrounds: 21381"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36818.0458784,
            "unit": "iter/sec",
            "range": "stddev: 0.00000185230478502",
            "extra": "mean: 27.1605941093 usec\nrounds: 33613"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 22516.5219218,
            "unit": "iter/sec",
            "range": "stddev: 0.00000255077793975",
            "extra": "mean: 44.4118324968 usec\nrounds: 17224"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 37717.076275,
            "unit": "iter/sec",
            "range": "stddev: 0.00000174835471861",
            "extra": "mean: 26.5131897475 usec\nrounds: 29078"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8526.26405338,
            "unit": "iter/sec",
            "range": "stddev: 0.0000075640011729",
            "extra": "mean: 117.284662279 usec\nrounds: 6668"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 32773.9272324,
            "unit": "iter/sec",
            "range": "stddev: 0.00000326829289786",
            "extra": "mean: 30.512058958 usec\nrounds: 21645"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 9103.37126027,
            "unit": "iter/sec",
            "range": "stddev: 0.0000244845458892",
            "extra": "mean: 109.84941418 usec\nrounds: 5521"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5136.39150523,
            "unit": "iter/sec",
            "range": "stddev: 0.0000574803176785",
            "extra": "mean: 194.689209143 usec\nrounds: 306"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3568.52287859,
            "unit": "iter/sec",
            "range": "stddev: 0.0000276425931755",
            "extra": "mean: 280.227991812 usec\nrounds: 3438"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3031.37684698,
            "unit": "iter/sec",
            "range": "stddev: 0.0000391330118451",
            "extra": "mean: 329.883102788 usec\nrounds: 2218"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 264.39162555,
            "unit": "iter/sec",
            "range": "stddev: 0.00167635785038",
            "extra": "mean: 3.78226805755 msec\nrounds: 957"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6126.49852489,
            "unit": "iter/sec",
            "range": "stddev: 0.00000995049249666",
            "extra": "mean: 163.225371872 usec\nrounds: 5048"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6061.0004135,
            "unit": "iter/sec",
            "range": "stddev: 0.00000942529946896",
            "extra": "mean: 164.989264441 usec\nrounds: 5021"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 598.80530515,
            "unit": "iter/sec",
            "range": "stddev: 0.00021193357422",
            "extra": "mean: 1.66999188451 msec\nrounds: 622"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 701.42366821,
            "unit": "iter/sec",
            "range": "stddev: 0.0000218775650815",
            "extra": "mean: 1.42567188038 msec\nrounds: 594"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 949.163075224,
            "unit": "iter/sec",
            "range": "stddev: 0.000106630570516",
            "extra": "mean: 1.0535597371 msec\nrounds: 80"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10229.3179767,
            "unit": "iter/sec",
            "range": "stddev: 0.00000465178495388",
            "extra": "mean: 97.7582280926 usec\nrounds: 8744"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7970.1881901,
            "unit": "iter/sec",
            "range": "stddev: 0.00000514756652547",
            "extra": "mean: 125.467551851 usec\nrounds: 6560"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4858.22983516,
            "unit": "iter/sec",
            "range": "stddev: 0.00000857542618328",
            "extra": "mean: 205.836288922 usec\nrounds: 3338"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2390.3662267,
            "unit": "iter/sec",
            "range": "stddev: 0.0000343714601918",
            "extra": "mean: 418.345937468 usec\nrounds: 2350"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2951.53696334,
            "unit": "iter/sec",
            "range": "stddev: 0.0000101446422251",
            "extra": "mean: 338.806531113 usec\nrounds: 2790"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3247.06778723,
            "unit": "iter/sec",
            "range": "stddev: 0.0000192213054996",
            "extra": "mean: 307.970164323 usec\nrounds: 2349"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1172.49788381,
            "unit": "iter/sec",
            "range": "stddev: 0.0000389606241344",
            "extra": "mean: 852.880004142 usec\nrounds: 1211"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1394.68376086,
            "unit": "iter/sec",
            "range": "stddev: 0.0000249521152505",
            "extra": "mean: 717.008420162 usec\nrounds: 1359"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 1961.633766,
            "unit": "iter/sec",
            "range": "stddev: 0.0000832717679409",
            "extra": "mean: 509.779153139 usec\nrounds: 1174"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 779.25655154,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179677302423",
            "extra": "mean: 1.28327442102 msec\nrounds: 772"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1402.72761307,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104670071818",
            "extra": "mean: 712.896781018 usec\nrounds: 1324"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 2021.07693838,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123332165105",
            "extra": "mean: 494.785715977 usec\nrounds: 1257"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1161.12409494,
            "unit": "iter/sec",
            "range": "stddev: 0.0000325047089659",
            "extra": "mean: 861.234388603 usec\nrounds: 1194"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1436.97743779,
            "unit": "iter/sec",
            "range": "stddev: 0.0000190606735954",
            "extra": "mean: 695.905150426 usec\nrounds: 1396"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2218.56448404,
            "unit": "iter/sec",
            "range": "stddev: 0.0000142429581706",
            "extra": "mean: 450.741913159 usec\nrounds: 1115"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 755.713844795,
            "unit": "iter/sec",
            "range": "stddev: 0.0000275574546896",
            "extra": "mean: 1.32325218982 msec\nrounds: 743"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1232.88364247,
            "unit": "iter/sec",
            "range": "stddev: 0.0000146098591299",
            "extra": "mean: 811.106551791 usec\nrounds: 1167"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1321.06636579,
            "unit": "iter/sec",
            "range": "stddev: 0.0000933079031536",
            "extra": "mean: 756.964241842 usec\nrounds: 914"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 345.951425657,
            "unit": "iter/sec",
            "range": "stddev: 0.0000458591777439",
            "extra": "mean: 2.89057921383 msec\nrounds: 337"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1326.41778268,
            "unit": "iter/sec",
            "range": "stddev: 0.0000233204621965",
            "extra": "mean: 753.91027854 usec\nrounds: 1037"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1938.06634727,
            "unit": "iter/sec",
            "range": "stddev: 0.0000367737146375",
            "extra": "mean: 515.978207561 usec\nrounds: 130"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 991.478269841,
            "unit": "iter/sec",
            "range": "stddev: 0.0000748200384517",
            "extra": "mean: 1.00859497421 msec\nrounds: 972"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1011.94718494,
            "unit": "iter/sec",
            "range": "stddev: 0.0000670721910911",
            "extra": "mean: 988.193865131 usec\nrounds: 926"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1122.68396433,
            "unit": "iter/sec",
            "range": "stddev: 0.000041773553384",
            "extra": "mean: 890.722618095 usec\nrounds: 1136"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1123.05043336,
            "unit": "iter/sec",
            "range": "stddev: 0.0000377010647055",
            "extra": "mean: 890.431961284 usec\nrounds: 1134"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.8371813114,
            "unit": "iter/sec",
            "range": "stddev: 0.000142669137438",
            "extra": "mean: 26.4290300001 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 37.8477696084,
            "unit": "iter/sec",
            "range": "stddev: 0.000205300628369",
            "extra": "mean: 26.4216362113 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 37.7435944024,
            "unit": "iter/sec",
            "range": "stddev: 0.000417556694085",
            "extra": "mean: 26.4945619471 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 38.2528854933,
            "unit": "iter/sec",
            "range": "stddev: 0.000196824194732",
            "extra": "mean: 26.1418187701 msec\nrounds: 39"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peterdsharpe@gmail.com",
            "name": "Peter Sharpe",
            "username": "peterdsharpe"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5b4785d5114efbfba938591f7aecd6c49c17e0d8",
          "message": "[BugFix] Keyword tensor operands give the dense result on a lazy stack (#3060)",
          "timestamp": "2026-10-19T11:15:30-04:00",
          "tree_id": "e1f4d770cbff972d603299cd2593d6d245292a82",
          "url": "https://github.com/pytorch/tensordict/commit/5b4785d5114efbfba938591f7aecd6c49c17e0d8"
        },
        "date": 1792424296949,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4534.46319013,
            "unit": "iter/sec",
            "range": "stddev: 0.00000709403006911",
            "extra": "mean: 220.5332711 usec\nrounds: 140"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3472.14694994,
            "unit": "iter/sec",
            "range": "stddev: 0.0000145993466587",
            "extra": "mean: 288.006243519 usec\nrounds: 3162"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 1965.51079287,
            "unit": "iter/sec",
            "range": "stddev: 0.0000170237191482",
            "extra": "mean: 508.773599019 usec\nrounds: 1786"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4321.56720456,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105034302527",
            "extra": "mean: 231.397535353 usec\nrounds: 4012"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3424.16616232,
            "unit": "iter/sec",
            "range": "stddev: 0.00000949103843672",
            "extra": "mean: 292.041902348 usec\nrounds: 3203"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1778.84146212,
            "unit": "iter/sec",
            "range": "stddev: 0.0000174680315677",
            "extra": "mean: 562.163644875 usec\nrounds: 1642"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2292.8135938,
            "unit": "iter/sec",
            "range": "stddev: 0.0000281649027809",
            "extra": "mean: 436.145355515 usec\nrounds: 76"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 589.178561193,
            "unit": "iter/sec",
            "range": "stddev: 0.0000447390406408",
            "extra": "mean: 1.69727832251 msec\nrounds: 608"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 155.223182862,
            "unit": "iter/sec",
            "range": "stddev: 0.000165869723185",
            "extra": "mean: 6.44233665076 msec\nrounds: 149"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2232.5465487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114730430301",
            "extra": "mean: 447.91899214 usec\nrounds: 1883"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 595.589686607,
            "unit": "iter/sec",
            "range": "stddev: 0.0000331737592569",
            "extra": "mean: 1.67900825432 msec\nrounds: 601"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 152.617017323,
            "unit": "iter/sec",
            "range": "stddev: 0.00015026172715",
            "extra": "mean: 6.55234925659 msec\nrounds: 156"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3063.86573146,
            "unit": "iter/sec",
            "range": "stddev: 0.0000289580685137",
            "extra": "mean: 326.385059806 usec\nrounds: 2423"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 821.505047257,
            "unit": "iter/sec",
            "range": "stddev: 0.0000816720540384",
            "extra": "mean: 1.21727797454 msec\nrounds: 825"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 210.916706061,
            "unit": "iter/sec",
            "range": "stddev: 0.000274309586448",
            "extra": "mean: 4.74120812276 msec\nrounds: 212"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3099.87703265,
            "unit": "iter/sec",
            "range": "stddev: 0.0000156675195012",
            "extra": "mean: 322.593441439 usec\nrounds: 2913"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 839.166805825,
            "unit": "iter/sec",
            "range": "stddev: 0.0000423378585036",
            "extra": "mean: 1.19165819365 msec\nrounds: 827"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 213.845503376,
            "unit": "iter/sec",
            "range": "stddev: 0.00015445497707",
            "extra": "mean: 4.67627321694 msec\nrounds: 212"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3703.93690266,
            "unit": "iter/sec",
            "range": "stddev: 0.00000731536599458",
            "extra": "mean: 269.983000866 usec\nrounds: 3043"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 992.558764429,
            "unit": "iter/sec",
            "range": "stddev: 0.0000380353938484",
            "extra": "mean: 1.00749702268 msec\nrounds: 970"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 253.936908064,
            "unit": "iter/sec",
            "range": "stddev: 0.000119496655843",
            "extra": "mean: 3.93798604395 msec\nrounds: 250"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3614.88419127,
            "unit": "iter/sec",
            "range": "stddev: 0.0000340549265789",
            "extra": "mean: 276.634035031 usec\nrounds: 3256"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 984.242458088,
            "unit": "iter/sec",
            "range": "stddev: 0.0000626072128285",
            "extra": "mean: 1.01600981728 msec\nrounds: 980"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 235.645775369,
            "unit": "iter/sec",
            "range": "stddev: 0.000549343375505",
            "extra": "mean: 4.24365766131 msec\nrounds: 251"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 179.925688873,
            "unit": "iter/sec",
            "range": "stddev: 0.000522846002514",
            "extra": "mean: 5.55785005613 msec\nrounds: 160"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 91.3852810799,
            "unit": "iter/sec",
            "range": "stddev: 0.00111727469364",
            "extra": "mean: 10.9426812303 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 151.951393192,
            "unit": "iter/sec",
            "range": "stddev: 0.000413753582325",
            "extra": "mean: 6.58105186792 msec\nrounds: 144"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 87.6985710236,
            "unit": "iter/sec",
            "range": "stddev: 0.000786301045559",
            "extra": "mean: 11.4026943464 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 12.5155471977,
            "unit": "iter/sec",
            "range": "stddev: 0.00260838976319",
            "extra": "mean: 79.9006215395 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.29542734278,
            "unit": "iter/sec",
            "range": "stddev: 0.00159779341143",
            "extra": "mean: 158.845451715 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 12.5864378095,
            "unit": "iter/sec",
            "range": "stddev: 0.00191766523467",
            "extra": "mean: 79.4505971538 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 5.53677479084,
            "unit": "iter/sec",
            "range": "stddev: 0.0151562870822",
            "extra": "mean: 180.6105608 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.0957019606,
            "unit": "iter/sec",
            "range": "stddev: 0.00102353759048",
            "extra": "mean: 43.2980994345 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.6745579786,
            "unit": "iter/sec",
            "range": "stddev: 0.00138421256408",
            "extra": "mean: 85.656347918 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 22.1769020722,
            "unit": "iter/sec",
            "range": "stddev: 0.00427764551143",
            "extra": "mean: 45.0919608495 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.8221825406,
            "unit": "iter/sec",
            "range": "stddev: 0.00084151933731",
            "extra": "mean: 84.5867500836 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 27.5295381423,
            "unit": "iter/sec",
            "range": "stddev: 0.000627027210901",
            "extra": "mean: 36.3246195715 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 13.8033939536,
            "unit": "iter/sec",
            "range": "stddev: 0.000300275108239",
            "extra": "mean: 72.4459508555 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 27.4236036506,
            "unit": "iter/sec",
            "range": "stddev: 0.000149998661838",
            "extra": "mean: 36.46493775 msec\nrounds: 28"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 13.7865389637,
            "unit": "iter/sec",
            "range": "stddev: 0.000247273955863",
            "extra": "mean: 72.5345210012 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6202.6989025,
            "unit": "iter/sec",
            "range": "stddev: 0.0000208911016766",
            "extra": "mean: 161.220142348 usec\nrounds: 42"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6725.66291712,
            "unit": "iter/sec",
            "range": "stddev: 0.00000911017760797",
            "extra": "mean: 148.684228205 usec\nrounds: 1318"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6682.57607708,
            "unit": "iter/sec",
            "range": "stddev: 0.00000567992872217",
            "extra": "mean: 149.642890476 usec\nrounds: 5005"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 22036.0910905,
            "unit": "iter/sec",
            "range": "stddev: 0.00000301918810386",
            "extra": "mean: 45.380099215 usec\nrounds: 11513"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 22389.51234,
            "unit": "iter/sec",
            "range": "stddev: 0.00000331244374127",
            "extra": "mean: 44.6637686795 usec\nrounds: 11573"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10363.9431227,
            "unit": "iter/sec",
            "range": "stddev: 0.00000877335871818",
            "extra": "mean: 96.4883720565 usec\nrounds: 188"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10431.8527701,
            "unit": "iter/sec",
            "range": "stddev: 0.00000956234659113",
            "extra": "mean: 95.8602486095 usec\nrounds: 7917"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 34.3615852828,
            "unit": "iter/sec",
            "range": "stddev: 0.000219550789146",
            "extra": "mean: 29.1022661431 msec\nrounds: 35"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 94.8659219893,
            "unit": "iter/sec",
            "range": "stddev: 0.0120869732303",
            "extra": "mean: 10.5411930758 msec\nrounds: 106"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 96.9063216064,
            "unit": "iter/sec",
            "range": "stddev: 0.000103629321348",
            "extra": "mean: 10.3192442291 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 94.7993434469,
            "unit": "iter/sec",
            "range": "stddev: 0.000104195832904",
            "extra": "mean: 10.5485962628 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3859.01486878,
            "unit": "iter/sec",
            "range": "stddev: 0.00000473875667528",
            "extra": "mean: 259.133492356 usec\nrounds: 3782"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 291.206977567,
            "unit": "iter/sec",
            "range": "stddev: 0.0000372904866996",
            "extra": "mean: 3.43398365092 msec\nrounds: 252"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 440.208087639,
            "unit": "iter/sec",
            "range": "stddev: 0.000208134506967",
            "extra": "mean: 2.27165294796 msec\nrounds: 441"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 433.701833167,
            "unit": "iter/sec",
            "range": "stddev: 0.000285717175233",
            "extra": "mean: 2.30573154994 msec\nrounds: 422"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 472.297118356,
            "unit": "iter/sec",
            "range": "stddev: 0.0000959491806463",
            "extra": "mean: 2.11731124568 msec\nrounds: 480"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 624.687871834,
            "unit": "iter/sec",
            "range": "stddev: 0.0000660349287718",
            "extra": "mean: 1.60079944735 msec\nrounds: 628"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 186.481829969,
            "unit": "iter/sec",
            "range": "stddev: 0.000374944139761",
            "extra": "mean: 5.36245273959 msec\nrounds: 192"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 111.651461095,
            "unit": "iter/sec",
            "range": "stddev: 0.000209232137478",
            "extra": "mean: 8.95644347319 msec\nrounds: 112"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 138.746216069,
            "unit": "iter/sec",
            "range": "stddev: 0.0000939420745316",
            "extra": "mean: 7.20740376445 msec\nrounds: 140"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 253.51551476,
            "unit": "iter/sec",
            "range": "stddev: 0.000201999244607",
            "extra": "mean: 3.94453176148 msec\nrounds: 260"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 102.389409375,
            "unit": "iter/sec",
            "range": "stddev: 0.00108815505337",
            "extra": "mean: 9.76663510519 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.6305911687,
            "unit": "iter/sec",
            "range": "stddev: 0.00265845850843",
            "extra": "mean: 94.0681458 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.35434900399,
            "unit": "iter/sec",
            "range": "stddev: 0.00387206380292",
            "extra": "mean: 229.6554546 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.448221498845,
            "unit": "iter/sec",
            "range": "stddev: 0.0202816956349",
            "extra": "mean: 2.231039793 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 41.7531949428,
            "unit": "iter/sec",
            "range": "stddev: 0.000272464620458",
            "extra": "mean: 23.9502630007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.43322456825,
            "unit": "iter/sec",
            "range": "stddev: 0.029853355273",
            "extra": "mean: 106.008289399 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.120886807,
            "unit": "iter/sec",
            "range": "stddev: 0.00209894701258",
            "extra": "mean: 98.8055710011 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 40.5934542163,
            "unit": "iter/sec",
            "range": "stddev: 0.00135488926805",
            "extra": "mean: 24.6345136009 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4802.93607144,
            "unit": "iter/sec",
            "range": "stddev: 0.00000737002942819",
            "extra": "mean: 208.205977578 usec\nrounds: 1791"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8071.73370403,
            "unit": "iter/sec",
            "range": "stddev: 0.000011351904239",
            "extra": "mean: 123.889121799 usec\nrounds: 6369"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6451.89121853,
            "unit": "iter/sec",
            "range": "stddev: 0.00000748715927948",
            "extra": "mean: 154.993313763 usec\nrounds: 3002"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5793.87327305,
            "unit": "iter/sec",
            "range": "stddev: 0.0000144274326034",
            "extra": "mean: 172.59611194 usec\nrounds: 5397"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 20246.2703478,
            "unit": "iter/sec",
            "range": "stddev: 0.00000267147693023",
            "extra": "mean: 49.3918130512 usec\nrounds: 15714"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30658.0268092,
            "unit": "iter/sec",
            "range": "stddev: 0.00000126230401269",
            "extra": "mean: 32.6178852352 usec\nrounds: 25913"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 57890.4756027,
            "unit": "iter/sec",
            "range": "stddev: 0.00000360851253533",
            "extra": "mean: 17.2739986948 usec\nrounds: 36311"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15452.3632379,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117566699684",
            "extra": "mean: 64.7150202595 usec\nrounds: 14443"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3699.05354863,
            "unit": "iter/sec",
            "range": "stddev: 0.0000058452482378",
            "extra": "mean: 270.339422464 usec\nrounds: 1328"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 4640.5673469,
            "unit": "iter/sec",
            "range": "stddev: 0.000035646437961",
            "extra": "mean: 215.490892653 usec\nrounds: 5004"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5725.54694277,
            "unit": "iter/sec",
            "range": "stddev: 0.00000481286525448",
            "extra": "mean: 174.655803192 usec\nrounds: 3242"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 15851.9645027,
            "unit": "iter/sec",
            "range": "stddev: 0.00000586837676442",
            "extra": "mean: 63.0836638467 usec\nrounds: 13541"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4007.50121685,
            "unit": "iter/sec",
            "range": "stddev: 0.0000199646981382",
            "extra": "mean: 249.532051494 usec\nrounds: 1343"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1833.53598707,
            "unit": "iter/sec",
            "range": "stddev: 0.000022112816992",
            "extra": "mean: 545.394258446 usec\nrounds: 1919"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4247.80658518,
            "unit": "iter/sec",
            "range": "stddev: 0.00000986850954835",
            "extra": "mean: 235.415615082 usec\nrounds: 3123"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3610.71800747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000208091250057",
            "extra": "mean: 276.953225905 usec\nrounds: 1274"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 12697.032255,
            "unit": "iter/sec",
            "range": "stddev: 0.00000738018950678",
            "extra": "mean: 78.7585618368 usec\nrounds: 11222"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5790.10276032,
            "unit": "iter/sec",
            "range": "stddev: 0.00000646934986322",
            "extra": "mean: 172.708506463 usec\nrounds: 3201"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2251.49956081,
            "unit": "iter/sec",
            "range": "stddev: 0.0000191125260118",
            "extra": "mean: 444.148432185 usec\nrounds: 2219"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4109.30954872,
            "unit": "iter/sec",
            "range": "stddev: 0.00000568944520627",
            "extra": "mean: 243.349883513 usec\nrounds: 1332"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 32214.9751444,
            "unit": "iter/sec",
            "range": "stddev: 0.00000268014922875",
            "extra": "mean: 31.041464273 usec\nrounds: 21983"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53740.5204792,
            "unit": "iter/sec",
            "range": "stddev: 0.00000111774935147",
            "extra": "mean: 18.607932917 usec\nrounds: 45912"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 46035.569488,
            "unit": "iter/sec",
            "range": "stddev: 0.00000137538286895",
            "extra": "mean: 21.722333646 usec\nrounds: 28969"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19328.166372,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126010401784",
            "extra": "mean: 51.7379652449 usec\nrounds: 17106"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3965.70677209,
            "unit": "iter/sec",
            "range": "stddev: 0.000222784681844",
            "extra": "mean: 252.161861042 usec\nrounds: 94"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 311.926004513,
            "unit": "iter/sec",
            "range": "stddev: 0.0000493939841261",
            "extra": "mean: 3.20588852975 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5034.74167019,
            "unit": "iter/sec",
            "range": "stddev: 0.000211260233455",
            "extra": "mean: 198.619922432 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 351.743880492,
            "unit": "iter/sec",
            "range": "stddev: 0.0000401835715949",
            "extra": "mean: 2.84297767626 msec\nrounds: 346"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6246.81626149,
            "unit": "iter/sec",
            "range": "stddev: 0.00000663811395763",
            "extra": "mean: 160.081545245 usec\nrounds: 3907"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18367.6367188,
            "unit": "iter/sec",
            "range": "stddev: 0.0000121742802278",
            "extra": "mean: 54.4435854927 usec\nrounds: 15596"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7533.93078225,
            "unit": "iter/sec",
            "range": "stddev: 0.00000682879446513",
            "extra": "mean: 132.732836139 usec\nrounds: 4376"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21986.3032156,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115836481808",
            "extra": "mean: 45.4828622254 usec\nrounds: 17665"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7568.79552404,
            "unit": "iter/sec",
            "range": "stddev: 0.00000564927774701",
            "extra": "mean: 132.121418372 usec\nrounds: 6859"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21990.4305162,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116869602154",
            "extra": "mean: 45.4743257193 usec\nrounds: 18512"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11659.2905566,
            "unit": "iter/sec",
            "range": "stddev: 0.00000488573156514",
            "extra": "mean: 85.7685118273 usec\nrounds: 9613"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 46114.0677736,
            "unit": "iter/sec",
            "range": "stddev: 0.000011058145009",
            "extra": "mean: 21.6853565144 usec\nrounds: 35001"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15574.9943825,
            "unit": "iter/sec",
            "range": "stddev: 0.00000498281556475",
            "extra": "mean: 64.2054806211 usec\nrounds: 9243"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 47574.7520566,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109087666146",
            "extra": "mean: 21.0195525309 usec\nrounds: 36285"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15485.8912845,
            "unit": "iter/sec",
            "range": "stddev: 0.00000384693172346",
            "extra": "mean: 64.5749076775 usec\nrounds: 10207"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 47387.9082217,
            "unit": "iter/sec",
            "range": "stddev: 0.000010831473081",
            "extra": "mean: 21.1024296604 usec\nrounds: 37763"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11492.9404448,
            "unit": "iter/sec",
            "range": "stddev: 0.00000558455916506",
            "extra": "mean: 87.0099349079 usec\nrounds: 9967"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 47233.5848626,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108712941778",
            "extra": "mean: 21.1713763185 usec\nrounds: 36656"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15497.1302455,
            "unit": "iter/sec",
            "range": "stddev: 0.00000390861407384",
            "extra": "mean: 64.5280761119 usec\nrounds: 13332"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 46944.9775451,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110310575233",
            "extra": "mean: 21.3015332479 usec\nrounds: 40323"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15205.2137159,
            "unit": "iter/sec",
            "range": "stddev: 0.00000397979535609",
            "extra": "mean: 65.7669151307 usec\nrounds: 12852"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 47421.3430215,
            "unit": "iter/sec",
            "range": "stddev: 0.0000110859821187",
            "extra": "mean: 21.0875512224 usec\nrounds: 37793"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 33367.1220901,
            "unit": "iter/sec",
            "range": "stddev: 0.00000292307602297",
            "extra": "mean: 29.9696209131 usec\nrounds: 23100"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6483.58824891,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135861695516",
            "extra": "mean: 154.235580917 usec\nrounds: 4286"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2577.93510487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000215779783877",
            "extra": "mean: 387.907359696 usec\nrounds: 2502"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 5837.41186276,
            "unit": "iter/sec",
            "range": "stddev: 0.0000387421094439",
            "extra": "mean: 171.308796349 usec\nrounds: 2774"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 5325.40962514,
            "unit": "iter/sec",
            "range": "stddev: 0.0000312375106375",
            "extra": "mean: 187.778982349 usec\nrounds: 3276"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5530.46308206,
            "unit": "iter/sec",
            "range": "stddev: 0.00000745684512573",
            "extra": "mean: 180.816684817 usec\nrounds: 3475"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5830.11778236,
            "unit": "iter/sec",
            "range": "stddev: 0.00000677296228915",
            "extra": "mean: 171.523121372 usec\nrounds: 5590"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 4966.00483968,
            "unit": "iter/sec",
            "range": "stddev: 0.000019731186496",
            "extra": "mean: 201.36911507 usec\nrounds: 3409"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 108581.190563,
            "unit": "iter/sec",
            "range": "stddev: 0.00000141279587015",
            "extra": "mean: 9.20969824344 usec\nrounds: 84176"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 40507.6865856,
            "unit": "iter/sec",
            "range": "stddev: 0.00000305117565672",
            "extra": "mean: 24.6866726859 usec\nrounds: 24655"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 65050.1299033,
            "unit": "iter/sec",
            "range": "stddev: 0.00000204920549071",
            "extra": "mean: 15.3727594624 usec\nrounds: 55097"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 32962.3458841,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031138016884",
            "extra": "mean: 30.3376465837 usec\nrounds: 18660"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 36337.0693524,
            "unit": "iter/sec",
            "range": "stddev: 0.00000306052167563",
            "extra": "mean: 27.5201059915 usec\nrounds: 32691"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 23064.4992831,
            "unit": "iter/sec",
            "range": "stddev: 0.00000371420649025",
            "extra": "mean: 43.3566750237 usec\nrounds: 16958"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 37757.2408172,
            "unit": "iter/sec",
            "range": "stddev: 0.00000416372276746",
            "extra": "mean: 26.4849861472 usec\nrounds: 29113"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8233.11799333,
            "unit": "iter/sec",
            "range": "stddev: 0.00000702158205072",
            "extra": "mean: 121.460666641 usec\nrounds: 7074"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 30216.7667119,
            "unit": "iter/sec",
            "range": "stddev: 0.00000406106274212",
            "extra": "mean: 33.094209236 usec\nrounds: 21778"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 9215.16274059,
            "unit": "iter/sec",
            "range": "stddev: 0.0000113039439617",
            "extra": "mean: 108.516803029 usec\nrounds: 5553"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5233.91143128,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115007789125",
            "extra": "mean: 191.061696998 usec\nrounds: 307"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3602.55705121,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127683247567",
            "extra": "mean: 277.580614488 usec\nrounds: 3453"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 2999.99760528,
            "unit": "iter/sec",
            "range": "stddev: 0.0000387017082457",
            "extra": "mean: 333.333599413 usec\nrounds: 1705"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 250.835029309,
            "unit": "iter/sec",
            "range": "stddev: 0.0017917704743",
            "extra": "mean: 3.98668400803 msec\nrounds: 1002"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6251.96979739,
            "unit": "iter/sec",
            "range": "stddev: 0.00000722904512808",
            "extra": "mean: 159.949589075 usec\nrounds: 5225"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 5741.25100201,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155177498774",
            "extra": "mean: 174.178066705 usec\nrounds: 5356"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 618.926317736,
            "unit": "iter/sec",
            "range": "stddev: 0.0000698034624932",
            "extra": "mean: 1.61570120925 msec\nrounds: 593"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 749.806067034,
            "unit": "iter/sec",
            "range": "stddev: 0.0000556911609071",
            "extra": "mean: 1.33367819222 msec\nrounds: 661"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1038.99635452,
            "unit": "iter/sec",
            "range": "stddev: 0.0000975608534828",
            "extra": "mean: 962.46728456 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10334.2439882,
            "unit": "iter/sec",
            "range": "stddev: 0.00000387808863009",
            "extra": "mean: 96.7656657941 usec\nrounds: 8664"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7860.1402247,
            "unit": "iter/sec",
            "range": "stddev: 0.00000737566472825",
            "extra": "mean: 127.224193388 usec\nrounds: 6675"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4706.63888137,
            "unit": "iter/sec",
            "range": "stddev: 0.00000819970895962",
            "extra": "mean: 212.465843504 usec\nrounds: 3226"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2370.64951184,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155297241908",
            "extra": "mean: 421.825324666 usec\nrounds: 2310"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2831.31987561,
            "unit": "iter/sec",
            "range": "stddev: 0.000019841059093",
            "extra": "mean: 353.192166174 usec\nrounds: 2581"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3216.41543733,
            "unit": "iter/sec",
            "range": "stddev: 0.0000139993929535",
            "extra": "mean: 310.905111446 usec\nrounds: 2332"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1192.08112341,
            "unit": "iter/sec",
            "range": "stddev: 0.000024635849644",
            "extra": "mean: 838.869083957 usec\nrounds: 1049"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1275.12937131,
            "unit": "iter/sec",
            "range": "stddev: 0.000051194983861",
            "extra": "mean: 784.234151057 usec\nrounds: 1191"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2096.20012749,
            "unit": "iter/sec",
            "range": "stddev: 0.0000137662002052",
            "extra": "mean: 477.053687234 usec\nrounds: 1240"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 759.397903112,
            "unit": "iter/sec",
            "range": "stddev: 0.0000298697941254",
            "extra": "mean: 1.31683271168 msec\nrounds: 742"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1338.06829062,
            "unit": "iter/sec",
            "range": "stddev: 0.0000206790305895",
            "extra": "mean: 747.346011416 usec\nrounds: 1302"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 1987.40453471,
            "unit": "iter/sec",
            "range": "stddev: 0.000012275959936",
            "extra": "mean: 503.168822722 usec\nrounds: 1230"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1138.95065883,
            "unit": "iter/sec",
            "range": "stddev: 0.0000205354316407",
            "extra": "mean: 878.001160319 usec\nrounds: 1186"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1352.91008305,
            "unit": "iter/sec",
            "range": "stddev: 0.000023610554998",
            "extra": "mean: 739.147421936 usec\nrounds: 1306"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2158.78034023,
            "unit": "iter/sec",
            "range": "stddev: 0.0000218464430269",
            "extra": "mean: 463.224526074 usec\nrounds: 1091"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 737.195388126,
            "unit": "iter/sec",
            "range": "stddev: 0.0000163287483495",
            "extra": "mean: 1.35649247962 msec\nrounds: 732"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1193.59821746,
            "unit": "iter/sec",
            "range": "stddev: 0.000039435518897",
            "extra": "mean: 837.802859765 usec\nrounds: 1176"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1320.799437,
            "unit": "iter/sec",
            "range": "stddev: 0.0000884232605608",
            "extra": "mean: 757.117221573 usec\nrounds: 948"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 332.288648605,
            "unit": "iter/sec",
            "range": "stddev: 0.000166963279846",
            "extra": "mean: 3.00943172209 msec\nrounds: 331"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1283.59603902,
            "unit": "iter/sec",
            "range": "stddev: 0.0000281017292696",
            "extra": "mean: 779.061300906 usec\nrounds: 1018"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1930.66392942,
            "unit": "iter/sec",
            "range": "stddev: 0.000037237783032",
            "extra": "mean: 517.956535448 usec\nrounds: 125"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1001.579357,
            "unit": "iter/sec",
            "range": "stddev: 0.0000439419207598",
            "extra": "mean: 998.423133434 usec\nrounds: 929"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1022.53470298,
            "unit": "iter/sec",
            "range": "stddev: 0.0000385321548226",
            "extra": "mean: 977.96191864 usec\nrounds: 1020"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1123.68487068,
            "unit": "iter/sec",
            "range": "stddev: 0.0000359075630772",
            "extra": "mean: 889.929219566 usec\nrounds: 1116"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1125.8695926,
            "unit": "iter/sec",
            "range": "stddev: 0.000036176449264",
            "extra": "mean: 888.202334066 usec\nrounds: 1119"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.134063103,
            "unit": "iter/sec",
            "range": "stddev: 0.000117946303485",
            "extra": "mean: 26.9294528106 msec\nrounds: 37"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 37.2676080218,
            "unit": "iter/sec",
            "range": "stddev: 0.000133035655417",
            "extra": "mean: 26.832953685 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 37.4719720627,
            "unit": "iter/sec",
            "range": "stddev: 0.0000725109667725",
            "extra": "mean: 26.6866125521 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 37.4474577423,
            "unit": "iter/sec",
            "range": "stddev: 0.000114692916845",
            "extra": "mean: 26.7040824742 msec\nrounds: 38"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peterdsharpe@gmail.com",
            "name": "Peter Sharpe",
            "username": "peterdsharpe"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8b8945b9301d4ba92c2dfbc0c9759311541d165e",
          "message": "[Doc] Stop the stable 0.15 docs from saying they are an unstable development version (#3062)",
          "timestamp": "2026-10-19T17:56:14-04:00",
          "tree_id": "d74c4be36ec2186a15412ae0a9daae3497193a4d",
          "url": "https://github.com/pytorch/tensordict/commit/8b8945b9301d4ba92c2dfbc0c9759311541d165e"
        },
        "date": 1792447665642,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-2]",
            "value": 4640.12409612,
            "unit": "iter/sec",
            "range": "stddev: 0.00000634226857237",
            "extra": "mean: 215.511477556 usec\nrounds: 134"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-8]",
            "value": 3688.75206963,
            "unit": "iter/sec",
            "range": "stddev: 0.00000683343902859",
            "extra": "mean: 271.094392121 usec\nrounds: 3262"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-32-32]",
            "value": 2072.69922486,
            "unit": "iter/sec",
            "range": "stddev: 0.00000856691111257",
            "extra": "mean: 482.462668972 usec\nrounds: 1840"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-2]",
            "value": 4640.56256933,
            "unit": "iter/sec",
            "range": "stddev: 0.0000061257252098",
            "extra": "mean: 215.491114506 usec\nrounds: 3720"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-8]",
            "value": 3669.08898749,
            "unit": "iter/sec",
            "range": "stddev: 0.00000878436966623",
            "extra": "mean: 272.547219054 usec\nrounds: 3191"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[stack_out-cuda-4096-32]",
            "value": 1797.80603964,
            "unit": "iter/sec",
            "range": "stddev: 0.000041742882462",
            "extra": "mean: 556.233530176 usec\nrounds: 1677"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-2]",
            "value": 2324.78611832,
            "unit": "iter/sec",
            "range": "stddev: 0.0000150224299047",
            "extra": "mean: 430.147097025 usec\nrounds: 72"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-8]",
            "value": 624.181270666,
            "unit": "iter/sec",
            "range": "stddev: 0.000024048455224",
            "extra": "mean: 1.60209869632 msec\nrounds: 609"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-32-32]",
            "value": 158.902271332,
            "unit": "iter/sec",
            "range": "stddev: 0.000068649841155",
            "extra": "mean: 6.2931762499 msec\nrounds: 156"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-2]",
            "value": 2321.54036105,
            "unit": "iter/sec",
            "range": "stddev: 0.0000134127191344",
            "extra": "mean: 430.748487848 usec\nrounds: 2062"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-8]",
            "value": 616.238743453,
            "unit": "iter/sec",
            "range": "stddev: 0.0000352502372136",
            "extra": "mean: 1.62274769418 msec\nrounds: 582"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[set_at_loop-cuda-4096-32]",
            "value": 157.76206641,
            "unit": "iter/sec",
            "range": "stddev: 0.000102168149264",
            "extra": "mean: 6.33865936696 msec\nrounds: 158"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-2]",
            "value": 3170.60305888,
            "unit": "iter/sec",
            "range": "stddev: 0.0000108878677542",
            "extra": "mean: 315.397412236 usec\nrounds: 2399"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-8]",
            "value": 857.767093066,
            "unit": "iter/sec",
            "range": "stddev: 0.0000179907282369",
            "extra": "mean: 1.16581763055 msec\nrounds: 820"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-32-32]",
            "value": 218.327648093,
            "unit": "iter/sec",
            "range": "stddev: 0.0000441170471911",
            "extra": "mean: 4.58027193869 msec\nrounds: 212"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-2]",
            "value": 3126.89766479,
            "unit": "iter/sec",
            "range": "stddev: 0.0000186616401798",
            "extra": "mean: 319.805797055 usec\nrounds: 2932"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-8]",
            "value": 838.800612465,
            "unit": "iter/sec",
            "range": "stddev: 0.0000532213974461",
            "extra": "mean: 1.19217843328 msec\nrounds: 824"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[copy_at-cuda-4096-32]",
            "value": 216.487133056,
            "unit": "iter/sec",
            "range": "stddev: 0.0000942535423382",
            "extra": "mean: 4.61921217157 msec\nrounds: 210"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-2]",
            "value": 3662.00124777,
            "unit": "iter/sec",
            "range": "stddev: 0.000011858231806",
            "extra": "mean: 273.074729455 usec\nrounds: 3057"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-8]",
            "value": 1001.15318595,
            "unit": "iter/sec",
            "range": "stddev: 0.0000344111790743",
            "extra": "mean: 998.848142358 usec\nrounds: 976"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-32-32]",
            "value": 256.681499248,
            "unit": "iter/sec",
            "range": "stddev: 0.0000897199397651",
            "extra": "mean: 3.89587875609 msec\nrounds: 254"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-2]",
            "value": 3732.29863924,
            "unit": "iter/sec",
            "range": "stddev: 0.0000100737794377",
            "extra": "mean: 267.931400099 usec\nrounds: 3404"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-8]",
            "value": 994.244333348,
            "unit": "iter/sec",
            "range": "stddev: 0.0000303231612754",
            "extra": "mean: 1.00578898613 msec\nrounds: 932"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_collector_preallocated_write[manual_leaf_copy-cuda-4096-32]",
            "value": 250.721157958,
            "unit": "iter/sec",
            "range": "stddev: 0.000114678795381",
            "extra": "mean: 3.98849466134 msec\nrounds: 251"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-128]",
            "value": 191.223096579,
            "unit": "iter/sec",
            "range": "stddev: 0.0000309467019738",
            "extra": "mean: 5.22949381059 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-128-256]",
            "value": 98.2077935249,
            "unit": "iter/sec",
            "range": "stddev: 0.000367448732204",
            "extra": "mean: 10.1824912678 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-128]",
            "value": 181.845663585,
            "unit": "iter/sec",
            "range": "stddev: 0.0000750390039547",
            "extra": "mean: 5.49916880219 msec\nrounds: 177"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[stack_out-cuda-4096-256]",
            "value": 94.3427199294,
            "unit": "iter/sec",
            "range": "stddev: 0.000198407495942",
            "extra": "mean: 10.5996520002 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-128]",
            "value": 13.4904983457,
            "unit": "iter/sec",
            "range": "stddev: 0.000784950661794",
            "extra": "mean: 74.1262460716 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-128-256]",
            "value": 6.71629427976,
            "unit": "iter/sec",
            "range": "stddev: 0.000821090691269",
            "extra": "mean: 148.891629572 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-128]",
            "value": 13.2941743046,
            "unit": "iter/sec",
            "range": "stddev: 0.0013085254425",
            "extra": "mean: 75.2209183577 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[set_at_loop-cuda-4096-256]",
            "value": 6.57013839361,
            "unit": "iter/sec",
            "range": "stddev: 0.00112618015781",
            "extra": "mean: 152.203795429 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-128]",
            "value": 23.9702643116,
            "unit": "iter/sec",
            "range": "stddev: 0.000169351039566",
            "extra": "mean: 41.718355167 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-128-256]",
            "value": 11.9647606809,
            "unit": "iter/sec",
            "range": "stddev: 0.000536468671536",
            "extra": "mean: 83.5787715835 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-128]",
            "value": 23.7168729043,
            "unit": "iter/sec",
            "range": "stddev: 0.000319811677615",
            "extra": "mean: 42.1640746667 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[copy_at-cuda-4096-256]",
            "value": 11.9183862638,
            "unit": "iter/sec",
            "range": "stddev: 0.000394612592081",
            "extra": "mean: 83.9039764166 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-128]",
            "value": 28.5766233794,
            "unit": "iter/sec",
            "range": "stddev: 0.000281534426926",
            "extra": "mean: 34.993637517 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-128-256]",
            "value": 14.3460671518,
            "unit": "iter/sec",
            "range": "stddev: 0.000923875777724",
            "extra": "mean: 69.7055150667 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-128]",
            "value": 28.6445307403,
            "unit": "iter/sec",
            "range": "stddev: 0.000234011536276",
            "extra": "mean: 34.9106783793 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/common/collector_write_benchmarks_test.py::test_wide_nested_preallocated_write[manual_leaf_copy-cuda-4096-256]",
            "value": 14.2728456162,
            "unit": "iter/sec",
            "range": "stddev: 0.000605324289159",
            "extra": "mean: 70.0631133334 msec\nrounds: 15"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_getitem_mask_many_leaves[cuda]",
            "value": 6486.16533955,
            "unit": "iter/sec",
            "range": "stddev: 0.0000175635922281",
            "extra": "mean: 154.17429986 usec\nrounds: 40"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-None]",
            "value": 6755.02608692,
            "unit": "iter/sec",
            "range": "stddev: 0.00000943674687263",
            "extra": "mean: 148.037918305 usec\nrounds: 1395"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_setitem_rows_store_with_device[cuda-cpu]",
            "value": 6677.30335075,
            "unit": "iter/sec",
            "range": "stddev: 0.00000659855437239",
            "extra": "mean: 149.761055844 usec\nrounds: 4675"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to",
            "value": 24545.9042042,
            "unit": "iter/sec",
            "range": "stddev: 0.00000294876288534",
            "extra": "mean: 40.7399944072 usec\nrounds: 11676"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_to_nonblocking",
            "value": 24764.5398264,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030542545908",
            "extra": "mean: 40.38031827 usec\nrounds: 12018"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[add]",
            "value": 10834.0811445,
            "unit": "iter/sec",
            "range": "stddev: 0.00000736503373193",
            "extra": "mean: 92.3013208656 usec\nrounds: 187"
          },
          {
            "name": "benchmarks/common/common_ops_test.py::test_add_cuda_scalar_tensor[sub]",
            "value": 10980.0311787,
            "unit": "iter/sec",
            "range": "stddev: 0.00000571777738294",
            "extra": "mean: 91.0744226248 usec\nrounds: 8135"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_blocking",
            "value": 55.0467653103,
            "unit": "iter/sec",
            "range": "stddev: 0.000261811320207",
            "extra": "mean: 18.1663717089 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_global_sync",
            "value": 94.77223901,
            "unit": "iter/sec",
            "range": "stddev: 0.0121891300158",
            "extra": "mean: 10.5516131142 msec\nrounds: 105"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_event_sync",
            "value": 97.1762869798,
            "unit": "iter/sec",
            "range": "stddev: 0.000262961714289",
            "extra": "mean: 10.2905763441 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_to_cpu_default",
            "value": 94.8947988707,
            "unit": "iter/sec",
            "range": "stddev: 0.000243886802652",
            "extra": "mean: 10.5379853469 msec\nrounds: 98"
          },
          {
            "name": "benchmarks/common/cpu_transfer_sync_test.py::TestCpuTransferSync::test_cpu_to_cpu_default",
            "value": 3879.50770362,
            "unit": "iter/sec",
            "range": "stddev: 0.00000777678660751",
            "extra": "mean: 257.764664075 usec\nrounds: 3858"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[False-None]",
            "value": 292.899942153,
            "unit": "iter/sec",
            "range": "stddev: 0.0000931300188197",
            "extra": "mean: 3.41413519118 msec\nrounds: 293"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[default-None]",
            "value": 444.882945757,
            "unit": "iter/sec",
            "range": "stddev: 0.000218789177763",
            "extra": "mean: 2.24778227517 msec\nrounds: 425"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestConsolidate::test_consolidate[reduce-overhead-None]",
            "value": 442.635418637,
            "unit": "iter/sec",
            "range": "stddev: 0.000296806739976",
            "extra": "mean: 2.25919562217 msec\nrounds: 458"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[False-False-None]",
            "value": 507.541712258,
            "unit": "iter/sec",
            "range": "stddev: 0.0000284836998672",
            "extra": "mean: 1.97028140909 msec\nrounds: 494"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-False-None]",
            "value": 652.992131203,
            "unit": "iter/sec",
            "range": "stddev: 0.0000393620669349",
            "extra": "mean: 1.53141202201 msec\nrounds: 636"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[within-False-None]",
            "value": 198.368810465,
            "unit": "iter/sec",
            "range": "stddev: 0.000039163553254",
            "extra": "mean: 5.04111507074 msec\nrounds: 198"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to[True-default-None]",
            "value": 114.39729397,
            "unit": "iter/sec",
            "range": "stddev: 0.000246713805403",
            "extra": "mean: 8.74146551285 msec\nrounds: 117"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[False-False-None]",
            "value": 139.474549771,
            "unit": "iter/sec",
            "range": "stddev: 0.000214196940264",
            "extra": "mean: 7.16976682587 msec\nrounds: 132"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[True-False-None]",
            "value": 264.78412323,
            "unit": "iter/sec",
            "range": "stddev: 0.0000404552318034",
            "extra": "mean: 3.77666148483 msec\nrounds: 262"
          },
          {
            "name": "benchmarks/common/h2d_test.py::TestTo::test_to_njt[within-False-None]",
            "value": 110.802209771,
            "unit": "iter/sec",
            "range": "stddev: 0.0000771956451659",
            "extra": "mean: 9.02509076365 msec\nrounds: 110"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model",
            "value": 10.635202689,
            "unit": "iter/sec",
            "range": "stddev: 0.00124118295815",
            "extra": "mean: 94.0273569999 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_archive",
            "value": 4.394954406,
            "unit": "iter/sec",
            "range": "stddev: 0.0138572019004",
            "extra": "mean: 227.533646 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_zarr",
            "value": 0.410282934615,
            "unit": "iter/sec",
            "range": "stddev: 0.535725135141",
            "extra": "mean: 2.4373424182 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_model_pickle",
            "value": 40.8627025278,
            "unit": "iter/sec",
            "range": "stddev: 0.000637674639307",
            "extra": "mean: 24.4721944007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights",
            "value": 9.70174729366,
            "unit": "iter/sec",
            "range": "stddev: 0.0250245856468",
            "extra": "mean: 103.0742164 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_returnearly",
            "value": 10.245837372,
            "unit": "iter/sec",
            "range": "stddev: 0.00138887893322",
            "extra": "mean: 97.6006121993 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/common/memmap_benchmarks_test.py::test_serialize_weights_pickle",
            "value": 40.0814056573,
            "unit": "iter/sec",
            "range": "stddev: 0.00107069983891",
            "extra": "mean: 24.9492247989 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-compile]",
            "value": 4678.26067943,
            "unit": "iter/sec",
            "range": "stddev: 0.00000950528033975",
            "extra": "mean: 213.754655528 usec\nrounds: 1762"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[tensordict-eager]",
            "value": 8220.20300597,
            "unit": "iter/sec",
            "range": "stddev: 0.00000799924519922",
            "extra": "mean: 121.651496839 usec\nrounds: 6882"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-compile]",
            "value": 6587.64872665,
            "unit": "iter/sec",
            "range": "stddev: 0.00000574123508637",
            "extra": "mean: 151.79922936 usec\nrounds: 3078"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_nested[pytree-eager]",
            "value": 5850.30451685,
            "unit": "iter/sec",
            "range": "stddev: 0.000013903586445",
            "extra": "mean: 170.931273256 usec\nrounds: 5423"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-compile]",
            "value": 20740.5169378,
            "unit": "iter/sec",
            "range": "stddev: 0.00000369915123844",
            "extra": "mean: 48.214805976 usec\nrounds: 15828"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[tensordict-eager]",
            "value": 30938.7902981,
            "unit": "iter/sec",
            "range": "stddev: 0.00000151994642256",
            "extra": "mean: 32.3218842872 usec\nrounds: 25707"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-compile]",
            "value": 63066.2716724,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021261513214",
            "extra": "mean: 15.8563360967 usec\nrounds: 37950"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_nested[pytree-eager]",
            "value": 15760.7353519,
            "unit": "iter/sec",
            "range": "stddev: 0.0000114260383686",
            "extra": "mean: 63.4488161671 usec\nrounds: 15161"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-compile]",
            "value": 3635.13063015,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106436847727",
            "extra": "mean: 275.093277723 usec\nrounds: 1289"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensordict-eager]",
            "value": 5354.3919639,
            "unit": "iter/sec",
            "range": "stddev: 0.00000719274027161",
            "extra": "mean: 186.762569259 usec\nrounds: 4982"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-compile]",
            "value": 5862.68701448,
            "unit": "iter/sec",
            "range": "stddev: 0.00000703817830475",
            "extra": "mean: 170.570251751 usec\nrounds: 3241"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[tensorclass-eager]",
            "value": 16461.3883191,
            "unit": "iter/sec",
            "range": "stddev: 0.00000391795359945",
            "extra": "mean: 60.748217624 usec\nrounds: 14571"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-compile]",
            "value": 4088.46799162,
            "unit": "iter/sec",
            "range": "stddev: 0.00000809595083",
            "extra": "mean: 244.590394752 usec\nrounds: 1353"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_one_flat[pytree-eager]",
            "value": 1969.17917107,
            "unit": "iter/sec",
            "range": "stddev: 0.0000185197042249",
            "extra": "mean: 507.825806149 usec\nrounds: 1878"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-eager]",
            "value": 4413.23931915,
            "unit": "iter/sec",
            "range": "stddev: 0.00000644137838804",
            "extra": "mean: 226.590929629 usec\nrounds: 4149"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensordict-compile]",
            "value": 3676.24675962,
            "unit": "iter/sec",
            "range": "stddev: 0.0000063717658302",
            "extra": "mean: 272.016560745 usec\nrounds: 1316"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-eager]",
            "value": 13429.9904524,
            "unit": "iter/sec",
            "range": "stddev: 0.00000315648127855",
            "extra": "mean: 74.4602167474 usec\nrounds: 11520"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[tensorclass-compile]",
            "value": 5639.14242385,
            "unit": "iter/sec",
            "range": "stddev: 0.00000746563640521",
            "extra": "mean: 177.331928304 usec\nrounds: 3306"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-eager]",
            "value": 2216.2508116,
            "unit": "iter/sec",
            "range": "stddev: 0.0000247172677046",
            "extra": "mean: 451.212468718 usec\nrounds: 2204"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_add_self_flat[pytree-compile]",
            "value": 4108.08667754,
            "unit": "iter/sec",
            "range": "stddev: 0.00000713213347414",
            "extra": "mean: 243.422322481 usec\nrounds: 1349"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-compile]",
            "value": 33728.5052216,
            "unit": "iter/sec",
            "range": "stddev: 0.00000285814496671",
            "extra": "mean: 29.648512243 usec\nrounds: 21418"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[tensordict-eager]",
            "value": 53579.4169863,
            "unit": "iter/sec",
            "range": "stddev: 0.00000116805097766",
            "extra": "mean: 18.6638835629 usec\nrounds: 45066"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-compile]",
            "value": 46478.9658149,
            "unit": "iter/sec",
            "range": "stddev: 0.00000208925283401",
            "extra": "mean: 21.5151086619 usec\nrounds: 27307"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_copy_flat[pytree-eager]",
            "value": 19548.448964,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117695820723",
            "extra": "mean: 51.1549536152 usec\nrounds: 17289"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-compile]",
            "value": 3900.82606421,
            "unit": "iter/sec",
            "range": "stddev: 0.000228603186816",
            "extra": "mean: 256.355957313 usec\nrounds: 94"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[tensordict-eager]",
            "value": 316.28958094,
            "unit": "iter/sec",
            "range": "stddev: 0.0000317724001567",
            "extra": "mean: 3.16165963175 msec\nrounds: 315"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-compile]",
            "value": 5063.25040688,
            "unit": "iter/sec",
            "range": "stddev: 0.000212038502416",
            "extra": "mean: 197.501588829 usec\nrounds: 90"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_assign_and_add[pytree-eager]",
            "value": 354.751106893,
            "unit": "iter/sec",
            "range": "stddev: 0.0000781824171005",
            "extra": "mean: 2.81887774434 msec\nrounds: 356"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-compile]",
            "value": 6034.23137835,
            "unit": "iter/sec",
            "range": "stddev: 0.00000881895457962",
            "extra": "mean: 165.721189212 usec\nrounds: 3789"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensordict-eager]",
            "value": 18204.8684808,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127905983071",
            "extra": "mean: 54.9303611314 usec\nrounds: 15092"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-compile]",
            "value": 7104.96467011,
            "unit": "iter/sec",
            "range": "stddev: 0.0000156383377804",
            "extra": "mean: 140.746653422 usec\nrounds: 4386"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-tensorclass-eager]",
            "value": 21362.4135719,
            "unit": "iter/sec",
            "range": "stddev: 0.0000126550418079",
            "extra": "mean: 46.8111899732 usec\nrounds: 17989"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-compile]",
            "value": 7244.26483665,
            "unit": "iter/sec",
            "range": "stddev: 0.00000675077809183",
            "extra": "mean: 138.040232177 usec\nrounds: 6628"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[tensor-pytree-eager]",
            "value": 21594.4588084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120852440471",
            "extra": "mean: 46.3081760406 usec\nrounds: 18488"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-compile]",
            "value": 11322.536843,
            "unit": "iter/sec",
            "range": "stddev: 0.00000532838705214",
            "extra": "mean: 88.319429989 usec\nrounds: 9349"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensordict-eager]",
            "value": 39098.3484711,
            "unit": "iter/sec",
            "range": "stddev: 0.0000117865011079",
            "extra": "mean: 25.5765278868 usec\nrounds: 33367"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-compile]",
            "value": 15316.1765125,
            "unit": "iter/sec",
            "range": "stddev: 0.00000420031255836",
            "extra": "mean: 65.2904462928 usec\nrounds: 12763"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-tensorclass-eager]",
            "value": 47833.4039155,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105041993837",
            "extra": "mean: 20.9058924965 usec\nrounds: 36063"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-compile]",
            "value": 15106.9488266,
            "unit": "iter/sec",
            "range": "stddev: 0.00000427950500129",
            "extra": "mean: 66.1947036081 usec\nrounds: 13357"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[slice-pytree-eager]",
            "value": 47710.2304315,
            "unit": "iter/sec",
            "range": "stddev: 0.0000105731344966",
            "extra": "mean: 20.9598652314 usec\nrounds: 35050"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-compile]",
            "value": 11239.3911761,
            "unit": "iter/sec",
            "range": "stddev: 0.00000469298261968",
            "extra": "mean: 88.9727908149 usec\nrounds: 9981"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensordict-eager]",
            "value": 47238.0050715,
            "unit": "iter/sec",
            "range": "stddev: 0.0000106979640438",
            "extra": "mean: 21.1693952462 usec\nrounds: 37370"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-compile]",
            "value": 15303.5079001,
            "unit": "iter/sec",
            "range": "stddev: 0.00000485814413527",
            "extra": "mean: 65.3444952965 usec\nrounds: 13398"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-tensorclass-eager]",
            "value": 47288.9830457,
            "unit": "iter/sec",
            "range": "stddev: 0.0000104536528199",
            "extra": "mean: 21.1465744365 usec\nrounds: 36969"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-compile]",
            "value": 15396.6887635,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036660649417",
            "extra": "mean: 64.9490299739 usec\nrounds: 13485"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_indexing[int-pytree-eager]",
            "value": 48129.2792338,
            "unit": "iter/sec",
            "range": "stddev: 0.000010489232566",
            "extra": "mean: 20.777373273 usec\nrounds: 36087"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-eager]",
            "value": 33049.6097308,
            "unit": "iter/sec",
            "range": "stddev: 0.00000256782843941",
            "extra": "mean: 30.2575433763 usec\nrounds: 24420"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[single-compile]",
            "value": 6698.9811482,
            "unit": "iter/sec",
            "range": "stddev: 0.0000127567813457",
            "extra": "mean: 149.276431427 usec\nrounds: 4300"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-eager]",
            "value": 2661.34555954,
            "unit": "iter/sec",
            "range": "stddev: 0.0000145413467333",
            "extra": "mean: 375.749776806 usec\nrounds: 2666"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_replace[multi-compile]",
            "value": 5993.20720519,
            "unit": "iter/sec",
            "range": "stddev: 0.0000135037671541",
            "extra": "mean: 166.855569274 usec\nrounds: 3436"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[eager]",
            "value": 6071.17694152,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123028391928",
            "extra": "mean: 164.712708859 usec\nrounds: 4682"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_tc_getattr_20[compile]",
            "value": 5518.14093686,
            "unit": "iter/sec",
            "range": "stddev: 0.00000787944636136",
            "extra": "mean: 181.220452947 usec\nrounds: 3751"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[eager]",
            "value": 5830.11684462,
            "unit": "iter/sec",
            "range": "stddev: 0.00000595515868701",
            "extra": "mean: 171.523148961 usec\nrounds: 5647"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_ttd_getattr_20[compile]",
            "value": 5160.89812729,
            "unit": "iter/sec",
            "range": "stddev: 0.0000120346656366",
            "extra": "mean: 193.764723763 usec\nrounds: 3558"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-eager]",
            "value": 108180.96754,
            "unit": "iter/sec",
            "range": "stddev: 0.00000187945096307",
            "extra": "mean: 9.24377016348 usec\nrounds: 85463"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[20-compile]",
            "value": 39767.5952647,
            "unit": "iter/sec",
            "range": "stddev: 0.00000349465877299",
            "extra": "mean: 25.1461018285 usec\nrounds: 26617"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-eager]",
            "value": 66184.3436748,
            "unit": "iter/sec",
            "range": "stddev: 0.00000148325955103",
            "extra": "mean: 15.1093135397 usec\nrounds: 55158"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[40-compile]",
            "value": 32465.3020159,
            "unit": "iter/sec",
            "range": "stddev: 0.00000290064084654",
            "extra": "mean: 30.8021160409 usec\nrounds: 23441"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-eager]",
            "value": 37213.5519219,
            "unit": "iter/sec",
            "range": "stddev: 0.00000183969715791",
            "extra": "mean: 26.8719310132 usec\nrounds: 33377"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_clone_shallow[80-compile]",
            "value": 24095.7748887,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027173386875",
            "extra": "mean: 41.5010517246 usec\nrounds: 18015"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[eager]",
            "value": 36835.3706281,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024993756713",
            "extra": "mean: 27.1478196893 usec\nrounds: 24790"
          },
          {
            "name": "benchmarks/compile/compile_td_test.py::test_compile_update_inplace[compile]",
            "value": 8524.84245958,
            "unit": "iter/sec",
            "range": "stddev: 0.00000888605616822",
            "extra": "mean: 117.304220546 usec\nrounds: 7118"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[eager]",
            "value": 31396.1965117,
            "unit": "iter/sec",
            "range": "stddev: 0.00000463838259806",
            "extra": "mean: 31.8509918749 usec\nrounds: 21322"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile]",
            "value": 8427.19119897,
            "unit": "iter/sec",
            "range": "stddev: 0.0000240044913749",
            "extra": "mean: 118.663499663 usec\nrounds: 5722"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_add[compile-overhead]",
            "value": 5300.8817487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000109255899074",
            "extra": "mean: 188.647860376 usec\nrounds: 308"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[eager]",
            "value": 3743.00587104,
            "unit": "iter/sec",
            "range": "stddev: 0.0000101462898954",
            "extra": "mean: 267.164956309 usec\nrounds: 3547"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile]",
            "value": 3149.68626558,
            "unit": "iter/sec",
            "range": "stddev: 0.00000880672844438",
            "extra": "mean: 317.491939095 usec\nrounds: 2477"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap[compile-overhead]",
            "value": 255.24168001,
            "unit": "iter/sec",
            "range": "stddev: 0.00173639029145",
            "extra": "mean: 3.91785542221 msec\nrounds: 964"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[False]",
            "value": 6183.11724091,
            "unit": "iter/sec",
            "range": "stddev: 0.00000695314437438",
            "extra": "mean: 161.73071948 usec\nrounds: 5219"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_cudagraph_mod_wrap[True]",
            "value": 6185.64929616,
            "unit": "iter/sec",
            "range": "stddev: 0.00000759212421479",
            "extra": "mean: 161.664516063 usec\nrounds: 5292"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[eager]",
            "value": 641.400251594,
            "unit": "iter/sec",
            "range": "stddev: 0.0000236280046067",
            "extra": "mean: 1.55908888017 msec\nrounds: 617"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile]",
            "value": 694.543314026,
            "unit": "iter/sec",
            "range": "stddev: 0.0000417968446373",
            "extra": "mean: 1.43979501322 msec\nrounds: 379"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_mod_wrap_and_backward[compile-overhead]",
            "value": 1072.29342556,
            "unit": "iter/sec",
            "range": "stddev: 0.0000789947272922",
            "extra": "mean: 932.58055693 usec\nrounds: 88"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[eager]",
            "value": 10254.9099006,
            "unit": "iter/sec",
            "range": "stddev: 0.00000570128977691",
            "extra": "mean: 97.5142648444 usec\nrounds: 8748"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile]",
            "value": 7886.92412181,
            "unit": "iter/sec",
            "range": "stddev: 0.00000529566350745",
            "extra": "mean: 126.792141593 usec\nrounds: 6716"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_add[compile-overhead]",
            "value": 4912.92650664,
            "unit": "iter/sec",
            "range": "stddev: 0.00000630634570315",
            "extra": "mean: 203.544669078 usec\nrounds: 3303"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[eager]",
            "value": 2539.86982355,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123183385756",
            "extra": "mean: 393.720965826 usec\nrounds: 2428"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile]",
            "value": 2916.09568338,
            "unit": "iter/sec",
            "range": "stddev: 0.00000724896874391",
            "extra": "mean: 342.924275668 usec\nrounds: 2717"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_seq_wrap[compile-overhead]",
            "value": 3251.69774692,
            "unit": "iter/sec",
            "range": "stddev: 0.00000711262317334",
            "extra": "mean: 307.531658176 usec\nrounds: 2077"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-eager]",
            "value": 1211.70096263,
            "unit": "iter/sec",
            "range": "stddev: 0.000028640643141",
            "extra": "mean: 825.286131513 usec\nrounds: 1217"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile]",
            "value": 1423.6295717,
            "unit": "iter/sec",
            "range": "stddev: 0.0000130447271499",
            "extra": "mean: 702.429915673 usec\nrounds: 1375"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[False-compile-overhead]",
            "value": 2120.98635678,
            "unit": "iter/sec",
            "range": "stddev: 0.0000111275787232",
            "extra": "mean: 471.478751764 usec\nrounds: 1297"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-eager]",
            "value": 791.402818273,
            "unit": "iter/sec",
            "range": "stddev: 0.0000211395416725",
            "extra": "mean: 1.26357902311 msec\nrounds: 780"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile]",
            "value": 1321.30330313,
            "unit": "iter/sec",
            "range": "stddev: 0.0000215487357843",
            "extra": "mean: 756.828502304 usec\nrounds: 1304"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_runtime[True-compile-overhead]",
            "value": 1995.08555849,
            "unit": "iter/sec",
            "range": "stddev: 0.0000184821553788",
            "extra": "mean: 501.231636779 usec\nrounds: 1148"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-eager]",
            "value": 1271.73242083,
            "unit": "iter/sec",
            "range": "stddev: 0.0000154624399215",
            "extra": "mean: 786.328934942 usec\nrounds: 1244"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile]",
            "value": 1390.16068741,
            "unit": "iter/sec",
            "range": "stddev: 0.0000226175437566",
            "extra": "mean: 719.341302811 usec\nrounds: 1364"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[False-compile-overhead]",
            "value": 2123.24378545,
            "unit": "iter/sec",
            "range": "stddev: 0.0000268313802676",
            "extra": "mean: 470.977476468 usec\nrounds: 1066"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-eager]",
            "value": 758.759031391,
            "unit": "iter/sec",
            "range": "stddev: 0.0000276136260669",
            "extra": "mean: 1.31794147895 msec\nrounds: 737"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile]",
            "value": 1207.57368895,
            "unit": "iter/sec",
            "range": "stddev: 0.0000250364111069",
            "extra": "mean: 828.106813814 usec\nrounds: 1117"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_func_call_cm_runtime[True-compile-overhead]",
            "value": 1334.9337835,
            "unit": "iter/sec",
            "range": "stddev: 0.0000454955619428",
            "extra": "mean: 749.100826092 usec\nrounds: 1035"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[eager]",
            "value": 333.183441948,
            "unit": "iter/sec",
            "range": "stddev: 0.00016836422718",
            "extra": "mean: 3.00134962936 msec\nrounds: 348"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile]",
            "value": 1335.53165592,
            "unit": "iter/sec",
            "range": "stddev: 0.0000115809436603",
            "extra": "mean: 748.765478954 usec\nrounds: 1019"
          },
          {
            "name": "benchmarks/compile/tensordict_nn_test.py::test_vmap_func_call_cm_runtime[compile-overhead]",
            "value": 1930.90102227,
            "unit": "iter/sec",
            "range": "stddev: 0.0000354381435783",
            "extra": "mean: 517.892936235 usec\nrounds: 126"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-True]",
            "value": 1057.02950274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000380042626603",
            "extra": "mean: 946.047387901 usec\nrounds: 972"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[True-False]",
            "value": 1056.15713908,
            "unit": "iter/sec",
            "range": "stddev: 0.0000391888840844",
            "extra": "mean: 946.828803211 usec\nrounds: 1062"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-True]",
            "value": 1144.4359545,
            "unit": "iter/sec",
            "range": "stddev: 0.0000397203283688",
            "extra": "mean: 873.792889913 usec\nrounds: 1135"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_mlp_speed_decorator[False-False]",
            "value": 1136.47386003,
            "unit": "iter/sec",
            "range": "stddev: 0.0000475513315481",
            "extra": "mean: 879.914651068 usec\nrounds: 1155"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-True]",
            "value": 37.8733723866,
            "unit": "iter/sec",
            "range": "stddev: 0.000394269815437",
            "extra": "mean: 26.4037749211 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[True-False]",
            "value": 38.3860695936,
            "unit": "iter/sec",
            "range": "stddev: 0.000126351508208",
            "extra": "mean: 26.0511172565 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-True]",
            "value": 38.127203522,
            "unit": "iter/sec",
            "range": "stddev: 0.000397847172438",
            "extra": "mean: 26.2279922896 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/nn/functional_benchmarks_test.py::test_vmap_transformer_speed_decorator[False-False]",
            "value": 37.8144858757,
            "unit": "iter/sec",
            "range": "stddev: 0.000424407674537",
            "extra": "mean: 26.4448921317 msec\nrounds: 38"
          }
        ]
      }
    ]
  }
}